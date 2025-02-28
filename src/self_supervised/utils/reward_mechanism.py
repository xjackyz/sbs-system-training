import os
import logging
import json
import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from .signal_tracker import SignalTracker
from .reward_calculator import RewardCalculator
from ..utils.logger import setup_logger

logger = setup_logger('reward_mechanism')

class RewardMechanism:
    """奖励机制
    
    根据信号跟踪器的数据，为自监督学习提供奖励信号，
    调整模型训练过程，优化模型对SBS序列的理解。
    """
    
    def __init__(self, signal_tracker: SignalTracker, config: Dict = None):
        """初始化奖励机制
        
        Args:
            signal_tracker: 信号跟踪器实例
            config: 奖励机制配置
        """
        self.signal_tracker = signal_tracker
        self.config = config or {}
        
        # 初始化奖励计算器
        self.reward_calculator = RewardCalculator(config)
        
        # 初始化统计
        self.stats = {
            'daily_pnl': 0.0,
            'max_daily_loss_hit': False,
            'benchmark_samples': set()  # 记录基准样本ID
        }
        
    def calculate_reward(self, llava_output: Dict, trade_result: Dict) -> float:
        """计算奖励值
        
        Args:
            llava_output: LLaVA模型的输出结果
            trade_result: 交易结果
            
        Returns:
            float: 奖励值
        """
        try:
            # 检查是否为基准样本
            sample_id = trade_result.get('sample_id')
            is_benchmark = sample_id in self.stats['benchmark_samples']
            
            # 从LLaVA输出中获取是否存在SBS序列
            has_sbs = llava_output.get('has_sbs_sequence', False)
            
            # 使用奖励计算器计算奖励
            reward = self.reward_calculator.calculate_reward(
                trade_result=trade_result,
                has_sbs_sequence=has_sbs,
                is_benchmark_sample=is_benchmark
            )
            
            return reward
            
        except Exception as e:
            logger.error(f"计算奖励时出错: {str(e)}")
            return 0.0
            
    def add_benchmark_sample(self, sample_id: str) -> None:
        """添加基准样本
        
        Args:
            sample_id: 样本ID
        """
        self.stats['benchmark_samples'].add(sample_id)
        
    def remove_benchmark_sample(self, sample_id: str) -> None:
        """移除基准样本
        
        Args:
            sample_id: 样本ID
        """
        self.stats['benchmark_samples'].discard(sample_id)
        
    def reset_daily_stats(self) -> None:
        """重置每日统计"""
        self.reward_calculator.reset_daily_stats()
        self.stats['daily_pnl'] = 0.0
        self.stats['max_daily_loss_hit'] = False
        
    def get_stats(self) -> Dict:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.reward_calculator.get_stats()
        stats.update({
            'benchmark_samples_count': len(self.stats['benchmark_samples'])
        })
        return stats

    def _calculate_profit_score(self, trade_result: Dict) -> float:
        """计算盈利得分
        
        Args:
            trade_result: 交易结果
            
        Returns:
            float: 盈利得分
        """
        try:
            # 计算盈亏
            pnl = trade_result.get('pnl', 0)
            
            # 更新每日盈亏
            self.stats['daily_pnl'] += pnl
            daily_pnl = self.stats['daily_pnl']
            
            # 计算基础得分
            if pnl >= 0:
                # 盈利情况
                base_score = min(pnl / self.config['daily_profit_target'], 1.0)
                
                # 超额盈利奖励
                if daily_pnl > self.config['daily_profit_target']:
                    excess_pnl = daily_pnl - self.config['daily_profit_target']
                    excess_score = excess_pnl / self.config['daily_profit_target'] * 0.2
                    base_score += excess_score
            else:
                # 亏损情况
                base_score = max(-1.0, pnl / abs(self.config['max_daily_loss']))
            
            # 检查是否触发止损
            if daily_pnl <= self.config['max_daily_loss']:
                base_score = -1.0
                self.stats['max_daily_loss_hit'] = True
                
            return base_score
            
        except Exception as e:
            logger.error(f"计算盈利得分时出错: {str(e)}")
            return 0.0
            
    def _evaluate_sbs_sequence(self, llava_output: Dict) -> float:
        """评估SBS序列质量
        
        Args:
            llava_output: LLaVA模型的输出结果
            
        Returns:
            float: 序列质量得分
        """
        try:
            score = 0.0
            
            # 检查是否识别到SBS序列
            if not llava_output.get('has_sbs_sequence', False):
                return 0.0
                
            # 1. 关键点位识别 (0.4)
            key_points = llava_output.get('key_points', {})
            if len(key_points) == 5:  # 需要识别全部5个关键点
                score += 0.4
            else:
                score += 0.4 * (len(key_points) / 5)
                
            # 2. 方向准确性 (0.3)
            if llava_output.get('direction_correct', False):
                score += 0.3
                
            # 3. 模型置信度 (0.3)
            confidence = llava_output.get('confidence', 0)
            score += 0.3 * confidence
            
            return score
            
        except Exception as e:
            logger.error(f"评估SBS序列时出错: {str(e)}")
            return 0.0
            
    def _evaluate_risk_control(self, signal_info: Dict) -> float:
        """评估风控执行情况
        
        Args:
            signal_info: 信号信息
            
        Returns:
            float: 风控评估得分
        """
        try:
            score = 0.0
            
            # 1. 止损设置评估 (50%)
            stop_loss = float(signal_info.get('stop_loss', 0))
            entry_price = float(signal_info.get('entry_price', 0))
            stop_distance = abs(stop_loss - entry_price) / entry_price
            
            if self.config['min_stop_distance'] <= stop_distance <= self.config['max_stop_distance']:
                score += 0.5
                
            # 2. 持仓时间评估 (30%)
            duration = signal_info.get('duration_minutes', 0)
            if duration <= self.config['max_trade_duration']:
                time_score = 1 - (duration / self.config['max_trade_duration'])
                score += time_score * 0.3
                
            # 3. 市场条件评估 (20%)
            if self._check_market_conditions(signal_info):
                score += 0.2
                
            return score
            
        except Exception as e:
            self.logger.warning(f"评估风控时出错: {e}")
            return 0.0
    
    def _check_market_conditions(self, signal_info: Dict) -> bool:
        """检查市场条件
        
        Args:
            signal_info: 信号信息
            
        Returns:
            bool: 是否满足市场条件
        """
        # 获取当前时间(格式: "HH:MM")
        current_time = signal_info.get('timestamp', '').split(' ')[1]
        
        # 避开高波动时段
        if "09:00" <= current_time <= "09:30":  # 美股开盘前
            return False
        if "14:00" <= current_time <= "14:30":  # 欧美交接班
            return False
            
        # 避开低波动时段
        if "12:00" <= current_time <= "13:30":  # 午间
            return False
            
        # 不在尾盘交易
        if current_time >= "15:30":
            return False
            
        # 检查波动率
        volatility = signal_info.get('market_volatility', 1.0)
        if not (0.5 <= volatility <= 2.0):
            return False
            
        # 检查交易量
        volume = signal_info.get('volume', 0)
        avg_volume = signal_info.get('average_volume', 1)
        if volume < avg_volume * 0.7:  # 交易量太小
            return False
            
        return True
    
    def get_trade_stats(self) -> Dict:
        """获取交易统计
        
        Returns:
            Dict: 交易统计字典
        """
        stats = self.stats.copy()
        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
        else:
            stats['win_rate'] = 0.0
        return stats
    
    def calculate_sample_weights(self, min_confidence: float = None) -> Tuple[List[Dict], np.ndarray]:
        """计算训练样本的权重
        
        根据信号的奖励值计算样本权重，用于训练时的加权采样
        
        Args:
            min_confidence: 最小置信度阈值，低于该阈值的样本将被排除
            
        Returns:
            samples: 训练样本列表
            weights: 对应的样本权重数组
        """
        if min_confidence is None:
            min_confidence = self.config['min_confidence']
        
        # 获取训练数据
        chart_data_list = []
        rewards = []

        # 获取所有已完成的信号及其状态
        signals = self.signal_tracker.get_completed_signals()
        
        # 遍历所有信号，计算奖励
        for signal_id, signal_info in signals.items():
            # 检查置信度是否达到阈值
            confidence = signal_info.get('confidence', 0.0)
            if confidence < min_confidence:
                continue
            
            # 计算信号的奖励
            reward = self.calculate_reward(signal_info, {})
            
            # 添加到数据列表
            chart_data_list.append(signal_info)
            rewards.append(reward)
        
        # 如果没有数据，返回空
        if not chart_data_list:
            return [], np.array([])
        
        # 将奖励值转换为样本权重
        weights = np.array(rewards)
        
        # 确保权重为正值
        weights = np.abs(weights) + 1.0  # 加1确保所有样本都有权重
        
        # 归一化权重
        weights = weights / np.sum(weights)
        
        return chart_data_list, weights
    
    def get_training_data(self, validation_dates: Optional[Tuple[str, str]] = None) -> Tuple[List[Dict], np.ndarray]:
        """
        获取训练数据
        
        Args:
            validation_dates: 验证集日期范围 (开始日期, 结束日期)，如果提供，这个范围内的数据将被排除
            
        Returns:
            训练样本和对应权重的元组
        """
        # 获取所有样本和权重
        all_samples, all_weights = self.calculate_sample_weights()
        
        # 如果没有指定验证集日期范围，返回所有数据
        if validation_dates is None:
            return all_samples, all_weights
        
        # 过滤掉验证集日期范围内的数据
        val_start, val_end = validation_dates
        
        # 初始化训练集
        train_samples = []
        train_weights = []
        
        # 过滤数据
        for i, sample in enumerate(all_samples):
            timestamp = sample.get('timestamp', '')
            
            # 如果时间戳在验证集范围内，跳过
            if timestamp and val_start <= timestamp <= val_end:
                continue
                
            train_samples.append(sample)
            train_weights.append(all_weights[i])
        
        # 重新归一化权重
        if train_weights:
            train_weights = np.array(train_weights)
            if np.sum(train_weights) > 0:
                train_weights = train_weights / np.sum(train_weights)
        else:
            train_weights = np.array([])
        
        return train_samples, train_weights
    
    def get_validation_data(self, validation_dates: Tuple[str, str]) -> List[Dict]:
        """
        获取验证数据
        
        Args:
            validation_dates: 验证集日期范围 (开始日期, 结束日期)
            
        Returns:
            验证样本列表
        """
        # 获取所有样本
        all_samples, _ = self.calculate_sample_weights(min_confidence=0.0)  # 不过滤置信度，获取所有样本
        
        # 验证集日期范围
        val_start, val_end = validation_dates
        
        # 筛选验证集样本
        validation_samples = []
        
        for sample in all_samples:
            timestamp = sample.get('timestamp', '')
            
            # 如果时间戳在验证集范围内
            if timestamp and val_start <= timestamp <= val_end:
                validation_samples.append(sample)
        
        return validation_samples
    
    def get_weighted_batch(self, batch_size: int = 32) -> Tuple[List, torch.Tensor]:
        """获取加权批次数据
        
        根据样本权重采样批次数据
        
        Args:
            batch_size: 批次大小
            
        Returns:
            批次数据和对应的伪标签
        """
        samples, weights = self.calculate_sample_weights()
        
        if not samples:
            return [], torch.tensor([])
        
        # 加权采样
        indices = np.random.choice(len(samples), size=min(batch_size, len(samples)), p=weights, replace=True)
        
        # 提取批次数据
        batch_samples = [samples[i] for i in indices]
        
        # 生成伪标签
        pseudo_labels = self._generate_pseudo_labels(batch_samples)
        
        return batch_samples, pseudo_labels
    
    def _generate_pseudo_labels(self, samples: List[Dict]) -> torch.Tensor:
        """生成伪标签
        
        根据样本信息生成伪标签
        
        Args:
            samples: 样本列表
            
        Returns:
            伪标签张量
        """
        pseudo_labels = []
        
        for sample in samples:
            # 获取标签信息
            label = sample.get('label', 'neutral')
            confidence = sample.get('confidence', 0.0)
            
            # 生成伪标签向量 [bullish, bearish, neutral]
            if label == 'bullish':
                pseudo_label = [confidence, 0.0, 1.0 - confidence]
            elif label == 'bearish':
                pseudo_label = [0.0, confidence, 1.0 - confidence]
            else:
                pseudo_label = [0.0, 0.0, 1.0]
            
            pseudo_labels.append(pseudo_label)
        
        return torch.tensor(pseudo_labels, dtype=torch.float32)
    
    def visualize_rewards(self, save_path: str = 'logs/rewards_distribution.png'):
        """可视化奖励分布
        
        Args:
            save_path: 图表保存路径
        """
        import matplotlib.pyplot as plt
        
        # 获取所有完成的信号
        signals = self.signal_tracker.get_completed_signals()
        
        # 计算每个信号的奖励
        rewards = []
        signal_types = []
        
        for signal_id, signal_info in signals.items():
            reward = self.calculate_reward(signal_info, {})
            signal_type = signal_info.get('label', 'neutral')
            
            rewards.append(reward)
            signal_types.append(signal_type)
        
        # 绘制奖励分布
        plt.figure(figsize=(12, 8))
        
        # 绘制总体分布
        plt.subplot(2, 1, 1)
        plt.hist(rewards, bins=30, alpha=0.7)
        plt.title('奖励分布')
        plt.xlabel('奖励值')
        plt.ylabel('频次')
        plt.grid(True)
        
        # 按信号类型分组
        plt.subplot(2, 1, 2)
        for signal_type in ['bullish', 'bearish', 'neutral']:
            type_rewards = [r for r, t in zip(rewards, signal_types) if t == signal_type]
            if type_rewards:
                plt.hist(type_rewards, bins=20, alpha=0.6, label=signal_type)
        
        plt.title('不同信号类型的奖励分布')
        plt.xlabel('奖励值')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"奖励分布图表已保存至 {save_path}")
        
        return save_path

    def weighted_loss_function(self, original_loss_fn):
        """使用奖励权重包装原始损失函数"""
        def weighted_loss(*args, **kwargs):
            loss = original_loss_fn(*args, **kwargs)
            if self.signal_tracker:
                reward = self.signal_tracker.get_latest_reward()
                if reward is not None:
                    weight = np.clip(abs(reward) / 1000, 0.5, 2.0)
                    loss = loss * weight
            return loss
        return weighted_loss

    def apply_rewards(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """应用奖励机制到优化器
        
        Args:
            optimizer: 原始优化器
            
        Returns:
            torch.optim.Optimizer: 应用了奖励的优化器
        """
        # 获取所有信号的奖励
        signals = self.signal_tracker.get_all_signals()
        if not signals:
            return optimizer
        
        # 计算每个信号的奖励
        rewards = [self.calculate_reward(signal, {}) for signal in signals]
        
        # 计算全局奖励因子
        global_reward = np.mean(rewards) if rewards else 1.0
        global_reward = np.clip(global_reward, 
                              self.config.get('min_reward_limit', -1000),
                              self.config.get('max_reward_limit', 2000))
        
        # 调整学习率
        for param_group in optimizer.param_groups:
            original_lr = param_group['lr']
            if global_reward > 0:
                # 如果奖励为正，增加学习率
                param_group['lr'] = original_lr * (1 + global_reward * 0.1)
            else:
                # 如果奖励为负，减小学习率
                param_group['lr'] = original_lr * (1 + global_reward * 0.05)
            
            # 确保学习率不会太大或太小
            param_group['lr'] = np.clip(param_group['lr'], 1e-6, 1e-2)
        
        return optimizer 