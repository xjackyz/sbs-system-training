import os
import json
import uuid
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class SignalTracker:
    """信号跟踪器
    
    用于记录、跟踪和评估交易信号的表现，为奖励机制提供数据支持。
    """
    
    def __init__(self, save_dir: str = 'data/signals', tracking_window: int = 5):
        """初始化信号跟踪器
        
        Args:
            save_dir: 信号数据保存目录
            tracking_window: 信号跟踪窗口（交易日）
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tracking_window = tracking_window
        
        # 信号数据库
        self.signals_db_path = self.save_dir / 'signals_database.json'
        self.signals = self._load_signals_db()
        
        # 统计数据
        self.stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'pending_signals': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
        
        # 更新统计数据
        self._update_stats()
    
    def _load_signals_db(self) -> Dict:
        """加载信号数据库"""
        if self.signals_db_path.exists():
            with open(self.signals_db_path, 'r') as f:
                return json.load(f)
        return {'signals': {}, 'metadata': {'last_updated': str(datetime.datetime.now())}}
    
    def _save_signals_db(self):
        """保存信号数据库"""
        self.signals['metadata']['last_updated'] = str(datetime.datetime.now())
        with open(self.signals_db_path, 'w') as f:
            json.dump(self.signals, f, indent=2)
    
    def _update_stats(self):
        """更新统计数据"""
        signals = self.signals['signals']
        
        # 计数
        self.stats['total_signals'] = len(signals)
        self.stats['successful_signals'] = sum(1 for s in signals.values() if s['status'] == 'success')
        self.stats['failed_signals'] = sum(1 for s in signals.values() if s['status'] == 'failure')
        self.stats['pending_signals'] = sum(1 for s in signals.values() if s['status'] == 'pending')
        
        # 胜率
        if self.stats['total_signals'] - self.stats['pending_signals'] > 0:
            self.stats['win_rate'] = self.stats['successful_signals'] / (self.stats['total_signals'] - self.stats['pending_signals'])
        
        # 平均盈亏
        profits = [s['actual_profit'] for s in signals.values() if s['status'] == 'success']
        losses = [s['actual_profit'] for s in signals.values() if s['status'] == 'failure']
        
        self.stats['avg_profit'] = np.mean(profits) if profits else 0.0
        self.stats['avg_loss'] = np.mean(losses) if losses else 0.0
        
        # 盈亏比
        total_profit = sum(profits) if profits else 0.0
        total_loss = abs(sum(losses)) if losses else 0.0
        
        if total_loss > 0:
            self.stats['profit_factor'] = total_profit / total_loss
    
    def record_signal(self, 
                     chart_data: Dict[str, Any], 
                     model_prediction: Dict[str, Any],
                     confidence: float) -> str:
        """记录新的交易信号
        
        Args:
            chart_data: 图表数据，包含时间、价格等信息
            model_prediction: 模型预测结果，包含点位、信号方向等
            confidence: 模型预测的置信度
            
        Returns:
            signal_id: 生成的信号ID
        """
        # 生成唯一ID
        signal_id = str(uuid.uuid4())
        
        # 当前时间
        timestamp = str(datetime.datetime.now())
        
        # 提取关键信息
        direction = model_prediction.get('direction', 'unknown')  # 多/空
        entry_price = model_prediction.get('entry_price', 0.0)
        stop_loss = model_prediction.get('stop_loss', 0.0)
        target_price = model_prediction.get('target_price', 0.0)
        
        # 计算风险回报比
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0.0
        
        # 创建信号记录
        signal = {
            'id': signal_id,
            'timestamp': timestamp,
            'chart_data': chart_data,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'confidence': confidence,
            'risk_reward_ratio': risk_reward_ratio,
            'status': 'pending'  # 初始状态为待处理
        }
        
        # 计算奖励
        reward_value = self._calculate_reward(signal)
        signal['reward_value'] = reward_value  # 将奖励值与信号关联
        
        # 记录信号
        self.signals['signals'][signal_id] = signal
        self._save_signals_db()  # 保存信号数据库
        
        # 更新统计数据
        self._update_stats()  # 更新统计数据
        
        return signal_id
    
    def _calculate_reward(self, signal: Dict[str, Any]) -> float:
        """计算信号的奖励值
        
        Args:
            signal: 信号字典
        
        Returns:
            奖励值
        """
        # 假设奖励值与风险回报比成正比
        return signal['risk_reward_ratio'] * 10  # 这里的10是一个示例系数
    
    def update_signal_tracking(self, signal_id: str, price_data: Dict[str, float]):
        """更新信号的跟踪数据
        
        Args:
            signal_id: 信号ID
            price_data: 价格数据，包含最高价、最低价、收盘价等
        """
        if signal_id not in self.signals['signals']:
            raise ValueError(f"信号ID {signal_id} 不存在")
        
        signal = self.signals['signals'][signal_id]
        
        # 如果信号已经完成评估，不再更新
        if signal['status'] != 'pending':
            return
        
        # 添加跟踪数据
        tracking_entry = {
            'timestamp': str(datetime.datetime.now()),
            'price_data': price_data
        }
        
        signal['tracking_data'].append(tracking_entry)
        
        # 检查是否达到止盈或止损
        high_price = price_data.get('high', 0.0)
        low_price = price_data.get('low', 0.0)
        
        # 多头信号
        if signal['direction'] == 'long':
            # 检查是否触及止损
            if low_price <= signal['stop_loss']:
                self._evaluate_signal(signal_id, 'failure', signal['stop_loss'])
                return
            
            # 检查是否达到目标价
            if high_price >= signal['target_price']:
                self._evaluate_signal(signal_id, 'success', signal['target_price'])
                return
        
        # 空头信号
        elif signal['direction'] == 'short':
            # 检查是否触及止损
            if high_price >= signal['stop_loss']:
                self._evaluate_signal(signal_id, 'failure', signal['stop_loss'])
                return
            
            # 检查是否达到目标价
            if low_price <= signal['target_price']:
                self._evaluate_signal(signal_id, 'success', signal['target_price'])
                return
        
        # 检查是否超过跟踪窗口
        if len(signal['tracking_data']) >= self.tracking_window:
            # 使用最后一个价格作为退出价格
            exit_price = price_data.get('close', 0.0)
            
            # 计算盈亏
            if signal['direction'] == 'long':
                profit = exit_price - signal['entry_price']
            else:  # short
                profit = signal['entry_price'] - exit_price
            
            # 确定状态
            status = 'success' if profit > 0 else 'failure'
            
            self._evaluate_signal(signal_id, status, exit_price)
        
        # 保存数据库
        self._save_signals_db()
    
    def _evaluate_signal(self, signal_id: str, status: str, exit_price: float):
        """评估信号的表现
        
        Args:
            signal_id: 信号ID
            status: 信号状态 (success/failure)
            exit_price: 退出价格
        """
        signal = self.signals['signals'][signal_id]
        
        # 更新状态
        signal['status'] = status
        signal['actual_exit'] = exit_price
        signal['evaluation_timestamp'] = str(datetime.datetime.now())
        
        # 计算实际盈亏
        if signal['direction'] == 'long':
            profit = exit_price - signal['entry_price']
        else:  # short
            profit = signal['entry_price'] - exit_price
        
        signal['actual_profit'] = profit
        
        # 计算奖励值
        if status == 'success':
            # 成功信号给予正奖励
            reward = 1.0
            
            # 根据风险回报比调整奖励
            if signal['risk_reward_ratio'] > 2.0:
                reward *= 1.5
            
            # 根据置信度调整奖励
            reward *= signal['confidence']
            
        else:  # failure
            # 失败信号给予负奖励
            reward = -1.0
            
            # 根据风险回报比调整奖励（风险回报比高的失败信号惩罚更轻）
            if signal['risk_reward_ratio'] > 2.0:
                reward *= 0.7
            
            # 根据置信度调整奖励（高置信度的失败信号惩罚更重）
            reward *= signal['confidence']
        
        signal['reward_value'] = reward
        
        # 更新统计数据
        self._update_stats()
    
    def get_signal(self, signal_id: str) -> Dict:
        """获取信号详情"""
        if signal_id not in self.signals['signals']:
            raise ValueError(f"信号ID {signal_id} 不存在")
        
        return self.signals['signals'][signal_id]
    
    def get_stats(self) -> Dict:
        """获取统计数据"""
        return self.stats
    
    def get_training_data(self, min_confidence: float = 0.0) -> Tuple[List, List]:
        """获取用于训练的数据和奖励
        
        Args:
            min_confidence: 最小置信度阈值，只返回置信度高于此值的信号
            
        Returns:
            chart_data_list: 图表数据列表
            rewards: 对应的奖励值列表
        """
        chart_data_list = []
        rewards = []
        
        for signal in self.signals['signals'].values():
            # 只使用已评估的信号
            if signal['status'] == 'pending':
                continue
                
            # 应用置信度过滤
            if signal['confidence'] < min_confidence:
                continue
                
            chart_data_list.append(signal['chart_data'])
            rewards.append(signal['reward_value'])
        
        return chart_data_list, rewards
    
    def export_to_csv(self, filepath: str = None) -> str:
        """导出信号数据到CSV文件
        
        Args:
            filepath: 导出文件路径，如果为None则使用默认路径
            
        Returns:
            导出文件的路径
        """
        if filepath is None:
            filepath = self.save_dir / f"signals_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 准备数据
        data = []
        for signal_id, signal in self.signals['signals'].items():
            row = {
                'id': signal_id,
                'timestamp': signal['timestamp'],
                'direction': signal['direction'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'target_price': signal['target_price'],
                'risk_reward_ratio': signal['risk_reward_ratio'],
                'confidence': signal['confidence'],
                'status': signal['status'],
                'actual_exit': signal['actual_exit'],
                'actual_profit': signal['actual_profit'],
                'reward_value': signal['reward_value']
            }
            data.append(row)
        
        # 创建DataFrame并导出
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        return filepath

    def get_all_signals(self) -> Dict[str, Dict]:
        """获取所有信号
        
        Returns:
            Dict[str, Dict]: 信号ID到信号数据的映射
        """
        return self.signals

    def get_completed_signals(self) -> Dict[str, Dict]:
        """获取所有已完成的信号
        
        Returns:
            Dict[str, Dict]: 已完成信号的ID到信号数据的映射
        """
        return {
            signal_id: signal_data 
            for signal_id, signal_data in self.signals.items()
            if signal_data.get('status') in ['completed', 'closed']
        } 