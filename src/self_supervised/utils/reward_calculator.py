"""
奖励计算器
用于计算模型预测的奖励，整合人工标注反馈
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import f1_score

from ..utils.logger import setup_logger

logger = setup_logger('reward_calculator')

class RewardCalculator:
    """奖励计算器类"""
    
    def __init__(self, config: Dict = None, alpha: float = 1.0, beta: float = 1.0):
        """
        初始化奖励计算器
        
        Args:
            config: 配置参数
            alpha: 盈利权重
            beta: 最大回撤权重
        """
        self.config = config or {}
        self.logger = setup_logger('reward_calculator')
        self.human_labels = {}  # 存储人工标注数据
        self.model_predictions = {}  # 存储模型预测
        self.reward_history = []  # 存储奖励历史
        self.alpha = alpha
        self.beta = beta
        
        # 奖励权重配置
        self.weights = {
            'accuracy': {
                'initial': 0.7,  # 初始准确性权重
                'final': 0.4,    # 最终准确性权重
                'transition_steps': 1000  # 权重过渡步数
            },
            'profit': {
                'initial': 0.3,  # 初始利润权重
                'final': 0.6     # 最终利润权重
            },
            'penalties': {
                'false_positive': 0.1,  # 虚假信号惩罚
                'false_negative': 0.1,  # 错过信号惩罚
                'timing_error': 0.05    # 时间点偏差惩罚
            }
        }
        
    def calculate_reward(self, 
                        prediction: Dict,
                        human_label: Dict,
                        trade_result: Optional[Dict] = None,
                        training_step: int = 0) -> Tuple[float, Dict]:
        """
        计算奖励值
        
        Args:
            prediction: 模型预测结果
            human_label: 人工标注数据
            trade_result: 交易结果（可选）
            training_step: 当前训练步数
            
        Returns:
            奖励值和详细信息
        """
        try:
            # 计算动态权重
            accuracy_weight = self._calculate_dynamic_weight(
                self.weights['accuracy']['initial'],
                self.weights['accuracy']['final'],
                training_step / self.weights['accuracy']['transition_steps']
            )
            profit_weight = 1 - accuracy_weight
            
            # 计算准确性奖励
            accuracy_reward = self._calculate_accuracy_reward(prediction, human_label)
            
            # 计算利润奖励
            profit_reward = self._calculate_profit_reward(trade_result) if trade_result else 0.0
            
            # 应用惩罚
            penalties = self._apply_penalties(prediction, human_label)
            
            # 计算总奖励
            total_reward = (
                accuracy_weight * accuracy_reward +
                profit_weight * profit_reward -
                sum(penalties.values())
            )
            
            # 计算复合奖励
            compound_reward = self.calculate_compound_reward(profit_reward, max_drawdown)
            
            # 记录详细信息
            details = {
                'accuracy_reward': accuracy_reward,
                'profit_reward': profit_reward,
                'compound_reward': compound_reward,
                'penalties': penalties,
                'weights': {
                    'accuracy': accuracy_weight,
                    'profit': profit_weight
                },
                'total_reward': total_reward
            }
            
            # 更新历史
            self._update_history(details)
            
            return total_reward, details
            
        except Exception as e:
            logger.error(f"计算奖励失败: {str(e)}")
            raise
            
    def _calculate_dynamic_weight(self,
                                initial: float,
                                final: float,
                                progress: float) -> float:
        """
        计算动态权重
        
        Args:
            initial: 初始权重
            final: 最终权重
            progress: 训练进度 (0-1)
            
        Returns:
            当前权重值
        """
        # 使用平滑过渡函数
        return initial + (final - initial) * (3 * progress ** 2 - 2 * progress ** 3)
        
    def _calculate_accuracy_reward(self, prediction: Dict, human_label: Dict) -> float:
        """计算准确性奖励"""
        try:
            total_score = 0.0
            weights = {
                'point1': 0.15,  # 第一次回调
                'point2': 0.20,  # 极值点
                'point3': 0.25,  # 流动性获取
                'point4': 0.25,  # 确认点
                'point5': 0.15   # 目标点
            }
            
            for point, weight in weights.items():
                if point in prediction and point in human_label:
                    # 计算位置误差
                    error = abs(prediction[point] - human_label[point])
                    max_error = 5  # 允许的最大误差
                    
                    # 计算点位得分
                    if error == 0:
                        score = 1.0
                    elif error <= max_error:
                        score = 1 - (error / max_error)
                    else:
                        score = 0
                        
                    total_score += weight * score
                    
            return total_score
            
        except Exception as e:
            logger.error(f"计算准确性奖励失败: {str(e)}")
            return 0.0
            
    def _calculate_profit_reward(self, trade_result: Dict) -> float:
        """计算利润奖励"""
        try:
            if not trade_result:
                return 0.0
                
            profit = trade_result.get('profit', 0)
            risk = trade_result.get('risk', 1)  # 避免除以零
            
            # 计算风险调整后的收益
            risk_adjusted_return = profit / risk
            
            # 根据风险收益比计算奖励
            if risk_adjusted_return >= 2.0:  # 优秀的风险收益比
                return 1.0
            elif risk_adjusted_return >= 1.0:  # 良好的风险收益比
                return 0.7
            elif risk_adjusted_return > 0:  # 正收益
                return 0.3
            else:  # 负收益
                return 0.0
                
        except Exception as e:
            logger.error(f"计算利润奖励失败: {str(e)}")
            return 0.0
            
    def _apply_penalties(self, prediction: Dict, human_label: Dict) -> Dict:
        """应用惩罚机制"""
        try:
            penalties = {}
            
            # 虚假信号惩罚
            if not human_label and prediction:
                penalties['false_positive'] = self.weights['penalties']['false_positive']
                
            # 错过信号惩罚
            if human_label and not prediction:
                penalties['false_negative'] = self.weights['penalties']['false_negative']
                
            # 时间点偏差惩罚
            timing_errors = []
            for point in ['point1', 'point2', 'point3', 'point4', 'point5']:
                if point in prediction and point in human_label:
                    error = abs(prediction[point] - human_label[point])
                    if error > 5:  # 超过5个K线的偏差
                        timing_errors.append(error)
                        
            if timing_errors:
                penalties['timing_error'] = (
                    self.weights['penalties']['timing_error'] *
                    (sum(timing_errors) / len(timing_errors)) / 10  # 归一化
                )
                
            return penalties
            
        except Exception as e:
            logger.error(f"应用惩罚失败: {str(e)}")
            return {}
            
    def add_human_label(self, sample_id: str, label_data: Dict) -> None:
        """添加人工标注数据"""
        self.human_labels[sample_id] = label_data
        
    def add_model_prediction(self, sample_id: str, prediction: Dict) -> None:
        """添加模型预测结果"""
        self.model_predictions[sample_id] = prediction
        
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        try:
            if not self.reward_history:
                return {}
                
            stats = {
                'total_samples': len(self.reward_history),
                'average_reward': np.mean([r['total_reward'] for r in self.reward_history]),
                'accuracy_trend': self._calculate_trend([r['accuracy_reward'] for r in self.reward_history]),
                'profit_trend': self._calculate_trend([r['profit_reward'] for r in self.reward_history]),
                'penalty_distribution': self._analyze_penalties()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取训练统计失败: {str(e)}")
            return {}
            
    def _calculate_trend(self, values: List[float], window: int = 100) -> float:
        """计算趋势"""
        if len(values) < window:
            return 0.0
            
        recent = np.mean(values[-window:])
        previous = np.mean(values[-2*window:-window])
        return (recent - previous) / previous if previous != 0 else 0.0
        
    def _analyze_penalties(self) -> Dict:
        """分析惩罚分布"""
        penalties = {
            'false_positive': 0,
            'false_negative': 0,
            'timing_error': 0
        }
        
        for record in self.reward_history:
            for penalty_type, value in record['penalties'].items():
                penalties[penalty_type] += 1
                
        total = len(self.reward_history)
        return {k: v/total if total > 0 else 0 for k, v in penalties.items()}
        
    def _update_history(self, details: Dict) -> None:
        """更新奖励历史"""
        self.reward_history.append(details)
        
        # 保持历史记录在合理范围内
        max_history = 10000
        if len(self.reward_history) > max_history:
            self.reward_history = self.reward_history[-max_history:]
            
    def save_state(self, filepath: str) -> None:
        """保存当前状态"""
        try:
            state = {
                'weights': self.weights,
                'reward_history': self.reward_history[-1000:],  # 只保存最近1000条记录
                'statistics': self.get_training_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"状态已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存状态失败: {str(e)}")
            raise
            
    def load_state(self, filepath: str) -> None:
        """加载状态"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.weights = state['weights']
            self.reward_history = state['reward_history']
            
            logger.info(f"已从 {filepath} 加载状态")
            
        except Exception as e:
            logger.error(f"加载状态失败: {str(e)}")
            raise

    def save_stats(self, save_dir: str) -> None:
        """
        保存统计信息
        
        Args:
            save_dir: 保存目录
        """
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 准备统计数据
            stats_data = {
                'mean_reward': float(np.mean(self.reward_history)),
                'total_trades': len(self.reward_history),
                'false_positive_rate': self._analyze_penalties()['false_positive'],
                'false_negative_rate': self._analyze_penalties()['false_negative'],
                'current_accuracy_weight': float(self.weights['accuracy']['initial']),
                'current_profit_weight': float(self.weights['profit']['initial']),
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存到文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stats_file = save_path / f'reward_stats_{timestamp}.json'
            
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
                
            self.logger.info(f"已保存奖励统计信息到: {stats_file}")
            
        except Exception as e:
            self.logger.error(f"保存统计信息时发生错误: {str(e)}")
            raise
            
    def get_current_stats(self) -> Dict:
        """
        获取当前统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'mean_reward': float(np.mean(self.reward_history)),
            'total_trades': len(self.reward_history),
            'false_positive_rate': self._analyze_penalties()['false_positive'],
            'false_negative_rate': self._analyze_penalties()['false_negative'],
            'training_progress': float(len(self.reward_history) / self.weights['accuracy']['transition_steps'])
        }

    def calculate_f1_score(self, true_labels: List[int], predicted_labels: List[int]) -> float:
        """
        计算F1分数
        
        Args:
            true_labels: 真实标签列表
            predicted_labels: 预测标签列表
        
        Returns:
            F1分数
        """
        return f1_score(true_labels, predicted_labels, average='weighted')

    def calculate_compound_reward(self, profit: float, max_drawdown: float) -> float:
        """
        计算复合奖励
        
        Args:
            profit: 盈利
            max_drawdown: 最大回撤
            
        Returns:
            复合奖励
        """
        return self.alpha * profit - self.beta * max_drawdown 