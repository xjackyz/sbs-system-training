#!/usr/bin/env python
"""
SBS奖励计算器 - 增强版
用于计算模型预测的奖励值，支持多种奖励函数及动态调整
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime, timedelta
import math
import random
from sklearn.metrics import f1_score
import copy

from ..utils.logger import setup_logger

logger = setup_logger('sbs_reward_calculator')

class SBSRewardCalculator:
    """SBS奖励计算器，用于计算模型预测的奖励值，支持动态调整和探索机制"""
    
    def __init__(self, config: Dict = None):
        """
        初始化奖励计算器
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        
        # 基础配置
        self.reward_type = self.config.get('reward_type', 'combined')
        self.point_value = float(self.config.get('point_value', 20.0))  # 每点价值
        
        # 奖励权重
        self.profit_multiplier = float(self.config.get('profit_multiplier', 1.0))
        self.accuracy_weight = float(self.config.get('accuracy_weight', 0.5))
        self.sequence_weight = float(self.config.get('sequence_weight', 0.3))
        
        # 风险调整
        self.drawdown_penalty = float(self.config.get('drawdown_penalty', 0.2))
        self.risk_reward_weight = float(self.config.get('risk_reward_weight', 0.5))
        
        # 市场条件调整
        self.market_condition_weight = float(self.config.get('market_condition_weight', 0.2))  # 统一的市场条件权重
        self.volatility_influence = float(self.config.get('volatility_influence', 0.5))  # 波动率在市场条件中的影响比例
        self.trend_influence = float(self.config.get('trend_influence', 0.5))  # 趋势在市场条件中的影响比例
        
        # 时间因素
        self.time_decay_factor = float(self.config.get('time_decay_factor', 20.0))
        
        # 交易特性调整
        self.trade_frequency_penalty = float(self.config.get('trade_frequency_penalty', 0.05))
        self.long_win_bonus = float(self.config.get('long_win_bonus', 0.1))
        self.short_win_bonus = float(self.config.get('short_win_bonus', 0.1))
        self.consecutive_win_bonus = float(self.config.get('consecutive_win_bonus', 0.05))
        
        # 探索机制
        self.exploration_rate = float(self.config.get('exploration_rate', 0.1))  # ε-greedy探索率
        self.exploration_decay = float(self.config.get('exploration_decay', 0.995))  # 探索率衰减
        self.min_exploration_rate = float(self.config.get('min_exploration_rate', 0.01))  # 最小探索率
        
        # 奖励限制
        self.min_reward = float(self.config.get('min_reward', -1000.0))
        self.max_reward = float(self.config.get('max_reward', 2000.0))
        
        # 状态跟踪
        self.trades_history = []
        self.win_streak = 0
        self.loss_streak = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.exploration_count = 0
        
        # 动态调整记录
        self.adjustment_history = []
        
        logger.info(f"增强版奖励计算器初始化完成，使用奖励类型: {self.reward_type}")
    
    def calculate(self, 
                 prediction: np.ndarray = None, 
                 ground_truth: np.ndarray = None, 
                 market_data: Dict = None,
                 trade_result: Dict = None,
                 allow_exploration: bool = True) -> float:
        """计算奖励值"""
        try:
            # 输入验证
            if prediction is None and trade_result is None:
                logger.warning("预测结果和交易结果均为空，无法计算奖励")
                return 0.0

            # 根据奖励类型计算
            reward = self._calculate_reward(
                prediction=prediction,
                ground_truth=ground_truth,
                market_data=market_data,
                trade_result=trade_result
            )

            # 应用奖励限制
            reward = np.clip(reward, self.min_reward, self.max_reward)
            
            return reward
            
        except Exception as e:
            logger.error(f"计算奖励时发生异常: {e}")
            # 返回默认奖励值
            return 0.0
    
    def _calculate_reward(self,
                        prediction: np.ndarray, 
                        ground_truth: np.ndarray, 
                        market_data: Dict = None,
                        trade_result: Dict = None) -> float:
        """计算奖励值核心方法"""
        # 根据奖励类型选择计算方法
        if self.reward_type == 'profit':
            reward = self._calculate_profit_reward(prediction, ground_truth, trade_result)
        elif self.reward_type == 'accuracy':
            reward = self._calculate_accuracy_reward(prediction, ground_truth)
        elif self.reward_type == 'sequence':
            reward = self._calculate_sequence_reward(prediction, ground_truth)
        elif self.reward_type == 'combined':
            reward = self._calculate_combined_reward(prediction, ground_truth, market_data, trade_result)
        elif self.reward_type == 'market_relative':
            reward = self._calculate_market_relative_reward(prediction, ground_truth, market_data, trade_result)
        else:
            logger.warning(f"未知的奖励类型: {self.reward_type}，使用组合奖励")
            reward = self._calculate_combined_reward(prediction, ground_truth, market_data, trade_result)
        
        # 应用奖励限制
        reward = np.clip(reward, self.min_reward, self.max_reward)
        
        return reward
    
    def _calculate_profit_reward(self, 
                              prediction: np.ndarray, 
                              ground_truth: np.ndarray, 
                              trade_result: Dict = None) -> float:
        """
        根据交易利润计算奖励
        
        参数:
            prediction: 模型预测结果
            ground_truth: 真实标签
            trade_result: 交易结果
            
        返回:
            奖励值
        """
        if not trade_result:
            logger.warning("未提供交易结果，无法计算利润奖励")
            return 0.0
            
        try:
            # 从交易结果中获取数据
            entry_price = trade_result.get('entry_price', 0.0)
            exit_price = trade_result.get('exit_price', 0.0)
            position_type = trade_result.get('direction', '多')  # '多'或'空'
            stop_loss = trade_result.get('stop_loss', 0.0)
            
            # 输入验证
            if not isinstance(entry_price, (int, float)) or entry_price <= 0:
                logger.warning(f"无效的入场价格: {entry_price}")
                return 0.0
                
            if not isinstance(exit_price, (int, float)) or exit_price <= 0:
                logger.warning(f"无效的出场价格: {exit_price}")
                return 0.0
                
            if position_type not in ['多', '空']:
                logger.warning(f"无效的仓位类型: {position_type}，使用默认值'多'")
                position_type = '多'
            
            # 计算价格差异
            price_diff = exit_price - entry_price
            if position_type == '空':
                price_diff = -price_diff
            
            # 基础美元收益
            base_reward = price_diff * self.point_value
            
            # 计算风险收益比
            if stop_loss != 0.0 and entry_price != 0.0:
                risk = abs(stop_loss - entry_price)
                risk_reward_ratio = abs(price_diff) / risk if risk > 0 else 1.0
            else:
                risk_reward_ratio = 1.0
            
            # 应用风险收益比调整
            rr_multiplier = min(risk_reward_ratio * self.risk_reward_weight, 2.0)
            
            # 应用时间衰减
            duration = trade_result.get('duration_minutes', 0)
            time_multiplier = np.exp(-duration / self.time_decay_factor) if duration > 0 else 1.0
            
            # 应用市场波动率调整
            volatility = trade_result.get('market_volatility', 1.0)
            vol_multiplier = 1.0 + (volatility - 1.0) * self.volatility_influence if volatility > 0 else 1.0
            
            # 计算最终奖励
            final_reward = base_reward * self.profit_multiplier * rr_multiplier * time_multiplier * vol_multiplier
            
            # 连续获胜奖励
            if price_diff > 0 and self.win_streak > 2:
                final_reward *= (1.0 + self.consecutive_win_bonus * min(self.win_streak / 5, 1.0))
            
            # 交易方向奖励
            if price_diff > 0:
                if position_type == '多':
                    final_reward *= (1.0 + self.long_win_bonus)
                else:
                    final_reward *= (1.0 + self.short_win_bonus)
            
            logger.debug(f"计算利润奖励: {final_reward:.2f} (基础:{base_reward:.2f}, 乘数:{self.profit_multiplier:.2f}, 风险比:{rr_multiplier:.2f}, 时间:{time_multiplier:.2f}, 波动:{vol_multiplier:.2f})")
            
            return final_reward
            
        except Exception as e:
            logger.error(f"计算利润奖励时发生异常: {e}")
            return 0.0
    
    def _calculate_accuracy_reward(self, 
                                  prediction: np.ndarray, 
                                  ground_truth: np.ndarray) -> float:
        """
        根据预测准确性计算奖励
        
        参数:
            prediction: 模型预测结果
            ground_truth: 真实标签
            
        返回:
            奖励值
        """
        if ground_truth is None:
            logger.warning("未提供真实标签，无法计算准确度奖励")
            return 0.0
        
        # 假设预测格式：[点1_x, 点1_y, ..., 点5_x, 点5_y, 方向]
        # 提取方向预测
        if len(prediction.shape) > 1 and prediction.shape[-1] > 1:
            # 多分类情况
            pred_direction = np.argmax(prediction[-1].reshape(-1, 2), axis=1)[0]
            true_direction = np.argmax(ground_truth[-1].reshape(-1, 2), axis=1)[0]
        else:
            # 二分类情况
            pred_direction = 1 if prediction[-1] > 0.5 else 0
            true_direction = 1 if ground_truth[-1] > 0.5 else 0
            
        # 方向预测正确给予奖励，错误给予惩罚
        if pred_direction == true_direction:
            reward = 1.0 * self.accuracy_weight
        else:
            reward = -1.0 * self.accuracy_weight
            
        logger.debug(f"计算准确度奖励: 预测={pred_direction}, 真实={true_direction}, 奖励={reward}")
            
        return reward
    
    def _calculate_sequence_reward(self, 
                                  prediction: np.ndarray, 
                                  ground_truth: np.ndarray) -> float:
        """
        根据SBS序列点预测的准确性计算奖励
        
        参数:
            prediction: 模型预测结果
            ground_truth: 真实标签
            
        返回:
            奖励值
        """
        if ground_truth is None:
            logger.warning("未提供真实标签，无法计算序列奖励")
            return 0.0
        
        # 提取序列点预测（不包括方向）
        sequence_preds = prediction[:-1]
        sequence_truth = ground_truth[:-1]
        
        # 计算均方误差
        mse = np.mean(np.square(sequence_preds - sequence_truth))
        
        # 计算归一化的奖励值（误差越小，奖励越大）
        # 使用指数衰减函数将MSE转换为0-1之间的奖励
        reward = math.exp(-mse) * self.sequence_weight
        
        logger.debug(f"计算序列奖励: MSE={mse}, 奖励={reward}")
        
        return reward
    
    def _calculate_market_condition_adjustment(self, 
                                             market_data: Dict = None,
                                             trade_result: Dict = None,
                                             position_type: str = None) -> float:
        """
        计算市场条件综合调整因子
        
        参数:
            market_data: 市场数据
            trade_result: 交易结果
            position_type: 仓位类型（多/空）
            
        返回:
            市场条件调整因子
        """
        if market_data is None:
            return 1.0  # 默认为不调整
            
        # 初始化调整因子
        volatility_factor = 1.0
        trend_factor = 1.0
        
        # 波动率调整
        if 'volatility' in market_data:
            volatility = market_data['volatility']
            # 波动率越高，奖励越大（波动性好的交易需要更多奖励）
            volatility_factor = 1.0 + max(0, (volatility - 1.0)) * self.volatility_influence
            
        # 趋势调整
        if 'market_trend' in market_data and position_type:
            trend = market_data['market_trend']  # 假设 1 为上升趋势，-1 为下降趋势，0 为盘整
            
            # 趋势与交易方向一致性
            trend_aligned = False
            if (position_type == '多' and trend > 0) or (position_type == '空' and trend < 0):
                trend_aligned = True
                
            # 如果方向一致，增加奖励；反之，减少奖励
            if trend_aligned:
                trend_factor = 1.0 + abs(trend) * self.trend_influence
        else:
                trend_factor = 1.0 - abs(trend) * self.trend_influence * 0.5  # 惩罚轻一些
        
        # 组合成最终的市场条件调整因子，权重决定对最终奖励的影响程度
        combined_factor = 1.0 + (((volatility_factor + trend_factor) / 2.0) - 1.0) * self.market_condition_weight
        )
        
        logger.debug(f"市场条件调整因子: {combined_factor:.4f} (波动率: {volatility_factor:.4f}, 趋势: {trend_factor:.4f})")
        
        return combined_factor
    
    def _calculate_combined_reward(self,
                                  prediction: np.ndarray, 
                                  ground_truth: np.ndarray, 
                                  market_data: Dict = None, 
                                  trade_result: Dict = None) -> float:
        """
        组合多种奖励计算方法，动态调整权重
        
        参数:
            prediction: 模型预测结果
            ground_truth: 真实标签
            market_data: 市场数据
            trade_result: 交易结果
            
        返回:
            组合奖励值
        """
        # 初始化组件奖励
        profit_reward = 0.0
        accuracy_reward = 0.0
        sequence_reward = 0.0
        
        # 计算利润奖励
        if trade_result:
            profit_reward = self._calculate_profit_reward(prediction, ground_truth, trade_result)
        
        # 计算准确性奖励
        if ground_truth is not None:
            accuracy_reward = self._calculate_accuracy_reward(prediction, ground_truth)
            sequence_reward = self._calculate_sequence_reward(prediction, ground_truth)
        
        # 基础组合奖励
        combined_reward = profit_reward + accuracy_reward + sequence_reward
        
        # 附加回撤惩罚
        if trade_result and trade_result.get('max_drawdown', 0) > 0:
            drawdown = trade_result['max_drawdown']
            drawdown_penalty = -drawdown * self.drawdown_penalty
            combined_reward += drawdown_penalty
            logger.debug(f"应用回撤惩罚: {drawdown_penalty:.2f}")
            
        # 获取交易方向
        position_type = trade_result.get('direction', '多') if trade_result else None
            
        # 市场条件调整（整合波动率和趋势因素）
        market_condition_factor = self._calculate_market_condition_adjustment(
            market_data=market_data,
            trade_result=trade_result,
            position_type=position_type
        )
        
        # 应用市场条件调整
        if market_condition_factor != 1.0:
            market_condition_adjustment = combined_reward * (market_condition_factor - 1.0)
            combined_reward += market_condition_adjustment
            logger.debug(f"应用市场条件调整: {market_condition_adjustment:.2f}")
            
        # 交易频率惩罚
        if len(self.trades_history) > 20:  # 假设我们有足够的交易历史
            trade_interval = self._calculate_trade_interval()
            # 如果获取到有效的交易间隔
            if trade_interval is not None:
                # 使用动态阈值，根据最近20次交易的中位数间隔计算，避免硬编码
                threshold = max(5, min(trade_interval * 0.5, 30))  # 最小5分钟，最大30分钟
                # 如果交易间隔过短，应用惩罚
                if trade_interval < threshold:
                    freq_penalty_factor = max(0, (threshold - trade_interval) / threshold)
                    frequency_penalty = -self.trade_frequency_penalty * freq_penalty_factor
                    combined_reward += frequency_penalty
                    logger.debug(f"应用交易频率惩罚: {frequency_penalty:.2f} (间隔={trade_interval:.1f}分钟, 阈值={threshold:.1f}分钟)")
        
        logger.debug(f"组合奖励计算: 利润={profit_reward:.2f}, 准确度={accuracy_reward:.2f}, 序列={sequence_reward:.2f}, 总计={combined_reward:.2f}")
        
        return combined_reward
    
    def _calculate_trade_interval(self) -> Optional[float]:
        """
        计算最近交易的时间间隔
        
        返回:
            平均交易间隔（分钟），如果无法计算则返回None
        """
        if len(self.trades_history) < 2:
            return None
            
        # 获取最近的交易记录
        recent_trades = self.trades_history[-20:] if len(self.trades_history) > 20 else self.trades_history
        
        # 提取时间戳
        trade_timestamps = []
        for trade in recent_trades:
            timestamp = trade.get('timestamp')
            
            # 尝试多种时间格式
            if timestamp:
                try:
                    # 尝试ISO格式
                    dt = self._parse_timestamp(timestamp)
                    if dt:
                        trade_timestamps.append(dt)
                except Exception as e:
                    logger.debug(f"解析时间戳失败: {timestamp}, 错误: {e}")
                    continue
        
        # 计算时间间隔
        intervals = []
        for i in range(1, len(trade_timestamps)):
            try:
                interval = (trade_timestamps[i] - trade_timestamps[i-1]).total_seconds() / 60  # 转换为分钟
                if interval > 0:  # 确保时间是按顺序的
                    intervals.append(interval)
            except Exception:
                continue
                
        # 返回平均间隔
        if intervals:
            # 使用中位数而不是平均值，以减少离群值的影响
            return np.median(intervals)
        return None
        
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        解析时间戳字符串为datetime对象，支持多种格式
        
        参数:
            timestamp_str: 时间戳字符串
            
        返回:
            datetime对象，如果解析失败则返回None
        """
        # 尝试多种时间格式
        formats = [
            '%Y-%m-%dT%H:%M:%S',          # ISO 8601 不带毫秒
            '%Y-%m-%dT%H:%M:%S.%f',        # ISO 8601 带毫秒
            '%Y-%m-%d %H:%M:%S',           # 标准格式 不带毫秒
            '%Y-%m-%d %H:%M:%S.%f',        # 标准格式 带毫秒
            '%Y/%m/%d %H:%M:%S',           # 斜杠分隔日期
            '%d/%m/%Y %H:%M:%S',           # 欧洲格式
            '%m/%d/%Y %H:%M:%S',           # 美国格式
            '%Y%m%d%H%M%S',                # 紧凑格式
        ]
        
        # 尝试解析时间戳
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
                
        # 尝试作为Unix时间戳解析
        try:
            timestamp_value = float(timestamp_str)
            return datetime.fromtimestamp(timestamp_value)
        except ValueError:
            pass
            
        # 如果所有尝试都失败，记录警告并返回None
        logger.warning(f"无法解析时间戳: {timestamp_str}")
        return None
    
    def _calculate_market_relative_reward(self,
                                         prediction: np.ndarray, 
                                         ground_truth: np.ndarray, 
                                         market_data: Dict = None, 
                                         trade_result: Dict = None) -> float:
        """
        相对于市场表现计算奖励
        
        参数:
            prediction: 模型预测结果
            ground_truth: 真实标签
            market_data: 市场数据
            trade_result: 交易结果
            
        返回:
            奖励值
        """
        if trade_result is None or market_data is None or 'market_return' not in market_data:
            logger.warning("无法计算市场相对奖励，缺少必要数据")
            return self._calculate_profit_reward(prediction, ground_truth, trade_result)
            
        # 获取交易利润
        trade_profit = trade_result.get('profit', 0.0)
        
        # 计算相同时期的市场回报
        trade_start = trade_result.get('entry_time')
        trade_end = trade_result.get('exit_time')
        
        if trade_start is None or trade_end is None:
            logger.warning("交易缺少开始或结束时间，无法计算市场相对回报")
            return self._calculate_profit_reward(prediction, ground_truth, trade_result)
        
        # 获取市场在相同时期的回报
        market_return = market_data['market_return']
        
        # 计算相对回报
        relative_profit = trade_profit - market_return
        
        # 如果模型表现优于市场，给予额外奖励；如果表现不如市场，给予惩罚
        if relative_profit > 0:
            reward = relative_profit * self.profit_multiplier * 1.2  # 优于市场，增加20%奖励
        else:
            reward = relative_profit * self.profit_multiplier * 1.5  # 不如市场，增加50%惩罚
            
        logger.debug(f"计算市场相对奖励: 交易利润={trade_profit:.2f}, 市场回报={market_return:.2f}, 相对利润={relative_profit:.2f}, 奖励={reward:.2f}")
        
        return reward
    
    def _update_trade_history(self, trade_result: Dict) -> None:
        """更新交易历史和统计信息"""
        # 添加奖励记录
        if 'reward' not in trade_result and 'profit' in trade_result:
            trade_result['reward'] = float(trade_result['profit']) * self.profit_multiplier
            
        # 添加时间戳（如果没有）
        if 'timestamp' not in trade_result:
            trade_result['timestamp'] = datetime.now().isoformat()
            
        # 深拷贝交易结果，避免引用问题
        trade_copy = copy.deepcopy(trade_result)
        
        # 记录当前的奖励参数
        trade_copy['reward_params'] = {
            'profit_multiplier': self.profit_multiplier,
            'accuracy_weight': self.accuracy_weight,
            'sequence_weight': self.sequence_weight,
            'risk_reward_weight': self.risk_reward_weight,
            'market_condition_weight': self.market_condition_weight,
            'exploration_rate': self.exploration_rate
        }
        
        self.trades_history.append(trade_copy)
        self.total_trades += 1
        
        # 更新连胜/连败记录
        profit = trade_result.get('profit', 0.0)
        if profit > 0:
            self.successful_trades += 1
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.win_streak = 0
            self.loss_streak += 1
            
        # 记录调整历史
        adjustment_record = {
            'timestamp': datetime.now().isoformat(),
            'trade_id': trade_result.get('trade_id', str(self.total_trades)),
            'profit': profit,
            'reward': trade_result.get('reward', profit * self.profit_multiplier),
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'win_rate': self.successful_trades / self.total_trades if self.total_trades > 0 else 0,
            'exploration_rate': self.exploration_rate,
            'is_exploration': trade_result.get('metadata', {}).get('exploration', {}).get('is_exploration', False)
        }
        
        self.adjustment_history.append(adjustment_record)
        
        # 根据最近的交易表现动态调整参数
        if self.total_trades % 20 == 0 and self.total_trades > 0:
            self._adjust_parameters_dynamically()
    
    def _adjust_parameters_dynamically(self) -> None:
        """根据最近的交易表现动态调整参数"""
        if len(self.trades_history) < 20:
            return  # 数据不足，不进行调整
            
        # 分析最近20次交易
        recent_trades = self.trades_history[-20:]
        recent_profits = [t.get('profit', 0) for t in recent_trades]
        recent_rewards = [t.get('reward', 0) for t in recent_trades]
        
        # 计算盈亏比和胜率
        win_count = sum(1 for p in recent_profits if p > 0)
        win_rate = win_count / len(recent_profits)
        
        # 利润统计
        avg_profit = sum(recent_profits) / len(recent_profits)
        positive_profits = [p for p in recent_profits if p > 0]
        negative_profits = [p for p in recent_profits if p < 0]
        avg_win = sum(positive_profits) / len(positive_profits) if positive_profits else 0
        avg_loss = sum(negative_profits) / len(negative_profits) if negative_profits else 0
        
        # 奖励统计
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        # 根据表现调整参数
        # 1. 如果胜率低但平均盈利较高，增加profit_multiplier
        if win_rate < 0.4 and avg_profit > 0:
            adjustment = 0.05
            old_value = self.profit_multiplier
            self.profit_multiplier = min(2.0, self.profit_multiplier * (1 + adjustment))
            logger.debug(f"调整profit_multiplier: {old_value:.2f} -> {self.profit_multiplier:.2f}")
            
        # 2. 如果胜率高但平均盈利较低，减少profit_multiplier，增加accuracy_weight
        elif win_rate > 0.6 and avg_profit < 0:
            profit_adj = -0.05
            acc_adj = 0.05
            old_profit_mult = self.profit_multiplier
            old_acc_weight = self.accuracy_weight
            
            self.profit_multiplier = max(0.5, self.profit_multiplier * (1 + profit_adj))
            self.accuracy_weight = min(1.0, self.accuracy_weight * (1 + acc_adj))
            
            logger.debug(f"调整profit_multiplier: {old_profit_mult:.2f} -> {self.profit_multiplier:.2f}")
            logger.debug(f"调整accuracy_weight: {old_acc_weight:.2f} -> {self.accuracy_weight:.2f}")
            
        # 3. 如果平均亏损过大，增加风险调整的权重
        if avg_loss < -50:
            adjustment = 0.1
            old_value = self.risk_reward_weight
            self.risk_reward_weight = min(1.0, self.risk_reward_weight * (1 + adjustment))
            logger.debug(f"调整risk_reward_weight: {old_value:.2f} -> {self.risk_reward_weight:.2f}")
            
        # 4. 如果表现稳定，轻微调整参数以探索更好的组合
        if 0.45 < win_rate < 0.55 and -10 < avg_profit < 10:
            # 随机选择一个参数进行小幅调整
            param_choice = random.choice([
                'profit_multiplier', 'accuracy_weight', 'sequence_weight', 
                'risk_reward_weight', 'market_condition_weight'
            ])
            
            adjustment = random.uniform(-0.05, 0.05)
            if param_choice == 'profit_multiplier':
                old_value = self.profit_multiplier
                self.profit_multiplier = max(0.5, min(2.0, self.profit_multiplier * (1 + adjustment)))
                logger.debug(f"随机调整profit_multiplier: {old_value:.2f} -> {self.profit_multiplier:.2f}")
                
            elif param_choice == 'accuracy_weight':
                old_value = self.accuracy_weight
                self.accuracy_weight = max(0.1, min(1.0, self.accuracy_weight * (1 + adjustment)))
                logger.debug(f"随机调整accuracy_weight: {old_value:.2f} -> {self.accuracy_weight:.2f}")
                
            elif param_choice == 'sequence_weight':
                old_value = self.sequence_weight
                self.sequence_weight = max(0.1, min(0.7, self.sequence_weight * (1 + adjustment)))
                logger.debug(f"随机调整sequence_weight: {old_value:.2f} -> {self.sequence_weight:.2f}")
                
            elif param_choice == 'risk_reward_weight':
                old_value = self.risk_reward_weight
                self.risk_reward_weight = max(0.1, min(1.0, self.risk_reward_weight * (1 + adjustment)))
                logger.debug(f"随机调整risk_reward_weight: {old_value:.2f} -> {self.risk_reward_weight:.2f}")
                
            elif param_choice == 'market_condition_weight':
                old_value = self.market_condition_weight
                self.market_condition_weight = max(0.05, min(0.5, self.market_condition_weight * (1 + adjustment)))
                logger.debug(f"随机调整market_condition_weight: {old_value:.2f} -> {self.market_condition_weight:.2f}")
        
        # 记录调整结果
        self.adjustment_history[-1]['parameter_adjustments'] = {
            'profit_multiplier': self.profit_multiplier,
            'accuracy_weight': self.accuracy_weight,
            'sequence_weight': self.sequence_weight,
            'risk_reward_weight': self.risk_reward_weight,
            'market_condition_weight': self.market_condition_weight
        }
    
    def _decay_exploration_rate(self) -> None:
        """衰减探索率，但使用更智能的方式确保足够的探索"""
        if len(self.trades_history) < 50:
            # 早期阶段维持较高的探索率
            self.exploration_rate = max(
                self.min_exploration_rate * 2, 
                self.exploration_rate * self.exploration_decay
            )
        else:
            # 获取探索和标准交易的表现差异
            exploration_analysis = self._analyze_exploration_effectiveness()
            exploration_benefit = exploration_analysis.get('exploration_benefit', {}).get('overall_benefit', 0)
            
            if exploration_benefit > 0.1:
                # 如果探索有明显收益，减缓衰减速度
                decay_factor = max(0.998, self.exploration_decay + 0.003)
                self.exploration_rate = max(
                    self.min_exploration_rate, 
                    self.exploration_rate * decay_factor
                )
                logger.debug(f"探索效果良好，减缓探索衰减 (衰减因子: {decay_factor:.4f})")
            elif exploration_benefit < -0.1:
                # 如果探索表现不佳，加速衰减
                decay_factor = min(0.99, self.exploration_decay - 0.005)
                self.exploration_rate = max(
                    self.min_exploration_rate, 
                    self.exploration_rate * decay_factor
                )
                logger.debug(f"探索效果不佳，加速探索衰减 (衰减因子: {decay_factor:.4f})")
            else:
                # 正常衰减
                self.exploration_rate = max(
                    self.min_exploration_rate, 
                    self.exploration_rate * self.exploration_decay
                )
                
        # 定期小幅度增加探索率，以防止过早陷入局部最优
        if len(self.trades_history) % 100 == 0 and len(self.trades_history) > 0:
            boost_factor = 1.05 + (random.random() * 0.1)  # 5%-15%的提升
            old_rate = self.exploration_rate
            self.exploration_rate = min(0.3, self.exploration_rate * boost_factor)  # 最高不超过0.3
            logger.info(f"探索率定期提升: {old_rate:.4f} -> {self.exploration_rate:.4f}")
    
    def _analyze_exploration_effectiveness(self) -> Dict:
        """
        分析探索效果，比较探索交易与标准交易的表现
        
        返回:
            探索效果分析结果
        """
        # 分离探索交易和标准交易
        exploration_trades = []
        standard_trades = []
        
        for trade in self.trades_history:
            if trade.get('metadata', {}).get('exploration', {}).get('is_exploration', False):
                exploration_trades.append(trade)
            else:
                standard_trades.append(trade)
                
        # 如果探索或标准交易太少，返回默认值
        min_trade_count = 5
        if len(exploration_trades) < min_trade_count or len(standard_trades) < min_trade_count:
            return {
                'exploration_trades': len(exploration_trades),
                'standard_trades': len(standard_trades),
                'exploration_benefit': {
                    'win_rate_difference': 0.0,
                    'profit_difference': 0.0,
                    'overall_benefit': 0.0
                },
                'exploration_by_type': {}
            }
            
        # 计算探索交易的胜率和平均利润
        exp_win_count = sum(1 for t in exploration_trades if t.get('profit', 0) > 0)
        exp_win_rate = exp_win_count / len(exploration_trades)
        exp_profits = [t.get('profit', 0) for t in exploration_trades]
        exp_avg_profit = sum(exp_profits) / len(exp_profits)
        
        # 计算标准交易的胜率和平均利润
        std_win_count = sum(1 for t in standard_trades if t.get('profit', 0) > 0)
        std_win_rate = std_win_count / len(standard_trades)
        std_profits = [t.get('profit', 0) for t in standard_trades]
        std_avg_profit = sum(std_profits) / len(std_profits)
        
        # 计算差异
        win_rate_diff = exp_win_rate - std_win_rate
        profit_diff = exp_avg_profit - std_avg_profit
        
        # 计算整体收益（加权组合）
        overall_benefit = win_rate_diff * 0.4 + (profit_diff / max(abs(std_avg_profit), 1.0)) * 0.6
        
        # 按探索类型分析
        exploration_types = {}
        for trade in exploration_trades:
            exploration_type = trade.get('metadata', {}).get('exploration', {}).get('type', 'unknown')
            if exploration_type not in exploration_types:
                exploration_types[exploration_type] = {
                    'count': 0,
                    'wins': 0,
                    'profits': []
                }
            
            exploration_types[exploration_type]['count'] += 1
            if trade.get('profit', 0) > 0:
                exploration_types[exploration_type]['wins'] += 1
            exploration_types[exploration_type]['profits'].append(trade.get('profit', 0))
            
        # 计算每种类型的指标
        exploration_by_type = {}
        for exp_type, data in exploration_types.items():
            if data['count'] > 0:
                win_rate = data['wins'] / data['count']
                avg_profit = sum(data['profits']) / data['count']
                
                exploration_by_type[exp_type] = {
                    'count': data['count'],
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'benefit_vs_standard': (win_rate - std_win_rate) * 0.4 + 
                                          (avg_profit - std_avg_profit) / max(abs(std_avg_profit), 1.0) * 0.6
                }
        
        return {
            'exploration_trades': len(exploration_trades),
            'standard_trades': len(standard_trades),
            'exploration_win_rate': exp_win_rate,
            'standard_win_rate': std_win_rate,
            'exploration_avg_profit': exp_avg_profit,
            'standard_avg_profit': std_avg_profit,
            'exploration_benefit': {
                'win_rate_difference': win_rate_diff,
                'profit_difference': profit_diff,
                'overall_benefit': overall_benefit
            },
            'exploration_by_type': exploration_by_type
        }
    
    def get_trade_statistics(self) -> Dict:
        """
        获取交易统计数据
        
        返回:
            交易统计信息
        """
        if not self.trades_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'max_drawdown': 0.0,
                'current_win_streak': self.win_streak,
                'current_loss_streak': self.loss_streak,
                'exploration_rate': self.exploration_rate,
                'exploration_count': self.exploration_count
            }
            
        total_trades = len(self.trades_history)
        winning_trades = sum(1 for trade in self.trades_history if trade.get('profit', 0) > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # 计算平均盈利和亏损
        profits = [trade.get('profit', 0) for trade in self.trades_history if trade.get('profit', 0) > 0]
        losses = [trade.get('profit', 0) for trade in self.trades_history if trade.get('profit', 0) < 0]
        
        avg_profit = sum(profits) / len(profits) if profits else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        # 计算总盈利和总亏损
        total_profit = sum(profits)
        total_loss = abs(sum(losses))
        
        # 计算盈亏比
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # 计算最大回撤
        equity_curve = [0]
        for trade in self.trades_history:
            equity_curve.append(equity_curve[-1] + trade.get('profit', 0))
            
        max_equity = 0
        max_drawdown = 0
        
        for equity in equity_curve:
            if equity > max_equity:
                max_equity = equity
            else:
                drawdown = max_equity - equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'max_drawdown': max_drawdown,
            'current_win_streak': self.win_streak,
            'current_loss_streak': self.loss_streak,
            'exploration_rate': self.exploration_rate,
            'exploration_count': self.exploration_count
        }
    
    def set_config(self, config: Dict) -> None:
        """
        更新配置参数
        
        参数:
            config: 新的配置参数
        """
        self.config.update(config)
        
        # 更新参数
        if 'reward_type' in config:
            self.reward_type = config['reward_type']
            
        if 'point_value' in config:
            self.point_value = float(config['point_value'])
            
        if 'profit_multiplier' in config:
            self.profit_multiplier = float(config['profit_multiplier'])
            
        if 'accuracy_weight' in config:
            self.accuracy_weight = float(config['accuracy_weight'])
            
        if 'sequence_weight' in config:
            self.sequence_weight = float(config['sequence_weight'])
            
        if 'drawdown_penalty' in config:
            self.drawdown_penalty = float(config['drawdown_penalty'])
            
        if 'volatility_influence' in config:
            self.volatility_influence = float(config['volatility_influence'])
            
        if 'trend_influence' in config:
            self.trend_influence = float(config['trend_influence'])
            
        if 'time_decay_factor' in config:
            self.time_decay_factor = float(config['time_decay_factor'])
            
        if 'trade_frequency_penalty' in config:
            self.trade_frequency_penalty = float(config['trade_frequency_penalty'])
            
        if 'long_win_bonus' in config:
            self.long_win_bonus = float(config['long_win_bonus'])
            
        if 'short_win_bonus' in config:
            self.short_win_bonus = float(config['short_win_bonus'])
            
        if 'consecutive_win_bonus' in config:
            self.consecutive_win_bonus = float(config['consecutive_win_bonus'])
            
        if 'market_condition_weight' in config:
            self.market_condition_weight = float(config['market_condition_weight'])
            
        if 'min_reward' in config:
            self.min_reward = float(config['min_reward'])
            
        if 'max_reward' in config:
            self.max_reward = float(config['max_reward'])
            
        if 'exploration_rate' in config:
            self.exploration_rate = float(config['exploration_rate'])
            
        logger.info(f"奖励计算器配置已更新: {config}")
        
    def adaptive_update(self, performance_metrics: Dict) -> None:
        """
        根据性能指标自适应调整奖励参数
        
        参数:
            performance_metrics: 性能指标字典
        """
        # 获取关键指标
        win_rate = performance_metrics.get('win_rate', 0.5)
        profit_factor = performance_metrics.get('profit_factor', 1.0)
        avg_profit = performance_metrics.get('avg_profit', 0.0)
        avg_loss = performance_metrics.get('avg_loss', 0.0)
        
        # 调整利润乘数
        if profit_factor < 1.0:
            # 如果盈亏比小于1，减少利润乘数，增加对利润的重视
            self.profit_multiplier = max(0.5, self.profit_multiplier * 0.95)
        elif profit_factor > 2.0:
            # 如果盈亏比很好，适度增加利润乘数
            self.profit_multiplier = min(2.0, self.profit_multiplier * 1.05)
            
        # 调整准确度权重
        if win_rate < 0.4:
            # 胜率低，增加准确度重视
            self.accuracy_weight = min(1.0, self.accuracy_weight * 1.1)
        elif win_rate > 0.6:
            # 胜率高，可以减少准确度重视
            self.accuracy_weight = max(0.1, self.accuracy_weight * 0.95)
            
        # 调整回撤惩罚
        if avg_loss < -50:
            # 平均亏损较大，增加回撤惩罚
            self.drawdown_penalty = min(0.5, self.drawdown_penalty * 1.1)
        elif avg_loss > -20:
            # 平均亏损较小，减少回撤惩罚
            self.drawdown_penalty = max(0.05, self.drawdown_penalty * 0.9)
            
        # 调整探索率
        if profit_factor < 1.0 or win_rate < 0.4:
            # 性能不佳，增加探索
            self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
        else:
            # 性能良好，减少探索
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * 0.9)
            
        logger.info(f"自适应更新奖励参数: profit_mult={self.profit_multiplier:.2f}, acc_weight={self.accuracy_weight:.2f}, drawdown_penalty={self.drawdown_penalty:.2f}, exploration_rate={self.exploration_rate:.2f}")
    
    def save_trade_history(self, filename: str) -> None:
        """
        保存交易历史记录并生成可视化
        
        参数:
            filename: 文件名
        """
        if not self.trades_history:
            logger.warning("没有交易历史记录可保存")
            return
            
        save_data = {
            'trades_history': self.trades_history,
            'adjustment_history': self.adjustment_history,
            'statistics': self.get_trade_statistics(),
            'exploration_analysis': self._analyze_exploration_effectiveness(),
            'config': self.config,
            'current_parameters': {
                'reward_type': self.reward_type,
                'profit_multiplier': self.profit_multiplier,
                'accuracy_weight': self.accuracy_weight,
                'sequence_weight': self.sequence_weight,
                'drawdown_penalty': self.drawdown_penalty,
                'volatility_influence': self.volatility_influence,
                'trend_influence': self.trend_influence,
                'market_condition_weight': self.market_condition_weight,
                'time_decay_factor': self.time_decay_factor,
                'trade_frequency_penalty': self.trade_frequency_penalty,
                'long_win_bonus': self.long_win_bonus,
                'short_win_bonus': self.short_win_bonus,
                'consecutive_win_bonus': self.consecutive_win_bonus,
                'exploration_rate': self.exploration_rate
            }
        }
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 保存JSON数据
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=4)
                
            logger.info(f"交易历史已保存至: {filename}")
            
            # 生成可视化
            try:
                self._generate_visualizations(filename)
            except Exception as e:
                logger.error(f"生成可视化时发生错误: {e}")
                
        except Exception as e:
            logger.error(f"保存交易历史失败: {e}")
            
    def _generate_visualizations(self, data_filename: str) -> None:
        """
        生成交易历史和奖励的可视化
        
        参数:
            data_filename: 数据文件名
        """
        # 检查是否有matplotlib
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        except ImportError:
            logger.warning("未安装matplotlib，无法生成可视化")
            return
            
        if not self.trades_history:
            logger.warning("没有交易历史数据，无法生成可视化")
            return
            
        # 提取数据
        timestamps = []
        profits = []
        rewards = []
        exploration_flags = []
        cumulative_profit = 0
        cumulative_profits = []
        
        for trade in self.trades_history:
            # 解析时间戳
            timestamp_str = trade.get('timestamp')
            timestamp = self._parse_timestamp(timestamp_str) if timestamp_str else None
            if timestamp:
                timestamps.append(timestamp)
                
                # 获取利润和奖励
                profit = trade.get('profit', 0)
                reward = trade.get('reward', profit * self.profit_multiplier)
                
                profits.append(profit)
                rewards.append(reward)
                
                # 累计利润
                cumulative_profit += profit
                cumulative_profits.append(cumulative_profit)
                
                # 是否为探索交易
                is_exploration = trade.get('metadata', {}).get('exploration', {}).get('is_exploration', False)
                exploration_flags.append(is_exploration)
        
        if len(timestamps) < 2:
            logger.warning("数据点不足，无法生成有意义的可视化")
            return
            
        # 创建图表目录
        vis_dir = os.path.join(os.path.dirname(data_filename), 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        base_filename = os.path.splitext(os.path.basename(data_filename))[0]
        
        # 1. 利润和奖励对比图
        fig1 = Figure(figsize=(12, 6))
        canvas1 = FigureCanvas(fig1)
        ax1 = fig1.add_subplot(111)
        
        ax1.plot(range(len(profits)), profits, label='交易利润', color='blue', alpha=0.7)
        ax1.plot(range(len(rewards)), rewards, label='奖励值', color='red', alpha=0.7)
        
        # 标记探索交易
        for i, is_exp in enumerate(exploration_flags):
            if is_exp:
                ax1.scatter(i, profits[i], color='green', s=50, zorder=5)
                
        ax1.set_title('交易利润与奖励值对比')
        ax1.set_xlabel('交易序号')
        ax1.set_ylabel('金额/奖励')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        profit_reward_fig_path = os.path.join(vis_dir, f"{base_filename}_profit_reward.png")
        fig1.savefig(profit_reward_fig_path)
        
        # 2. 累计利润图
        fig2 = Figure(figsize=(12, 6))
        canvas2 = FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111)
        
        ax2.plot(range(len(cumulative_profits)), cumulative_profits, label='累计利润', color='green')
        
        # 标记最大回撤
        if len(cumulative_profits) > 0:
            max_drawdown, max_dd_start, max_dd_end = self._calculate_max_drawdown(cumulative_profits)
            if max_dd_start is not None and max_dd_end is not None:
                ax2.plot([max_dd_start, max_dd_end], 
                         [cumulative_profits[max_dd_start], cumulative_profits[max_dd_end]], 
                         'r-', linewidth=2, label=f'最大回撤: {max_drawdown:.2f}')
                
        ax2.set_title('累计利润曲线')
        ax2.set_xlabel('交易序号')
        ax2.set_ylabel('累计利润')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        cumulative_profit_fig_path = os.path.join(vis_dir, f"{base_filename}_cumulative_profit.png")
        fig2.savefig(cumulative_profit_fig_path)
        
        # 3. 参数变化图
        param_changes = []
        param_timestamps = []
        
        for item in self.adjustment_history:
            if 'parameter_adjustments' in item:
                param_changes.append(item['parameter_adjustments'])
                timestamp_str = item.get('timestamp')
                ts = self._parse_timestamp(timestamp_str) if timestamp_str else None
                param_timestamps.append(ts if ts else len(param_changes))
                
        if len(param_changes) > 1:
            fig3 = Figure(figsize=(12, 8))
            canvas3 = FigureCanvas(fig3)
            ax3 = fig3.add_subplot(111)
            
            # 提取所有参数名
            param_names = set()
            for params in param_changes:
                param_names.update(params.keys())
                
            # 为每个参数画一条线
            for param_name in param_names:
                values = [params.get(param_name, float('nan')) for params in param_changes]
                indices = range(len(values))
                ax3.plot(indices, values, label=param_name)
                
            ax3.set_title('奖励参数变化')
            ax3.set_xlabel('调整次数')
            ax3.set_ylabel('参数值')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            param_changes_fig_path = os.path.join(vis_dir, f"{base_filename}_param_changes.png")
            fig3.savefig(param_changes_fig_path)
            
        # 4. 探索率变化图
        exploration_rates = [item.get('exploration_rate', float('nan')) for item in self.adjustment_history]
        if any(not math.isnan(rate) for rate in exploration_rates):
            fig4 = Figure(figsize=(12, 4))
            canvas4 = FigureCanvas(fig4)
            ax4 = fig4.add_subplot(111)
            
            ax4.plot(range(len(exploration_rates)), exploration_rates)
            ax4.set_title('探索率变化')
            ax4.set_xlabel('交易序号')
            ax4.set_ylabel('探索率')
            ax4.grid(True, alpha=0.3)
            
            exploration_rate_fig_path = os.path.join(vis_dir, f"{base_filename}_exploration_rate.png")
            fig4.savefig(exploration_rate_fig_path)
            
        logger.info(f"交易数据可视化已保存至目录: {vis_dir}")
        
    def _calculate_max_drawdown(self, cumulative_profits: List[float]) -> Tuple[float, Optional[int], Optional[int]]:
        """
        计算最大回撤及其起始和结束位置
        
        参数:
            cumulative_profits: 累计利润列表
            
        返回:
            (最大回撤值, 起始位置, 结束位置)
        """
        if not cumulative_profits:
            return 0, None, None
            
        max_dd = 0
        max_dd_start = None
        max_dd_end = None
        
        peak = cumulative_profits[0]
        peak_idx = 0
        
        for i, profit in enumerate(cumulative_profits):
            if profit > peak:
                peak = profit
                peak_idx = i
            else:
                current_dd = peak - profit
                if current_dd > max_dd:
                    max_dd = current_dd
                    max_dd_start = peak_idx
                    max_dd_end = i
                    
        return max_dd, max_dd_start, max_dd_end
    
    def reset(self) -> None:
        """重置交易历史记录和统计信息"""
        self.trades_history = []
        self.adjustment_history = []
        self.win_streak = 0
        self.loss_streak = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.exploration_count = 0
        
        # 重置探索率到初始值
        self.exploration_rate = float(self.config.get('exploration_rate', 0.1))
        
        logger.info("奖励计算器已重置")
        
    def is_compatible_with_output(self, output_shape) -> bool:
        """
        检查奖励计算器是否与模型输出兼容
        
        参数:
            output_shape: 模型输出形状
            
        返回:
            是否兼容
        """
        # 如果没有指定输出形状，默认为兼容
        if output_shape is None:
            return True
            
        # 解析输出形状
        try:
            _, seq_len, feature_dim = output_shape
            
            # 检查特征维度是否满足最低要求
            # SBS序列模型通常需要输出5个点的预测，每个点有x,y坐标，共10个值，加上方向预测
            min_feature_dim = 10 + 1  # 5个点(x,y)坐标 + 1个方向
            
            if feature_dim < min_feature_dim:
                logger.warning(f"模型输出特征维度({feature_dim})小于奖励计算所需的最小维度({min_feature_dim})")
                return False
                
            return True
        except (ValueError, TypeError):
            logger.warning(f"无法解析模型输出形状: {output_shape}")
            return False
        
    def load_trade_history(self, filename: str) -> bool:
        """
        加载交易历史记录
        
        参数:
            filename: 文件名
            
        返回:
            加载是否成功
        """
        try:
            if not os.path.exists(filename):
                logger.error(f"文件不存在: {filename}")
                return False
                
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # 加载交易历史
            self.trades_history = data.get('trades_history', [])
            self.adjustment_history = data.get('adjustment_history', [])
            
            # 恢复参数
            parameters = data.get('current_parameters', {})
            for key, value in parameters.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
            # 重新计算统计信息
            stats = self.get_trade_statistics()
            self.win_streak = stats.get('current_win_streak', 0)
            self.loss_streak = stats.get('current_loss_streak', 0)
            self.total_trades = stats.get('total_trades', 0)
            self.successful_trades = stats.get('winning_trades', 0)
            self.exploration_count = stats.get('exploration_count', 0)
                
            logger.info(f"已加载{len(self.trades_history)}条交易历史记录")
            
            # 尝试加载配置
            if 'config' in data:
                # 仅更新那些没有显式在当前对象中设置的配置项
                for key, value in data['config'].items():
                    if key not in self.config:
                        self.config[key] = value
                logger.info("已加载历史配置数据")
                
            # 验证加载的数据有效性
            self._validate_loaded_data()
                
            return True
            
        except json.JSONDecodeError:
            logger.error(f"JSON解析错误，文件可能已损坏: {filename}")
            return False
        except Exception as e:
            logger.error(f"加载交易历史失败: {str(e)}")
            return False
            
    def _validate_loaded_data(self):
        """验证加载的数据有效性并修复潜在问题"""
        # 检查交易历史中是否有必要的字段
        required_fields = ['profit', 'timestamp']
        valid_trades = []
        
        for trade in self.trades_history:
            is_valid = True
            for field in required_fields:
                if field not in trade:
                    is_valid = False
                    break
                    
            # 确保reward字段存在
            if 'reward' not in trade and 'profit' in trade:
                trade['reward'] = float(trade['profit']) * self.profit_multiplier
                
            if is_valid:
                valid_trades.append(trade)
            else:
                logger.warning(f"忽略无效交易记录: 缺少必要字段 {required_fields}")
                
        # 更新交易历史为有效交易
        if len(valid_trades) < len(self.trades_history):
            logger.warning(f"从{len(self.trades_history)}个交易记录中移除了{len(self.trades_history) - len(valid_trades)}个无效记录")
            self.trades_history = valid_trades 