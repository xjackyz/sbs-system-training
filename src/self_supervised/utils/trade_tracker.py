"""
交易结果跟踪系统
用于跟踪和评估交易结果和利润
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import random
import gc
import gzip
from itertools import groupby
import concurrent.futures
from functools import partial
from tqdm import tqdm

from ..utils.logger import setup_logger
from .exploration_config import ExplorationConfig
from .exploration_manager import ExplorationManager

logger = setup_logger('trade_tracker')

class TradeResultTracker:
    """交易结果跟踪器类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化交易结果跟踪器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = setup_logger('trade_tracker')
        
        # 交易记录
        self.active_trades = {}  # 活跃交易
        self.completed_trades = []  # 已完成交易
        self.trade_history = []  # 交易历史记录
        
        # 存储路径
        self.storage_dir = Path(self.config.get('storage_dir', 'data/trade_results'))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 探索管理器
        self.exploration_manager = ExplorationManager(self.config)
        
        # 探索机制参数 - 从config中获取所有参数
        self.exploration_enabled = self.config.get('exploration_enabled', False)
        self.exploration_rate = self.config.get('exploration_rate', 0.1)  # ε 值
        self.min_exploration_rate = self.config.get('min_exploration_rate', 0.01)
        self.exploration_decay = self.config.get('exploration_decay', 0.995)  # 探索率衰减系数
        
        # 探索参数增强 - 添加新配置项
        self.exploration_boost_interval = self.config.get('exploration_boost_interval', 100)
        self.exploration_boost_factor = self.config.get('exploration_boost_factor', 1.05)
        self.exploration_success_threshold = self.config.get('exploration_success_threshold', 0.6)
        self.exploration_failure_threshold = self.config.get('exploration_failure_threshold', 0.3)
        self.exploration_success_rate_adjust = self.config.get('exploration_success_rate_adjust', 0.05)
        self.exploration_failure_rate_adjust = self.config.get('exploration_failure_rate_adjust', 0.05)
        
        # 内存管理参数
        self.memory_management_enabled = self.config.get('memory_management_enabled', False)
        self.max_trades_in_memory = self.config.get('max_trades_in_memory', 5000)
        self.trade_archive_threshold = self.config.get('trade_archive_threshold', 2000)  # 达到多少交易时归档
        self.archive_dir = self.storage_dir / 'archives'
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.trade_count_since_cleanup = 0
        self.cleanup_interval = self.config.get('cleanup_interval', 1000)  # 多少交易后清理一次内存
        
        self.exploration_metrics = {
            'total_explorations': 0,
            'successful_explorations': 0,
            'exploration_trades': []
        }
        
        # 加载历史数据
        self._load_trade_history()
        
    def should_explore(self, market_state: Optional[Dict[str, Any]] = None) -> bool:
        """
        决定当前是否应该进行探索交易
        
        参数:
            market_state: 市场状态数据，可选
            
        返回:
            是否应该探索
        """
        return self.exploration_manager.should_explore(market_state)
    
    def enable_exploration(self, enabled: bool = True, rate: float = None) -> None:
        """
        启用或禁用探索机制
        
        参数:
            enabled: 是否启用探索
            rate: 探索率，如果提供则更新当前探索率
        """
        # 更新探索管理器配置
        self.exploration_manager.exploration_config.enabled = enabled
        
        # 如果提供了新的探索率，则更新
        if rate is not None:
            self.exploration_manager.exploration_config.exploration_rate = rate
            
        self.logger.info(f"探索机制已{'启用' if enabled else '禁用'}, 当前探索率: {self.exploration_manager.exploration_config.exploration_rate:.4f}")
    
    def update_exploration_rate(self, decay: bool = True, new_rate: float = None) -> None:
        """
        更新探索率
        
        参数:
            decay: 是否应用衰减
            new_rate: 新的探索率，如果提供则直接设置
        """
        current_rate = self.exploration_manager.exploration_config.exploration_rate
        
        if new_rate is not None:
            # 直接设置新的探索率
            self.exploration_manager.exploration_config.exploration_rate = max(
                self.exploration_manager.exploration_config.min_exploration_rate,
                new_rate
            )
        elif decay:
            # 使用探索管理器的内部更新机制
            self.exploration_manager._update_exploration_rate()
        
        new_rate = self.exploration_manager.exploration_config.exploration_rate
        if current_rate != new_rate:
            self.logger.debug(f"探索率更新: {current_rate:.4f} -> {new_rate:.4f}")
        
    def add_trade(self, 
                 trade_id: str, 
                 symbol: str, 
                 direction: str, 
                 entry_price: float,
                 stop_loss: float,
                 take_profit: float,
                 entry_time: str = None,
                 timeframe: str = None,
                 risk_percentage: float = 1.0,
                 metadata: Dict = None,
                 sequence_points: Dict = None,
                 confirmation_signal: str = None,
                 market_data: Dict = None,
                 is_exploration: bool = None) -> Dict:
        """
        添加新交易
        
        参数:
            trade_id: 交易ID
            symbol: 交易品种
            direction: 交易方向，多/空
            entry_price: 入场价格
            stop_loss: 止损价格
            take_profit: 止盈价格
            entry_time: 入场时间，ISO格式
            timeframe: 时间框架
            risk_percentage: 风险百分比
            metadata: 元数据字典
            sequence_points: SBS序列点位数据
            confirmation_signal: 确认信号
            market_data: 市场数据
            is_exploration: 是否为探索交易，如果为None则自动判断
        
        返回:
            交易记录字典
        """
        if trade_id in self.active_trades:
            self.logger.warning(f"交易ID已存在: {trade_id}")
            return self.active_trades[trade_id]
        
        # 设置交易时间
        if entry_time is None:
            entry_time = datetime.now().isoformat()
        
        # 处理元数据
        if metadata is None:
            metadata = {}
        
        # 确定是否为探索交易
        if is_exploration is None and hasattr(self, 'exploration_manager'):
            is_exploration = self.exploration_manager.last_decision_was_exploration
        
        # 添加探索标记到元数据
        if is_exploration is not None:
            metadata['is_exploration'] = is_exploration
        
        # 计算交易质量分数
        quality_score = None
        if sequence_points and stop_loss and take_profit:
            try:
                quality_results = self.calculate_trade_quality_score(
                    sequence_points=sequence_points,
                    confirmation_signal=confirmation_signal,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    market_data=market_data
                )
                quality_score = quality_results.get('quality_score')
                # 添加质量评估到元数据
                metadata['quality_assessment'] = quality_results
            except Exception as e:
                self.logger.error(f"计算交易质量评分失败: {e}")
        
        # 创建交易记录
        trade = {
            'trade_id': trade_id,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'current_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': entry_time,
            'last_update_time': entry_time,
            'timeframe': timeframe,
            'risk_percentage': risk_percentage,
            'status': 'active',
            'profit_percentage': 0.0,
            'max_profit_percentage': 0.0,
            'max_drawdown': 0.0,
            'metadata': metadata,
            'is_exploration': is_exploration,
            'quality_score': quality_score
        }
        
        # 添加到活跃交易
        self.active_trades[trade_id] = trade
        
        # 记录到交易历史
        self.trade_history.append({
            'action': 'open',
            'trade_id': trade_id,
            'trade': trade.copy(),
            'timestamp': datetime.now().isoformat()
        })
        
        # 记录探索决策的市场状态
        if is_exploration and market_data:
            # 存储市场状态，用于探索机制的学习
            self.exploration_manager.decision_market_state = market_data
        
        self.logger.info(f"添加{'探索' if is_exploration else '常规'}交易: {trade_id}, {symbol}, {direction}, 入场价格: {entry_price}")
        
        return trade
        
    def update_trade(self, 
                    trade_id: str, 
                    current_price: float,
                    current_time: str = None) -> Dict:
        """
        更新交易状态
        
        Args:
            trade_id: 交易ID
            current_price: 当前价格
            current_time: 当前时间
            
        Returns:
            更新后的交易信息
        """
        if trade_id not in self.active_trades:
            logger.warning(f"交易ID不存在: {trade_id}")
            return None
            
        trade = self.active_trades[trade_id]
        current_time = current_time or datetime.now().isoformat()
        
        # 计算当前盈亏百分比
        direction = trade['direction']
        entry_price = trade['entry_price']
        profit_percentage = self._calculate_profit(entry_price, current_price, direction)
        
        # 更新最大盈利
        if profit_percentage > trade['max_profit_percentage']:
            trade['max_profit_percentage'] = profit_percentage
            
        # 更新最大回撤
        if profit_percentage < 0 and abs(profit_percentage) > trade['max_drawdown']:
            trade['max_drawdown'] = abs(profit_percentage)
            
        # 检查是否触发止损或止盈
        trade_closed = False
        exit_reason = None
        
        if direction == "多":
            # 检查止损
            if current_price <= trade['stop_loss']:
                trade_closed = True
                exit_reason = 'stop_loss'
            # 检查止盈
            elif current_price >= trade['take_profit']:
                trade_closed = True
                exit_reason = 'take_profit'
        else:  # 空
            # 检查止损
            if current_price >= trade['stop_loss']:
                trade_closed = True
                exit_reason = 'stop_loss'
            # 检查止盈
            elif current_price <= trade['take_profit']:
                trade_closed = True
                exit_reason = 'take_profit'
                
        # 更新交易信息
        trade['current_price'] = current_price
        trade['profit_percentage'] = profit_percentage
        trade['last_update_time'] = current_time
        
        # 如果交易被关闭
        if trade_closed:
            self.close_trade(trade_id, current_price, exit_reason, current_time)
            
        return trade
        
    def close_trade(self, 
                   trade_id: str, 
                   exit_price: float,
                   exit_reason: str = 'manual',
                   exit_time: str = None) -> Dict:
        """
        关闭交易
        
        参数:
            trade_id: 交易ID
            exit_price: 出场价格
            exit_reason: 出场原因
            exit_time: 出场时间，ISO格式
            
        返回:
            关闭的交易记录
        """
        if trade_id not in self.active_trades:
            self.logger.warning(f"交易ID不存在: {trade_id}")
            return None
            
        # 获取交易记录
        trade = self.active_trades[trade_id]
        
        # 设置出场时间
        if exit_time is None:
            exit_time = datetime.now().isoformat()
        
        # 计算持续时间
        duration = self._calculate_duration(trade['entry_time'], exit_time)
        
        # 计算利润
        profit_percentage = self._calculate_profit(trade['entry_price'], exit_price, trade['direction'])
        
        # 更新交易记录
        trade['exit_price'] = exit_price
        trade['exit_time'] = exit_time
        trade['exit_reason'] = exit_reason
        trade['profit_percentage'] = profit_percentage
        trade['duration'] = duration
        trade['status'] = 'closed'
        
        # 完成交易
        completed_trade = self.active_trades.pop(trade_id)
        self.completed_trades.append(completed_trade)
        
        # 记录到交易历史
        self.trade_history.append({
            'action': 'close',
            'trade_id': trade_id,
            'trade': completed_trade.copy(),
            'timestamp': datetime.now().isoformat()
        })
        
        # 处理探索结果
        is_exploration = trade.get('is_exploration', False)
        if hasattr(self, 'exploration_manager') and is_exploration:
            # 获取当前市场状态
            current_market_data = None
            if trade.get('metadata') and 'market_data' in trade['metadata']:
                current_market_data = trade['metadata']['market_data']
            
            # 更新探索管理器
            self.exploration_manager.update_from_result(
                was_exploration=is_exploration,
                profit=profit_percentage,
                market_state=self.exploration_manager.decision_market_state,
                next_market_state=current_market_data,
                metadata={
                    'trade_id': trade_id, 
                    'symbol': trade['symbol'], 
                    'timeframe': trade.get('timeframe'),
                    'duration': duration,
                    'exit_reason': exit_reason,
                    'quality_score': trade.get('quality_score')
                }
            )
        
        self.logger.info(
            f"关闭{'探索' if is_exploration else '常规'}交易: {trade_id}, "
            f"出场价格: {exit_price}, 盈亏: {profit_percentage:.2f}%, "
            f"原因: {exit_reason}"
        )
        
        return completed_trade
        
    def get_trade_stats(self) -> Dict:
        """获取交易统计信息，使用pandas优化计算效率"""
        # 统计活跃交易
        active_count = len(self.active_trades)
        active_profit = sum([t['profit_percentage'] for t in self.active_trades.values()])
        
        # 统计已完成交易
        completed_count = len(self.completed_trades)
        if completed_count == 0:
            return {
                'active_trades': active_count,
                'active_profit': active_profit,
                'completed_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'last_trade_result': None,
                'current_streak': 0,
                'longest_win_streak': 0,
                'longest_loss_streak': 0,
                'avg_win_duration': 0.0,
                'avg_loss_duration': 0.0,
                'avg_risk_reward': 0.0
            }
            
        # 使用Pandas优化计算效率
        try:
            # 转换交易数据为DataFrame
            trades_df = pd.DataFrame(self.completed_trades)
            
            # 计算基本指标
            winning_trades = trades_df[trades_df['profit_percentage'] > 0]
            losing_trades = trades_df[trades_df['profit_percentage'] <= 0]
            
            winning_count = len(winning_trades)
            losing_count = len(losing_trades)
            win_rate = winning_count / completed_count
            
            # 利润统计
            if not winning_trades.empty:
                avg_profit = winning_trades['profit_percentage'].mean()
                total_profit = winning_trades['profit_percentage'].sum()
            else:
                avg_profit = 0.0
                total_profit = 0.0
                
            if not losing_trades.empty:
                avg_loss = losing_trades['profit_percentage'].mean()
                total_loss = abs(losing_trades['profit_percentage'].sum())
            else:
                avg_loss = 0.0
                total_loss = 0.0
                
            # 计算其他指标
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            expectancy = (win_rate * avg_profit) + ((1 - win_rate) * avg_loss) if win_rate > 0 else 0.0
            
            # 计算最大回撤
            if 'exit_time' in trades_df.columns:
                try:
                    # 按退出时间排序
                    trades_df['exit_datetime'] = trades_df['exit_time'].apply(self._parse_datetime)
                    trades_df = trades_df.sort_values(by='exit_datetime')
                except Exception as e:
                    self.logger.warning(f"排序交易按退出时间时出错: {e}")
            
            # 计算累计利润和最大回撤
            trades_df['cumulative_profit'] = trades_df['profit_percentage'].cumsum()
            trades_df['drawdown'] = trades_df['cumulative_profit'].cummax() - trades_df['cumulative_profit']
            max_drawdown = trades_df['drawdown'].max()
            
            # 计算其他统计信息
            streak_metrics = self.calculate_streak_metrics()
            
            # 计算交易持续时间统计
            if 'duration' in trades_df.columns:
                avg_win_duration = winning_trades['duration'].mean() if not winning_trades.empty else 0.0
                avg_loss_duration = losing_trades['duration'].mean() if not losing_trades.empty else 0.0
            else:
                avg_win_duration = 0.0
                avg_loss_duration = 0.0
            
            # 获取最后一笔交易结果
            last_trade = trades_df.iloc[-1] if len(trades_df) > 0 else None
            last_result = 'win' if last_trade is not None and last_trade['profit_percentage'] > 0 else 'loss'
            
            # 计算平均风险收益比
            if 'risk_reward_ratio' in trades_df.columns:
                avg_risk_reward = trades_df['risk_reward_ratio'].mean()
            else:
                avg_risk_reward = 0.0
                
            # 按交易方向分组统计
            if 'direction' in trades_df.columns:
                direction_stats = trades_df.groupby('direction')['profit_percentage'].agg(['count', 'mean', lambda x: (x > 0).mean()])
                direction_stats.columns = ['count', 'avg_profit', 'win_rate']
                
                long_stats = direction_stats.loc['多'] if '多' in direction_stats.index else pd.Series({'count': 0, 'avg_profit': 0.0, 'win_rate': 0.0})
                short_stats = direction_stats.loc['空'] if '空' in direction_stats.index else pd.Series({'count': 0, 'avg_profit': 0.0, 'win_rate': 0.0})
            else:
                long_stats = pd.Series({'count': 0, 'avg_profit': 0.0, 'win_rate': 0.0})
                short_stats = pd.Series({'count': 0, 'avg_profit': 0.0, 'win_rate': 0.0})
                
            # 交易质量分析
            quality_stats = {}
            if 'trade_quality' in trades_df.columns or any('trade_quality' in t for t in self.completed_trades):
                trades_with_quality = [t for t in self.completed_trades if 'trade_quality' in t]
                
                if trades_with_quality:
                    quality_df = pd.DataFrame([
                        {
                            'trade_id': t['trade_id'],
                            'quality_score': t['trade_quality']['total_score'],
                            'is_profitable': t['profit_percentage'] > 0,
                            'profit_percentage': t['profit_percentage']
                        }
                        for t in trades_with_quality
                    ])
                    
                    # 按质量分组
                    quality_df['quality_category'] = pd.cut(
                        quality_df['quality_score'], 
                        bins=[0, 5, 7, 10], 
                        labels=['low', 'medium', 'high']
                    )
                    
                    # 计算各质量组的成功率
                    quality_metrics = quality_df.groupby('quality_category')['is_profitable'].agg(
                        ['mean', 'count']
                    ).to_dict()
                    
                    # 平均质量分数
                    avg_quality_score = quality_df['quality_score'].mean()
                    
                    quality_stats = {
                        'avg_quality_score': avg_quality_score,
                        'high_quality_count': quality_metrics['count'].get('high', 0),
                        'medium_quality_count': quality_metrics['count'].get('medium', 0),
                        'low_quality_count': quality_metrics['count'].get('low', 0),
                        'high_quality_win_rate': quality_metrics['mean'].get('high', 0.0),
                        'medium_quality_win_rate': quality_metrics['mean'].get('medium', 0.0),
                        'low_quality_win_rate': quality_metrics['mean'].get('low', 0.0)
                    }
                
                    # 计算质量分数与利润的相关性
                    if len(quality_df) > 1:
                        quality_stats['quality_profit_correlation'] = quality_df['quality_score'].corr(
                            quality_df['profit_percentage']
                        )
                    else:
                        quality_stats['quality_profit_correlation'] = 0.0
            
            # SBS指标
            sbs_metrics = self.calculate_sbs_metrics()
            
            # 探索交易分析
            exploration_analysis = self._analyze_exploration_effectiveness()
                
        except Exception as e:
            self.logger.error(f"使用pandas计算交易统计时出错: {e}")
            # 回退到原始计算方法
            return self._get_trade_stats_legacy()
            
        # 计算其他高级指标
        sharpe_ratio = self.calculate_sharpe_ratio()
        sortino_ratio = self.calculate_sortino_ratio()
            
        return {
            'active_trades': active_count,
            'active_profit': active_profit,
            'completed_trades': completed_count,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'last_trade_result': last_result,
            'current_streak': streak_metrics['current_streak'],
            'longest_win_streak': streak_metrics['longest_win_streak'],
            'longest_loss_streak': streak_metrics['longest_loss_streak'],
            'avg_win_duration': avg_win_duration,
            'avg_loss_duration': avg_loss_duration,
            'avg_risk_reward': avg_risk_reward,
            'long_trades_count': int(long_stats['count']),
            'short_trades_count': int(short_stats['count']),
            'long_win_rate': float(long_stats['win_rate']),
            'short_win_rate': float(short_stats['win_rate']),
            'quality_stats': quality_stats,
            'sbs_metrics': sbs_metrics,
            'exploration': {
                'enabled': self.exploration_enabled,
                'current_rate': self.exploration_rate,
                'count': exploration_analysis.get('exploration_count', 0),
                'success_rate': exploration_analysis.get('exploration_success_rate', 0.0),
                'standard_success_rate': exploration_analysis.get('standard_success_rate', 0.0),
                'overall_effect': exploration_analysis.get('overall_effect', 0.0)
            }
        }
        
    def _get_trade_stats_legacy(self) -> Dict:
        """
        原始的交易统计计算方法，作为pandas方法的备份
        """
        # 统计活跃交易
        active_count = len(self.active_trades)
        active_profit = sum([t['profit_percentage'] for t in self.active_trades.values()])
        
        # 统计已完成交易
        completed_count = len(self.completed_trades)
        if completed_count == 0:
            return {
                'active_trades': active_count,
                'active_profit': active_profit,
                'completed_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'max_drawdown': 0.0
            }
            
        # 计算胜率
        winning_trades = [t for t in self.completed_trades if t['profit_percentage'] > 0]
        losing_trades = [t for t in self.completed_trades if t['profit_percentage'] <= 0]
        
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        
        win_rate = winning_count / completed_count
        
        # 计算平均利润
        avg_profit = np.mean([t['profit_percentage'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t['profit_percentage'] for t in losing_trades]) if losing_trades else 0.0
        
        # 计算总利润和总亏损
        total_profit = sum([t['profit_percentage'] for t in winning_trades])
        total_loss = abs(sum([t['profit_percentage'] for t in losing_trades]))
        
        # 计算盈亏比
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # 计算期望值
        expectancy = (win_rate * avg_profit) + ((1 - win_rate) * avg_loss)
        
        # 计算最大回撤
        trades_by_time = sorted(self.completed_trades, key=lambda t: t.get('exit_time', ''))
        equity_curve = [0]
        
        for trade in trades_by_time:
            equity_curve.append(equity_curve[-1] + trade['profit_percentage'])
            
        max_equity = 0
        max_drawdown = 0
        
        for equity in equity_curve:
            if equity > max_equity:
                max_equity = equity
            drawdown = max_equity - equity
            max_drawdown = max(max_drawdown, drawdown)
        
        # 计算其他高级指标
        streak_metrics = self.calculate_streak_metrics()
        sharpe_ratio = self.calculate_sharpe_ratio()
        sortino_ratio = self.calculate_sortino_ratio()
        
        # 计算平均交易持续时间
        avg_win_duration = np.mean([t.get('duration', 0) for t in winning_trades]) if winning_trades else 0.0
        avg_loss_duration = np.mean([t.get('duration', 0) for t in losing_trades]) if losing_trades else 0.0
        
        # 获取最后一笔交易结果
        last_trade = trades_by_time[-1] if trades_by_time else None
        last_result = 'win' if last_trade and last_trade['profit_percentage'] > 0 else 'loss'
        
        # 计算平均风险收益比
        trades_with_rr = [t for t in self.completed_trades if 'risk_reward_ratio' in t]
        avg_risk_reward = np.mean([t['risk_reward_ratio'] for t in trades_with_rr]) if trades_with_rr else 0.0
        
        return {
            'active_trades': active_count,
            'active_profit': active_profit,
            'completed_trades': completed_count,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'last_trade_result': last_result,
            'current_streak': streak_metrics['current_streak'],
            'longest_win_streak': streak_metrics['longest_win_streak'],
            'longest_loss_streak': streak_metrics['longest_loss_streak'],
            'avg_win_duration': avg_win_duration,
            'avg_loss_duration': avg_loss_duration,
            'avg_risk_reward': avg_risk_reward
        }
        
    def get_trade(self, trade_id: str) -> Optional[Dict]:
        """获取指定交易的信息"""
        # 首先在活跃交易中查找
        if trade_id in self.active_trades:
            return self.active_trades[trade_id]
            
        # 然后在已完成交易中查找
        for trade in self.completed_trades:
            if trade['trade_id'] == trade_id:
                return trade
                
        return None
        
    def get_active_trades(self) -> List[Dict]:
        """获取所有活跃交易"""
        return list(self.active_trades.values())
        
    def get_completed_trades(self, limit: int = None) -> List[Dict]:
        """获取已完成交易"""
        if limit is None:
            return self.completed_trades
        else:
            return self.completed_trades[-limit:]
            
    def export_trades(self, filepath: Optional[str] = None) -> str:
        """
        导出交易数据
        
        Args:
            filepath: 导出文件路径，如果为None则使用默认路径
            
        Returns:
            导出文件路径
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.storage_dir / f"trade_export_{timestamp}.json"
            
        # 准备导出数据
        export_data = {
            'active_trades': list(self.active_trades.values()),
            'completed_trades': self.completed_trades,
            'stats': self.get_trade_stats(),
            'export_time': datetime.now().isoformat()
        }
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"交易数据已导出至: {filepath}")
        return str(filepath)
        
    def _calculate_profit(self, entry_price: float, current_price: float, direction: str) -> float:
        """计算盈亏百分比"""
        if direction == "多":
            return (current_price - entry_price) / entry_price * 100
        else:  # 空
            return (entry_price - current_price) / entry_price * 100
            
    def _calculate_duration(self, entry_time: str, exit_time: str) -> Optional[int]:
        """
        计算交易持续时间（分钟）
        
        Args:
            entry_time: 入场时间
            exit_time: 出场时间
            
        Returns:
            持续时间（分钟）
        """
        try:
            # 尝试多种时间格式解析
            entry_dt = self._parse_datetime(entry_time)
            exit_dt = self._parse_datetime(exit_time)
            
            if entry_dt and exit_dt:
                duration = (exit_dt - entry_dt).total_seconds() / 60
                return int(duration)
                
        except Exception as e:
            self.logger.warning(f"计算交易持续时间出错: {e}")
            
        return None
        
    def _parse_datetime(self, time_str: str) -> Optional[datetime]:
        """
        增强的时间戳解析方法，支持多种格式并具有异常处理
        
        Args:
            time_str: 时间字符串
            
        Returns:
            解析后的datetime对象，失败则返回None
        """
        if not time_str:
            return None
            
        # 尝试多种时间格式
        formats = [
            '%Y-%m-%dT%H:%M:%S.%f',  # ISO格式带毫秒
            '%Y-%m-%dT%H:%M:%S',     # ISO格式不带毫秒
            '%Y-%m-%d %H:%M:%S.%f',  # 标准格式带毫秒
            '%Y-%m-%d %H:%M:%S',     # 标准格式不带毫秒
            '%Y/%m/%d %H:%M:%S',     # 斜杠分隔日期
            '%d/%m/%Y %H:%M:%S',     # 欧洲日期格式
            '%m/%d/%Y %H:%M:%S',     # 美国日期格式
            '%Y%m%d%H%M%S'           # 紧凑格式
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
                
        # 尝试使用fromisoformat
        try:
            return datetime.fromisoformat(time_str)
        except (ValueError, AttributeError):
            pass
            
        # 尝试作为Unix时间戳解析
        try:
            timestamp = float(time_str)
            return datetime.fromtimestamp(timestamp)
        except (ValueError, TypeError, OverflowError):
            pass
            
        self.logger.warning(f"无法解析时间戳: {time_str}")
        return None
            
    def _save_trade_history(self) -> None:
        """保存交易历史记录"""
        history_file = self.storage_dir / "trade_history.json"
        
        try:
            # 保存交易历史
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.trade_history, f, ensure_ascii=False, indent=2)
                
            # 保存统计数据
            stats_file = self.storage_dir / "trade_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.get_trade_stats(), f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存交易历史记录失败: {str(e)}")
            
    def _load_trade_history(self) -> None:
        """加载交易历史记录"""
        history_file = self.storage_dir / "trade_history.json"
        
        if not history_file.exists():
            return
            
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                self.trade_history = json.load(f)
                
            # 分类活跃和已完成交易
            for trade in self.trade_history:
                if trade['status'] == 'open':
                    self.active_trades[trade['trade_id']] = trade
                else:
                    self.completed_trades.append(trade)
                    
            logger.info(f"已加载 {len(self.trade_history)} 条交易历史记录")
            
        except Exception as e:
            logger.error(f"加载交易历史记录失败: {str(e)}")
            
    def close_all_trades(self, 
                        current_prices: Dict[str, float],
                        exit_reason: str = 'force_close') -> List[Dict]:
        """
        关闭所有活跃交易
        
        Args:
            current_prices: 当前价格字典 {symbol: price}
            exit_reason: 出场原因
            
        Returns:
            已关闭的交易列表
        """
        closed_trades = []
        
        for trade_id, trade in list(self.active_trades.items()):
            symbol = trade['symbol']
            
            # 获取当前价格
            if symbol in current_prices:
                price = current_prices[symbol]
            else:
                # 如果没有提供价格，使用入场价格（假设盈亏为0）
                price = trade['entry_price']
                
            # 关闭交易
            closed_trade = self.close_trade(trade_id, price, exit_reason)
            if closed_trade:
                closed_trades.append(closed_trade)
                
        return closed_trades
        
    def analyze_performance(self, timeframe: str = 'monthly') -> Dict:
        """
        分析交易表现
        
        Args:
            timeframe: 分析时间段 ('daily', 'weekly', 'monthly')
            
        Returns:
            表现分析结果
        """
        if not self.completed_trades:
            return {'timeframes': [], 'win_rates': [], 'profits': []}
            
        # 按时间排序
        sorted_trades = sorted(self.completed_trades, key=lambda t: t['entry_time'])
        
        # 准备结果容器
        timeframes = []
        win_rates = []
        profits = []
        
        # 设置时间格式
        if timeframe == 'daily':
            fmt = '%Y-%m-%d'
        elif timeframe == 'weekly':
            fmt = '%Y-W%W'
        else:  # monthly
            fmt = '%Y-%m'
            
        # 分组数据
        groups = {}
        for trade in sorted_trades:
            try:
                entry_dt = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                period = entry_dt.strftime(fmt)
                
                if period not in groups:
                    groups[period] = []
                    
                groups[period].append(trade)
            except Exception as e:
                logger.error(f"分析交易表现时出错: {str(e)}")
                
        # 计算每个时间段的表现
        for period, trades in groups.items():
            winning = len([t for t in trades if t['profit_percentage'] > 0])
            total = len(trades)
            win_rate = winning / total if total > 0 else 0
            avg_profit = np.mean([t['profit_percentage'] for t in trades])
            
            timeframes.append(period)
            win_rates.append(win_rate)
            profits.append(avg_profit)
            
        return {
            'timeframes': timeframes,
            'win_rates': win_rates,
            'profits': profits
        } 
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        计算夏普比率 - 衡量风险调整后的回报
        
        Args:
            risk_free_rate: 无风险收益率 (默认为0)
            
        Returns:
            夏普比率
        """
        if not self.completed_trades:
            return 0.0
            
        # 获取所有交易的收益率
        returns = [trade['profit_percentage'] / 100 for trade in self.completed_trades]
        
        # 计算平均收益率和标准差
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # 避免除以零
        if std_return == 0:
            return 0.0
            
        # 计算夏普比率
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        计算索提诺比率 - 只考虑下行风险的回报衡量
        
        Args:
            risk_free_rate: 无风险收益率 (默认为0)
            
        Returns:
            索提诺比率
        """
        if not self.completed_trades:
            return 0.0
            
        # 获取所有交易的收益率
        returns = [trade['profit_percentage'] / 100 for trade in self.completed_trades]
        
        # 计算平均收益率
        mean_return = np.mean(returns)
        
        # 计算下行偏差（只考虑负收益的标准差）
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf') if mean_return > risk_free_rate else 0.0
            
        downside_deviation = np.sqrt(np.mean(np.square(negative_returns)))
        
        # 避免除以零
        if downside_deviation == 0:
            return 0.0
            
        # 计算索提诺比率
        sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
        
        return sortino_ratio
    
    def calculate_streak_metrics(self) -> Dict:
        """
        计算连续交易指标，包括最大连续盈利和亏损交易数
        
        Returns:
            连续交易指标字典
        """
        if not self.completed_trades:
            return {
                'max_win_streak': 0,
                'max_loss_streak': 0,
                'current_win_streak': 0,
                'current_loss_streak': 0
            }
            
        # 按时间排序
        sorted_trades = sorted(self.completed_trades, key=lambda t: t['exit_time'])
        
        # 计算连续交易序列
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in sorted_trades:
            if trade['profit_percentage'] > 0:
                # 盈利交易
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                # 亏损交易
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
                
        return {
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'current_win_streak': current_win_streak,
            'current_loss_streak': current_loss_streak
        }
        
    def calculate_sbs_metrics(self) -> Dict:
        """
        计算SBS特定的交易指标，包括流动性获取成功率、双顶双底形态的成功率等
        并结合交易质量评分进行分析
        
        Returns:
            SBS特定指标字典
        """
        if not self.completed_trades:
            return {
                'liquidation_success_rate': 0.0,
                'double_top_bottom_success_rate': 0.0,
                'sce_success_rate': 0.0,
                'sbs_completion_rate': 0.0,
                'high_quality_success_rate': 0.0,
                'medium_quality_success_rate': 0.0,
                'low_quality_success_rate': 0.0
            }
            
        # 筛选含有SBS相关元数据的交易
        sbs_trades = [t for t in self.completed_trades if 'metadata' in t and 'sbs_data' in t['metadata']]
        
        if not sbs_trades:
            return {
                'liquidation_success_rate': 0.0,
                'double_top_bottom_success_rate': 0.0,
                'sce_success_rate': 0.0,
                'sbs_completion_rate': 0.0,
                'high_quality_success_rate': 0.0,
                'medium_quality_success_rate': 0.0,
                'low_quality_success_rate': 0.0
            }
            
        try:
            # 使用pandas提高计算效率
            sbs_df = pd.DataFrame(sbs_trades)
            
            # 创建特征列
            sbs_df['is_liquidation'] = sbs_df['metadata'].apply(
                lambda x: x.get('sbs_data', {}).get('liquidation', False))
            sbs_df['is_double_pattern'] = sbs_df['metadata'].apply(
                lambda x: x.get('sbs_data', {}).get('double_pattern', False))
            sbs_df['is_sce'] = sbs_df['metadata'].apply(
                lambda x: x.get('sbs_data', {}).get('sce', False))
            sbs_df['is_complete_sequence'] = sbs_df['metadata'].apply(
                lambda x: x.get('sbs_data', {}).get('complete_sequence', False))
            sbs_df['is_profitable'] = sbs_df['profit_percentage'] > 0
            
            # 按特征分组计算成功率
            metrics = {}
            
            for feature in ['is_liquidation', 'is_double_pattern', 'is_sce', 'is_complete_sequence']:
                feature_trades = sbs_df[sbs_df[feature]]
                if len(feature_trades) > 0:
                    success_rate = feature_trades['is_profitable'].mean()
                    count = len(feature_trades)
                else:
                    success_rate = 0.0
                    count = 0
                
                metrics[f"{feature[3:]}_success_rate"] = success_rate
                metrics[f"{feature[3:]}_count"] = count
            
            # 整合交易质量分析
            if 'trade_quality' in sbs_df.columns or any('trade_quality' in t for t in sbs_trades):
                # 如果DataFrame没有trade_quality列，需要从原始数据提取
                trades_with_quality = [t for t in sbs_trades if 'trade_quality' in t]
                
                if trades_with_quality:
                    quality_df = pd.DataFrame([
                        {
                            'trade_id': t['trade_id'],
                            'quality_score': t['trade_quality']['total_score'],
                            'is_profitable': t['profit_percentage'] > 0,
                            'profit_percentage': t['profit_percentage'],
                            'is_liquidation': t.get('metadata', {}).get('sbs_data', {}).get('liquidation', False),
                            'is_double_pattern': t.get('metadata', {}).get('sbs_data', {}).get('double_pattern', False),
                            'is_sce': t.get('metadata', {}).get('sbs_data', {}).get('sce', False),
                            'is_complete_sequence': t.get('metadata', {}).get('sbs_data', {}).get('complete_sequence', False)
                        }
                        for t in trades_with_quality
                    ])
                    
                    # 按质量分组
                    quality_df['quality_category'] = pd.cut(
                        quality_df['quality_score'], 
                        bins=[0, 3, 7, 10], 
                        labels=['low', 'medium', 'high']
                    )
                    
                    # 计算各质量组的成功率
                    quality_metrics = quality_df.groupby('quality_category')['is_profitable'].agg(
                        ['mean', 'count']
                    ).to_dict()
                    
                    # 添加到指标中
                    for category in ['high', 'medium', 'low']:
                        if category in quality_metrics['mean']:
                            metrics[f"{category}_quality_success_rate"] = quality_metrics['mean'][category]
                            metrics[f"{category}_quality_count"] = quality_metrics['count'][category]
                        else:
                            metrics[f"{category}_quality_success_rate"] = 0.0
                            metrics[f"{category}_quality_count"] = 0
                    
                    # 计算质量分数与利润的相关性
                    if len(quality_df) > 1:
                        metrics['quality_profit_correlation'] = quality_df['quality_score'].corr(
                            quality_df['profit_percentage']
                        )
                    else:
                        metrics['quality_profit_correlation'] = 0.0
                        
                    # 按SBS特征和质量分类的组合分析
                    for feature in ['is_liquidation', 'is_double_pattern', 'is_sce', 'is_complete_sequence']:
                        feature_name = feature[3:]  # 移除'is_'前缀
                        
                        # 按特征和质量分组
                        if len(quality_df[quality_df[feature]]) > 0:
                            feature_quality_stats = quality_df[quality_df[feature]].groupby('quality_category')['is_profitable'].agg(
                                ['mean', 'count']
                            ).to_dict()
                            
                            # 添加高质量特征交易的成功率
                            if 'high' in feature_quality_stats['mean']:
                                metrics[f"high_quality_{feature_name}_success_rate"] = feature_quality_stats['mean']['high']
                                metrics[f"high_quality_{feature_name}_count"] = feature_quality_stats['count']['high']
                            else:
                                metrics[f"high_quality_{feature_name}_success_rate"] = 0.0
                                metrics[f"high_quality_{feature_name}_count"] = 0
            
        except Exception as e:
            self.logger.warning(f"计算SBS指标时出错: {e}")
            # 回退到原始计算方法
            
            # 统计不同SBS特征的交易
            liquidation_trades = [t for t in sbs_trades if t['metadata'].get('sbs_data', {}).get('liquidation', False)]
            double_pattern_trades = [t for t in sbs_trades if t['metadata'].get('sbs_data', {}).get('double_pattern', False)]
            sce_trades = [t for t in sbs_trades if t['metadata'].get('sbs_data', {}).get('sce', False)]
            complete_sbs = [t for t in sbs_trades if t['metadata'].get('sbs_data', {}).get('complete_sequence', False)]
            
            # 计算成功率
            liquidation_success = len([t for t in liquidation_trades if t['profit_percentage'] > 0])
            double_pattern_success = len([t for t in double_pattern_trades if t['profit_percentage'] > 0])
            sce_success = len([t for t in sce_trades if t['profit_percentage'] > 0])
            complete_sbs_success = len([t for t in complete_sbs if t['profit_percentage'] > 0])
            
            liquidation_success_rate = liquidation_success / len(liquidation_trades) if liquidation_trades else 0.0
            double_pattern_success_rate = double_pattern_success / len(double_pattern_trades) if double_pattern_trades else 0.0
            sce_success_rate = sce_success / len(sce_trades) if sce_trades else 0.0
            sbs_completion_rate = complete_sbs_success / len(complete_sbs) if complete_sbs else 0.0
            
            metrics = {
                'liquidation_success_rate': liquidation_success_rate,
                'double_top_bottom_success_rate': double_pattern_success_rate,
                'sce_success_rate': sce_success_rate,
                'sbs_completion_rate': sbs_completion_rate,
                'liquidation_count': len(liquidation_trades),
                'double_pattern_count': len(double_pattern_trades),
                'sce_count': len(sce_trades),
                'complete_sbs_count': len(complete_sbs)
            }
            
            # 尝试添加质量分析
            trades_with_quality = [t for t in sbs_trades if 'trade_quality' in t]
            if trades_with_quality:
                # 按质量分组
                high_quality = [t for t in trades_with_quality if t['trade_quality']['total_score'] >= 7.0]
                medium_quality = [t for t in trades_with_quality if 3.0 <= t['trade_quality']['total_score'] < 7.0]
                low_quality = [t for t in trades_with_quality if t['trade_quality']['total_score'] < 3.0]
                
                # 计算各质量组的成功率
                high_quality_success_rate = len([t for t in high_quality if t['profit_percentage'] > 0]) / len(high_quality) if high_quality else 0.0
                medium_quality_success_rate = len([t for t in medium_quality if t['profit_percentage'] > 0]) / len(medium_quality) if medium_quality else 0.0
                low_quality_success_rate = len([t for t in low_quality if t['profit_percentage'] > 0]) / len(low_quality) if low_quality else 0.0
                
                metrics.update({
                    'high_quality_success_rate': high_quality_success_rate,
                    'medium_quality_success_rate': medium_quality_success_rate,
                    'low_quality_success_rate': low_quality_success_rate,
                    'high_quality_count': len(high_quality),
                    'medium_quality_count': len(medium_quality),
                    'low_quality_count': len(low_quality)
                })
        
        return metrics
        
    def add_sbs_metadata(self, trade_id: str, sbs_data: Dict) -> bool:
        """
        添加SBS序列特定的元数据到交易记录中
        
        Args:
            trade_id: 交易ID
            sbs_data: SBS数据，包含序列点、类型等信息
            
        Returns:
            是否成功添加
        """
        # 首先检查交易是否存在
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"找不到交易ID: {trade_id}，无法添加SBS元数据")
            return False
            
        # 确保metadata字段存在
        if 'metadata' not in trade:
            trade['metadata'] = {}
            
        # 添加SBS数据
        trade['metadata']['sbs_data'] = sbs_data
        
        # 如果交易已关闭，更新交易历史
        if trade['status'] == 'closed':
            self._save_trade_history()
            
        logger.info(f"已添加SBS元数据到交易: {trade_id}")
        return True
        
    def record_sbs_sequence(self, 
                          trade_id: str,
                          point1: Dict = None,
                          point2: Dict = None,
                          point3: Dict = None,
                          point4: Dict = None,
                          point5: Dict = None,
                          liquidation: bool = False,
                          double_pattern: bool = False,
                          sce: bool = False,
                          complete_sequence: bool = False,
                          market_data: Dict = None) -> bool:
        """
        记录SBS序列点位信息
        
        Args:
            trade_id: 交易ID
            point1-point5: SBS序列各点位信息，每个点包含 {x, y, time}
            liquidation: 是否有流动性获取
            double_pattern: 是否为双顶/双底形态
            sce: 是否为SCE入场
            complete_sequence: 是否完成整个序列
            market_data: 市场数据（可选）
            
        Returns:
            是否成功记录
        """
        # 获取交易信息
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"找不到交易ID: {trade_id}，无法记录SBS序列")
            return False
            
        sbs_data = {
            'point1': point1,
            'point2': point2,
            'point3': point3,
            'point4': point4,
            'point5': point5,
            'liquidation': liquidation,
            'double_pattern': double_pattern,
            'sce': sce,
            'complete_sequence': complete_sequence,
            'record_time': datetime.now().isoformat()
        }
        
        # 确定确认信号类型
        confirmation_signal = None
        if sce:
            confirmation_signal = 'sce'
        elif double_pattern:
            confirmation_signal = 'double_pattern'
        elif liquidation:
            confirmation_signal = 'liquidity'
            
        # 计算交易质量评分
        sequence_points = {k: v for k, v in sbs_data.items() if k.startswith('point') and v}
        
        if confirmation_signal and sequence_points:
            trade_quality = self.calculate_trade_quality_score(
                sequence_points=sequence_points,
                confirmation_signal=confirmation_signal,
                entry_price=trade['entry_price'],
                stop_loss=trade['stop_loss'],
                take_profit=trade['take_profit'],
                market_data=market_data
            )
            
            # 添加交易质量评分
            trade['trade_quality'] = trade_quality
            sbs_data['trade_quality'] = trade_quality
            
            self.logger.info(f"交易 {trade_id} 的质量评分: {trade_quality['total_score']:.1f}/10")
        
        # 添加SBS数据到交易的元数据中
        if 'metadata' not in trade:
            trade['metadata'] = {}
            
        trade['metadata']['sbs_data'] = sbs_data
        
        # 如果交易已关闭，更新交易历史
        if trade['status'] == 'closed':
            self._save_trade_history()
            
        self.logger.info(f"已记录SBS序列到交易: {trade_id}")
        return True
        
    def get_visualization_data(self, limit: int = 20) -> Dict:
        """
        获取用于可视化的交易数据
        
        Args:
            limit: 获取的最近交易数量
            
        Returns:
            可视化数据字典
        """
        # 获取最近的交易
        recent_trades = self.get_completed_trades(limit)
        
        # 准备时间序列数据
        dates = []
        profits = []
        cumulative_profit = []
        
        running_total = 0
        for trade in recent_trades:
            try:
                exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00'))
                dates.append(exit_time.strftime('%Y-%m-%d %H:%M'))
                profit = trade['profit_percentage']
                profits.append(profit)
                
                running_total += profit
                cumulative_profit.append(running_total)
            except Exception as e:
                logger.error(f"准备可视化数据时出错: {str(e)}")
                
        # 准备SBS特定数据
        sbs_trades = [t for t in recent_trades if 'metadata' in t and 'sbs_data' in t.get('metadata', {})]
        
        sbs_points = {
            'point1': [],
            'point2': [],
            'point3': [],
            'point4': [],
            'point5': []
        }
        
        sbs_features = {
            'liquidation': 0,
            'double_pattern': 0,
            'sce': 0,
            'complete_sequence': 0
        }
        
        for trade in sbs_trades:
            sbs_data = trade['metadata']['sbs_data']
            
            # 收集点位数据
            for point_name in ['point1', 'point2', 'point3', 'point4', 'point5']:
                if point_name in sbs_data and sbs_data[point_name]:
                    sbs_points[point_name].append(sbs_data[point_name])
                    
            # 统计特征出现次数
            for feature in ['liquidation', 'double_pattern', 'sce', 'complete_sequence']:
                if feature in sbs_data and sbs_data[feature]:
                    sbs_features[feature] += 1
                    
        return {
            'dates': dates,
            'profits': profits,
            'cumulative_profit': cumulative_profit,
            'sbs_points': sbs_points,
            'sbs_features': sbs_features,
            'stats': self.get_trade_stats()
        }
        
    def calculate_trade_quality_score(self, 
                                     sequence_points: Dict,
                                     confirmation_signal: str,
                                     entry_price: float,
                                     stop_loss: float,
                                     take_profit: float,
                                     market_data: Dict = None) -> Dict:
        """
        计算交易质量综合评分
        
        Args:
            sequence_points: SBS序列的点位信息
            confirmation_signal: 确认信号类型
            entry_price: 入场价格
            stop_loss: 止损价格
            take_profit: 止盈价格
            market_data: 市场数据（可选）
            
        Returns:
            交易质量评分字典
        """
        scores = {}
        
        # 1. 序列完整性评分 (基于序列点位是否完整)
        sequence_score = 0
        required_points = ['point1', 'point2', 'point3', 'point4']
        for point in required_points:
            if point in sequence_points and sequence_points[point]:
                sequence_score += 2.5  # 每个点位2.5分，总分10分
        scores['sequence_completeness'] = sequence_score
        
        # 2. 确认信号清晰度评分
        signal_score = 0
        if confirmation_signal == 'double_pattern':
            signal_score = 9  # 双顶/双底信号是最明确的信号
        elif confirmation_signal == 'liquidity':
            signal_score = 8  # 流动性获取
        elif confirmation_signal == 'sce':
            signal_score = 8  # SCE信号
        elif confirmation_signal:
            signal_score = 6  # 其他确认信号
        scores['signal_clarity'] = signal_score
        
        # 3. 风险回报比评分
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk > 0:
            rrr = reward / risk
            # 将RRR映射到1-10分
            rrr_score = min(10, rrr * 3.33)  # RRR=3时得满分
        else:
            rrr_score = 1  # 风险为0时，给予最低分
        scores['risk_reward_ratio'] = rrr_score
        
        # 4. 入场时机评分 (如果有点4和确认信号，则认为入场时机良好)
        timing_score = 0
        if 'point4' in sequence_points and sequence_points['point4'] and confirmation_signal:
            timing_score = 9  # 有点4和确认信号，入场时机很好
        elif 'point4' in sequence_points and sequence_points['point4']:
            timing_score = 6  # 有点4但无明确确认信号
        elif confirmation_signal:
            timing_score = 5  # 有确认信号但点位不完整
        else:
            timing_score = 3  # 既无点4也无确认信号
        scores['entry_timing'] = timing_score
        
        # 5. 计算总体得分（加权平均）
        weights = {
            'sequence_completeness': 0.35,
            'signal_clarity': 0.25,
            'risk_reward_ratio': 0.25,
            'entry_timing': 0.15
        }
        
        total_score = sum(scores[key] * weights[key] for key in weights)
        scores['total_score'] = total_score
        
        return scores
        
    def _analyze_trade_quality(self) -> Dict:
        """
        分析交易质量评分及其与交易结果的关系
        
        Returns:
            交易质量分析结果
        """
        trades_with_quality = [t for t in self.completed_trades if 'trade_quality' in t]
        if not trades_with_quality:
            return {
                'trades_with_quality': 0,
                'quality_score_correlation': 0.0
            }
            
        # 提取质量评分和利润
        quality_scores = [t['trade_quality']['total_score'] for t in trades_with_quality]
        profits = [t['profit_percentage'] for t in trades_with_quality]
        
        # 计算相关性
        try:
            correlation = np.corrcoef(quality_scores, profits)[0, 1]
        except:
            correlation = 0.0
            
        # 按得分组统计
        score_buckets = {
            '9-10': {'count': 0, 'wins': 0, 'avg_profit': 0.0},
            '7-9': {'count': 0, 'wins': 0, 'avg_profit': 0.0},
            '5-7': {'count': 0, 'wins': 0, 'avg_profit': 0.0},
            '3-5': {'count': 0, 'wins': 0, 'avg_profit': 0.0},
            '0-3': {'count': 0, 'wins': 0, 'avg_profit': 0.0}
        }
        
        for trade in trades_with_quality:
            score = trade['trade_quality']['total_score']
            profit = trade['profit_percentage']
            is_win = profit > 0
            
            if score >= 9.0:
                bucket = '9-10'
            elif score >= 7.0:
                bucket = '7-9'
            elif score >= 5.0:
                bucket = '5-7'
            elif score >= 3.0:
                bucket = '3-5'
            else:
                bucket = '0-3'
                
            score_buckets[bucket]['count'] += 1
            if is_win:
                score_buckets[bucket]['wins'] += 1
            score_buckets[bucket]['avg_profit'] += profit
            
        # 计算平均值
        for bucket in score_buckets:
            if score_buckets[bucket]['count'] > 0:
                score_buckets[bucket]['win_rate'] = score_buckets[bucket]['wins'] / score_buckets[bucket]['count']
                score_buckets[bucket]['avg_profit'] = score_buckets[bucket]['avg_profit'] / score_buckets[bucket]['count']
            else:
                score_buckets[bucket]['win_rate'] = 0.0
                score_buckets[bucket]['avg_profit'] = 0.0
                
        # 按指标分析
        metrics_analysis = {}
        for metric in ['sequence_completeness', 'signal_clarity', 'risk_reward_ratio', 'entry_timing']:
            metric_scores = [t['trade_quality'][metric] for t in trades_with_quality]
            try:
                metric_correlation = np.corrcoef(metric_scores, profits)[0, 1]
            except:
                metric_correlation = 0.0
                
            metrics_analysis[metric] = {
                'correlation': metric_correlation,
                'avg_score': np.mean(metric_scores)
            }
            
        # 计算潜在收益分析
        potential_profit_analysis = self._analyze_potential_profit(trades_with_quality)
            
        return {
            'trades_with_quality': len(trades_with_quality),
            'quality_score_correlation': correlation,
            'score_buckets': score_buckets,
            'metrics_analysis': metrics_analysis,
            'potential_profit_analysis': potential_profit_analysis
        }
        
    def _analyze_potential_profit(self, trades_with_quality: List[Dict]) -> Dict:
        """
        分析潜在收益情况，评估交易的时机是否最优
        
        Args:
            trades_with_quality: 含有质量评分的交易列表
            
        Returns:
            潜在收益分析结果
        """
        if not trades_with_quality:
            return {
                'avg_profit_efficiency': 0.0,
                'missed_profit_percentage': 0.0
            }
            
        total_efficiency = 0.0
        total_missed = 0.0
        count = 0
        
        for trade in trades_with_quality:
            if 'max_profit_percentage' in trade and trade['profit_percentage'] != 0:
                # 计算盈利效率 (实际盈利 / 最大可能盈利)
                max_profit = trade['max_profit_percentage']
                actual_profit = trade['profit_percentage']
                
                if max_profit > 0:
                    if actual_profit > 0:
                        # 盈利交易的效率
                        efficiency = actual_profit / max_profit
                        missed = max_profit - actual_profit
                    else:
                        # 亏损交易，效率为0
                        efficiency = 0.0
                        missed = max_profit
                        
                    total_efficiency += efficiency
                    total_missed += missed
                    count += 1
        
        avg_efficiency = total_efficiency / count if count > 0 else 0.0
        avg_missed = total_missed / count if count > 0 else 0.0
        
        return {
            'avg_profit_efficiency': avg_efficiency,
            'missed_profit_percentage': avg_missed,
            'trades_analyzed': count
        }
        
    def export_for_labelstudio(self, filepath: Optional[str] = None) -> str:
        """
        导出交易数据为LabelStudio预标注格式
        
        Args:
            filepath: 导出文件路径，如果为None则使用默认路径
            
        Returns:
            导出文件路径
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.storage_dir / f"labelstudio_export_{timestamp}.json"
            
        # 过滤有SBS序列数据的交易
        sbs_trades = [t for t in self.completed_trades if 'metadata' in t and 'sbs_data' in t.get('metadata', {})]
        
        if not sbs_trades:
            logger.warning("没有包含SBS序列数据的交易可导出")
            return ""
            
        # 准备LabelStudio格式数据
        label_studio_data = []
        export_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        for i, trade in enumerate(sbs_trades):
            sbs_data = trade['metadata']['sbs_data']
            trade_timestamp = trade.get('entry_time', '').replace(':', '-').replace(' ', '_')
            
            # 生成唯一ID (结合序列号、交易ID和时间戳)
            unique_id = f"{export_timestamp}_{i+1:04d}_{trade['trade_id']}"
            
            # 构建矩形标注数据
            annotations = []
            
            for point_name in ['point1', 'point2', 'point3', 'point4', 'point5']:
                if point_name in sbs_data and sbs_data[point_name]:
                    point = sbs_data[point_name]
                    if 'x' in point and 'y' in point:
                        # LabelStudio矩形格式
                        x = point['x']
                        y = point['y']
                        width = 2.0  # 默认宽度
                        height = 2.0  # 默认高度
                        
                        annotation = {
                            "id": f"{unique_id}_{point_name}",
                            "type": "rectanglelabels",
                            "value": {
                                "x": x,
                                "y": y,
                                "width": width,
                                "height": height,
                                "rotation": 0,
                                "rectanglelabels": [point_name]
                            }
                        }
                        annotations.append(annotation)
            
            # 添加方向标注
            direction_annotation = {
                "id": f"{unique_id}_direction",
                "type": "choices",
                "value": {
                    "choices": [trade['direction'] == '多' and 'bullish' or 'bearish']
                }
            }
            annotations.append(direction_annotation)
            
            # 添加确认信号标注
            signal_types = []
            if sbs_data.get('double_pattern', False):
                signal_types.append("double_pattern")
            if sbs_data.get('liquidation', False):
                signal_types.append("liquidity")
            if sbs_data.get('sce', False):
                signal_types.append("sce")
                
            if signal_types:
                signal_annotation = {
                    "id": f"{unique_id}_signal",
                    "type": "choices",
                    "value": {
                        "choices": signal_types
                    }
                }
                annotations.append(signal_annotation)
            
            # 构建更详细的图片文件名
            # 格式: 日期_时间_交易ID_品种.png
            image_filename = f"{trade_timestamp}_{trade['trade_id']}_{trade['symbol']}.png"
            
            # 构建完整的任务项
            task_item = {
                "id": unique_id,
                "data": {
                    "image": f"charts/{image_filename}",
                    "symbol": trade['symbol'],
                    "timeframe": trade.get('timeframe', 'unknown'),
                    "entry_time": trade.get('entry_time', ''),
                    "direction": trade['direction'],
                    "trade_id": trade['trade_id']
                },
                "annotations": [{
                    "id": f"{unique_id}_annotation",
                    "result": annotations
                }]
            }
            
            # 添加交易结果信息作为参考
            if 'profit_percentage' in trade:
                task_item["data"]["profit_percentage"] = trade['profit_percentage']
                task_item["data"]["trade_result"] = "win" if trade['profit_percentage'] > 0 else "loss"
            
            label_studio_data.append(task_item)
            
        # 保存到文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(label_studio_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"LabelStudio预标注数据已导出至: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"导出LabelStudio数据失败: {str(e)}")
            return ""
        
    def analyze_sbs_effectiveness(self) -> Dict:
        """
        分析SBS交易策略的有效性
        
        Returns:
            SBS策略有效性分析结果
        """
        if not self.completed_trades:
            return {
                'overall_effectiveness': 0.0,
                'pattern_effectiveness': {}
            }
            
        # 获取包含SBS元数据的交易
        sbs_trades = [t for t in self.completed_trades if 'metadata' in t and 'sbs_data' in t.get('metadata', {})]
        
        if not sbs_trades:
            return {
                'overall_effectiveness': 0.0,
                'pattern_effectiveness': {}
            }
            
        # 计算总体有效性
        profitable_sbs = [t for t in sbs_trades if t['profit_percentage'] > 0]
        overall_effectiveness = len(profitable_sbs) / len(sbs_trades)
        
        # 分析各种模式的有效性
        pattern_counts = {
            'liquidation': 0,
            'double_pattern': 0,
            'sce': 0,
            'complete_sequence': 0,
            'point1_only': 0,
            'points1_2': 0,
            'points1_2_3': 0,
            'points1_2_3_4': 0,
            'full_sequence': 0
        }
        
        pattern_profits = {
            'liquidation': [],
            'double_pattern': [],
            'sce': [],
            'complete_sequence': [],
            'point1_only': [],
            'points1_2': [],
            'points1_2_3': [],
            'points1_2_3_4': [],
            'full_sequence': []
        }
        
        for trade in sbs_trades:
            sbs_data = trade['metadata']['sbs_data']
            profit = trade['profit_percentage']
            
            # 分析特征
            for feature in ['liquidation', 'double_pattern', 'sce', 'complete_sequence']:
                if feature in sbs_data and sbs_data[feature]:
                    pattern_counts[feature] += 1
                    pattern_profits[feature].append(profit)
                    
            # 分析序列完成度
            points_present = sum(1 for p in ['point1', 'point2', 'point3', 'point4', 'point5'] 
                               if p in sbs_data and sbs_data[p])
                               
            if points_present == 1:
                pattern_counts['point1_only'] += 1
                pattern_profits['point1_only'].append(profit)
            elif points_present == 2:
                pattern_counts['points1_2'] += 1
                pattern_profits['points1_2'].append(profit)
            elif points_present == 3:
                pattern_counts['points1_2_3'] += 1
                pattern_profits['points1_2_3'].append(profit)
            elif points_present == 4:
                pattern_counts['points1_2_3_4'] += 1
                pattern_profits['points1_2_3_4'].append(profit)
            elif points_present == 5:
                pattern_counts['full_sequence'] += 1
                pattern_profits['full_sequence'].append(profit)
                
        # 计算各模式的有效性
        pattern_effectiveness = {}
        for pattern, profits in pattern_profits.items():
            if profits:
                win_rate = len([p for p in profits if p > 0]) / len(profits)
                avg_profit = np.mean(profits)
                count = pattern_counts[pattern]
                
                pattern_effectiveness[pattern] = {
                    'count': count,
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'effectiveness': win_rate * avg_profit if avg_profit > 0 else 0
                }
            else:
                pattern_effectiveness[pattern] = {
                    'count': 0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'effectiveness': 0.0
                }
                
        # 添加交易质量分析
        quality_analysis = self._analyze_trade_quality()
        
        return {
            'overall_effectiveness': overall_effectiveness,
            'pattern_effectiveness': pattern_effectiveness,
            'total_sbs_trades': len(sbs_trades),
            'profitable_sbs_trades': len(profitable_sbs),
            'quality_analysis': quality_analysis
        }
        
    def generate_detailed_report(self, filepath: Optional[str] = None, include_trade_details: bool = True) -> str:
        """
        生成详细的交易报告
        
        Args:
            filepath: 导出文件路径，如果为None则使用默认路径
            include_trade_details: 是否包含每笔交易的详细信息
            
        Returns:
            报告文件路径
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.storage_dir / f"trade_report_{timestamp}.json"
            
        # 获取统计数据
        stats = self.get_trade_stats()
        
        # 获取SBS有效性分析
        sbs_effectiveness = self.analyze_sbs_effectiveness()
        
        # 获取时间序列表现
        daily_performance = self.analyze_performance('daily')
        weekly_performance = self.analyze_performance('weekly')
        monthly_performance = self.analyze_performance('monthly')
        
        # 计算交易持续时间分布
        duration_distribution = self._calculate_duration_distribution()
        
        # 计算盈利分布
        profit_distribution = self._calculate_profit_distribution()
        
        # 分析探索策略效果
        exploration_analysis = self._analyze_exploration_effectiveness()
        
        # 添加交易质量分析
        quality_analysis = self._analyze_trade_quality()
        
        # 准备报告数据
        report_data = {
            'report_time': datetime.now().isoformat(),
            'report_type': 'detailed_trade_analysis',
            'summary_stats': stats,
            'sbs_analysis': sbs_effectiveness,
            'performance_over_time': {
                'daily': daily_performance,
                'weekly': weekly_performance,
                'monthly': monthly_performance
            },
            'duration_distribution': duration_distribution,
            'profit_distribution': profit_distribution,
            'streak_analysis': self.calculate_streak_metrics(),
            'exploration_analysis': exploration_analysis,
            'quality_analysis': quality_analysis
        }
        
        # 如果要包含交易详情
        if include_trade_details:
            report_data['trade_details'] = {
                'active_trades': list(self.active_trades.values()),
                'completed_trades': self.completed_trades
            }
            
        # 保存到文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"详细交易报告已生成: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"生成详细报告失败: {str(e)}")
            return ""
        
    def _calculate_duration_distribution(self) -> Dict:
        """计算交易持续时间分布"""
        if not self.completed_trades:
            return {
                'bins': [],
                'counts': []
            }
            
        # 提取持续时间（小时）
        durations = []
        for trade in self.completed_trades:
            try:
                entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00'))
                duration_hours = (exit_time - entry_time).total_seconds() / 3600
                durations.append(duration_hours)
            except Exception as e:
                logger.error(f"计算交易持续时间时出错: {e}")
                
        if not durations:
            return {'bins': [], 'counts': []}
            
        # 创建时间分布区间
        bins = [0, 1, 4, 12, 24, 48, 72, 168, float('inf')]  # 小时: 0-1, 1-4, 4-12, 12-24, 24-48, 48-72, 72-168, >168
        bin_labels = ['<1h', '1-4h', '4-12h', '12-24h', '1-2d', '2-3d', '3-7d', '>7d']
        
        # 统计每个区间的交易数量
        counts = [0] * (len(bins) - 1)
        for duration in durations:
            for i in range(len(bins) - 1):
                if bins[i] <= duration < bins[i+1]:
                    counts[i] += 1
                    break
                    
        return {
            'bin_labels': bin_labels,
            'counts': counts,
            'percentages': [count / len(durations) * 100 for count in counts]
        }
        
    def _calculate_profit_distribution(self) -> Dict:
        """计算盈利分布"""
        if not self.completed_trades:
            return {
                'bins': [],
                'counts': []
            }
            
        # 提取盈利百分比
        profits = [trade['profit_percentage'] for trade in self.completed_trades]
        
        if not profits:
            return {'bins': [], 'counts': []}
            
        # 创建利润分布区间
        bins = [-float('inf'), -20, -10, -5, 0, 5, 10, 20, float('inf')]  # 百分比
        bin_labels = ['<-20%', '-20~-10%', '-10~-5%', '-5~0%', '0~5%', '5~10%', '10~20%', '>20%']
        
        # 统计每个区间的交易数量
        counts = [0] * (len(bins) - 1)
        for profit in profits:
            for i in range(len(bins) - 1):
                if bins[i] <= profit < bins[i+1]:
                    counts[i] += 1
                    break
                    
        return {
            'bin_labels': bin_labels,
            'counts': counts,
            'percentages': [count / len(profits) * 100 for count in counts]
        }
        
    def export_trade_csv(self, filepath: Optional[str] = None) -> str:
        """
        将交易数据导出为CSV格式
        
        Args:
            filepath: CSV文件路径，如果为None则使用默认路径
            
        Returns:
            CSV文件路径
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.storage_dir / f"trade_export_{timestamp}.csv"
            
        if not self.completed_trades:
            logger.warning("没有已完成的交易可导出")
            return ""
            
        try:
            # 构建DataFrame
            trades_data = []
            
            for trade in self.completed_trades:
                # 提取基本字段
                trade_data = {
                    'trade_id': trade['trade_id'],
                    'symbol': trade['symbol'],
                    'direction': trade['direction'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'entry_time': trade['entry_time'],
                    'exit_time': trade['exit_time'],
                    'profit_percentage': trade['profit_percentage'],
                    'max_drawdown': trade['max_drawdown'],
                    'exit_reason': trade['exit_reason'],
                    'duration': trade['duration'],
                    'risk_reward_ratio': trade['risk_reward_ratio']
                }
                
                # 提取SBS相关数据
                if 'metadata' in trade and 'sbs_data' in trade['metadata']:
                    sbs_data = trade['metadata']['sbs_data']
                    trade_data.update({
                        'liquidation': sbs_data.get('liquidation', False),
                        'double_pattern': sbs_data.get('double_pattern', False),
                        'sce': sbs_data.get('sce', False),
                        'complete_sequence': sbs_data.get('complete_sequence', False)
                    })
                    
                trades_data.append(trade_data)
                
            # 创建DataFrame并保存为CSV
            df = pd.DataFrame(trades_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            logger.info(f"交易数据已导出为CSV: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"导出CSV失败: {str(e)}")
            return ""
        
    def get_exploration_metrics(self) -> Dict:
        """
        获取探索策略的指标
        
        Returns:
            探索指标字典
        """
        total = self.exploration_metrics['total_explorations']
        successful = self.exploration_metrics['successful_explorations']
        
        return {
            'exploration_enabled': self.exploration_enabled,
            'exploration_rate': self.exploration_rate,
            'min_exploration_rate': self.min_exploration_rate,
            'exploration_decay': self.exploration_decay,
            'total_explorations': total,
            'successful_explorations': successful,
            'success_rate': successful / max(1, total),
            'exploration_trades_count': len(self.exploration_metrics['exploration_trades'])
        }
        
    def _analyze_exploration_effectiveness(self) -> Dict:
        """
        分析探索策略的有效性，限制为SBS交易探索
        
        Returns:
            探索策略分析结果
        """
        # 获取探索交易和标准交易
        exploration_trades = [t for t in self.completed_trades if 
                             'metadata' in t and 
                             'exploration' in t.get('metadata', {}) and 
                             t['metadata']['exploration'].get('is_exploration', False) and
                             'sbs_data' in t.get('metadata', {})]  # 限制为SBS交易
                             
        standard_trades = [t for t in self.completed_trades if 
                          t not in exploration_trades and
                          'metadata' in t and 
                          'sbs_data' in t.get('metadata', {})]  # 同样限制为SBS交易
        
        if not exploration_trades:
            return {
                'exploration_count': 0,
                'exploration_enabled': self.exploration_enabled,
                'current_exploration_rate': self.exploration_rate,
                'sbs_focused': True
            }
        
        # 按SBS特征类型分组统计
        exploration_by_feature = {
            'liquidation': {'count': 0, 'success': 0, 'avg_profit': 0.0},
            'double_pattern': {'count': 0, 'success': 0, 'avg_profit': 0.0},
            'sce': {'count': 0, 'success': 0, 'avg_profit': 0.0},
            'complete_sequence': {'count': 0, 'success': 0, 'avg_profit': 0.0}
        }
        
        # 探索参数调整类型统计
        exploration_by_param = {
            'direction': {'count': 0, 'success': 0, 'avg_profit': 0.0},
            'stop_loss': {'count': 0, 'success': 0, 'avg_profit': 0.0},
            'take_profit': {'count': 0, 'success': 0, 'avg_profit': 0.0},
            'risk': {'count': 0, 'success': 0, 'avg_profit': 0.0}
        }
        
        # 统计探索交易的表现
        for trade in exploration_trades:
            # 获取交易利润
            profit = trade.get('profit_percentage', 0.0)
            is_success = profit > 0
            
            # 按SBS特征分类统计
            sbs_data = trade.get('metadata', {}).get('sbs_data', {})
            for feature in ['liquidation', 'double_pattern', 'sce', 'complete_sequence']:
                if sbs_data.get(feature, False):
                    exploration_by_feature[feature]['count'] += 1
                    if is_success:
                        exploration_by_feature[feature]['success'] += 1
                    exploration_by_feature[feature]['avg_profit'] += profit
            
            # 按探索参数类型统计
            exploration_data = trade.get('metadata', {}).get('exploration', {})
            original_params = exploration_data.get('original_params', {})
            
            # 确定探索类型
            param_type = None
            if 'direction' in original_params and original_params['direction'] != trade['direction']:
                param_type = 'direction'
            elif 'stop_loss' in original_params and original_params['stop_loss'] != trade['stop_loss']:
                param_type = 'stop_loss'
            elif 'take_profit' in original_params and original_params['take_profit'] != trade['take_profit']:
                param_type = 'take_profit'
            elif 'risk_percentage' in original_params and original_params['risk_percentage'] != trade['risk_percentage']:
                param_type = 'risk'
            
            if param_type:
                exploration_by_param[param_type]['count'] += 1
                if is_success:
                    exploration_by_param[param_type]['success'] += 1
                exploration_by_param[param_type]['avg_profit'] += profit
        
        # 计算平均值
        for stats in [exploration_by_feature, exploration_by_param]:
            for feature, data in stats.items():
                if data['count'] > 0:
                    data['success_rate'] = data['success'] / data['count']
                    data['avg_profit'] = data['avg_profit'] / data['count']
                else:
                    data['success_rate'] = 0.0
                    data['avg_profit'] = 0.0
        
        # 计算标准交易表现
        std_count = len(standard_trades)
        std_success = len([t for t in standard_trades if t.get('profit_percentage', 0.0) > 0])
        std_success_rate = std_success / std_count if std_count > 0 else 0.0
        std_avg_profit = sum([t.get('profit_percentage', 0.0) for t in standard_trades]) / std_count if std_count > 0 else 0.0
        
        # 计算整体探索表现
        exp_count = len(exploration_trades)
        exp_success = len([t for t in exploration_trades if t.get('profit_percentage', 0.0) > 0])
        exp_success_rate = exp_success / exp_count if exp_count > 0 else 0.0
        exp_avg_profit = sum([t.get('profit_percentage', 0.0) for t in exploration_trades]) / exp_count if exp_count > 0 else 0.0
        
        # 比较探索与标准交易
        success_rate_diff = exp_success_rate - std_success_rate
        profit_diff = exp_avg_profit - std_avg_profit
        
        # 计算整体效果（结合成功率和利润差异）
        overall_effect = (success_rate_diff * 0.5) + (profit_diff / 100 * 0.5)
        
        return {
            'sbs_focused': True,
            'exploration_count': exp_count,
            'standard_count': std_count,
            'exploration_success_rate': exp_success_rate,
            'standard_success_rate': std_success_rate,
            'exploration_avg_profit': exp_avg_profit,
            'standard_avg_profit': std_avg_profit,
            'success_rate_difference': success_rate_diff,
            'profit_difference': profit_diff,
            'overall_effect': overall_effect,
            'exploration_by_feature': exploration_by_feature,
            'exploration_by_param': exploration_by_param,
            'exploration_enabled': self.exploration_enabled,
            'current_exploration_rate': self.exploration_rate
        } 

    def update_config(self, new_config: Dict):
        """
        更新配置参数
        
        Args:
            new_config: 新配置参数
        """
        self.config.update(new_config)
        
        # 更新探索相关参数
        if 'exploration_enabled' in new_config:
            self.exploration_enabled = new_config['exploration_enabled']
            
        if 'exploration_rate' in new_config:
            self.exploration_rate = new_config['exploration_rate']
            
        if 'min_exploration_rate' in new_config:
            self.min_exploration_rate = new_config['min_exploration_rate']
            
        if 'exploration_decay' in new_config:
            self.exploration_decay = new_config['exploration_decay']
            
        if 'exploration_boost_interval' in new_config:
            self.exploration_boost_interval = new_config['exploration_boost_interval']
            
        if 'exploration_boost_factor' in new_config:
            self.exploration_boost_factor = new_config['exploration_boost_factor']
            
        if 'exploration_success_threshold' in new_config:
            self.exploration_success_threshold = new_config['exploration_success_threshold']
            
        if 'exploration_failure_threshold' in new_config:
            self.exploration_failure_threshold = new_config['exploration_failure_threshold']
            
        if 'exploration_success_rate_adjust' in new_config:
            self.exploration_success_rate_adjust = new_config['exploration_success_rate_adjust']
            
        if 'exploration_failure_rate_adjust' in new_config:
            self.exploration_failure_rate_adjust = new_config['exploration_failure_rate_adjust']
            
        self.logger.info(f"配置已更新: {new_config}")
        
    def _memory_cleanup(self):
        """内存清理和优化"""
        try:
            # 检查是否需要清理
            if len(self.completed_trades) <= self.max_trades_in_memory:
                return

            # 计算需要归档的交易数量
            trades_to_archive = len(self.completed_trades) - self.max_trades_in_memory + self.trade_archive_threshold

            # 按时间排序，归档最早的交易
            sorted_trades = sorted(
                self.completed_trades,
                key=lambda x: x.get('entry_time', ''),
                reverse=False
            )

            # 提取要归档的交易
            trades_to_save = sorted_trades[:trades_to_archive]
            
            # 压缩并保存到文件
            if trades_to_save:
                import gzip
                import json
                
                # 按月份分组保存
                from itertools import groupby
                from datetime import datetime
                
                def get_month_key(trade):
                    try:
                        entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                        return f"{entry_time.year:04d}-{entry_time.month:02d}"
                    except:
                        return "unknown"
                
                # 按月份分组
                trades_to_save.sort(key=get_month_key)
                for month_key, month_trades in groupby(trades_to_save, key=get_month_key):
                    month_trades = list(month_trades)
                    
                    # 创建月度归档文件
                    archive_path = Path(self.storage_dir) / 'archives' / f"trades_{month_key}.json.gz"
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 如果文件已存在，先读取并合并
                    existing_trades = []
                    if archive_path.exists():
                        with gzip.open(archive_path, 'rt', encoding='utf-8') as f:
                            try:
                                existing_trades = json.load(f)
                            except:
                                pass
                    
                    # 合并并保存
                    all_trades = existing_trades + month_trades
                    with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
                        json.dump(all_trades, f, ensure_ascii=False)
                
                # 从内存中移除已归档的交易
                trade_ids_to_remove = {t['trade_id'] for t in trades_to_save}
                self.completed_trades = [t for t in self.completed_trades if t['trade_id'] not in trade_ids_to_remove]
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                logger.info(f"已归档{len(trades_to_save)}笔交易，当前内存中剩余{len(self.completed_trades)}笔交易")
                
        except Exception as e:
            logger.error(f"内存清理过程出错: {e}")

    def load_archived_trades(self, start_date: str = None, end_date: str = None) -> int:
        """惰性加载归档的交易数据"""
        try:
            archive_dir = Path(self.storage_dir) / 'archives'
            if not archive_dir.exists():
                return 0
                
            # 解析日期范围
            start_dt = None
            end_dt = None
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date)
                except:
                    logger.warning(f"无效的开始日期格式: {start_date}")
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date)
                except:
                    logger.warning(f"无效的结束日期格式: {end_date}")
                    
            # 获取所有归档文件
            archive_files = sorted(archive_dir.glob("trades_*.json.gz"))
            loaded_count = 0
            
            for archive_file in archive_files:
                # 从文件名提取月份
                try:
                    month_str = archive_file.stem.split('_')[1].split('.')[0]
                    file_dt = datetime.strptime(month_str, "%Y-%m")
                    
                    # 检查是否在日期范围内
                    if start_dt and file_dt < start_dt:
                        continue
                    if end_dt and file_dt > end_dt:
                        continue
                        
                    # 惰性加载文件内容
                    with gzip.open(archive_file, 'rt', encoding='utf-8') as f:
                        trades = json.load(f)
                        
                    # 过滤具体日期范围内的交易
                    if start_dt or end_dt:
                        filtered_trades = []
                        for trade in trades:
                            trade_dt = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                            if start_dt and trade_dt < start_dt:
                                continue
                            if end_dt and trade_dt > end_dt:
                                continue
                            filtered_trades.append(trade)
                        trades = filtered_trades
                    
                    # 添加到内存
                    self.completed_trades.extend(trades)
                    loaded_count += len(trades)
                    
                except Exception as e:
                    logger.error(f"加载归档文件 {archive_file} 时出错: {e}")
                    continue
                    
            logger.info(f"已加载{loaded_count}笔归档交易")
            return loaded_count
            
        except Exception as e:
            logger.error(f"加载归档交易时出错: {e}")
            return 0
        
    def visualize_exploration_effects(self, filepath: Optional[str] = None) -> str:
        """
        可视化探索效果，包括探索率变化、成功率比较、不同探索类型效果等
        
        Args:
            filepath: 保存图表的文件路径，不提供则显示图表
            
        Returns:
            保存的文件路径或空字符串
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import numpy as np
            from matplotlib.ticker import PercentFormatter
        except ImportError:
            self.logger.error("缺少可视化所需的库，请安装matplotlib")
            return ""
            
        # 获取探索分析数据
        exploration_analysis = self._analyze_exploration_effectiveness()
        
        if exploration_analysis.get('exploration_count', 0) == 0:
            self.logger.warning("没有探索交易数据可供可视化")
            return ""
            
        # 创建图表
        plt.style.use('seaborn-darkgrid')
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # 1. 探索交易与标准交易数量对比
        ax1 = fig.add_subplot(gs[0, 0])
        exp_count = exploration_analysis.get('exploration_count', 0)
        std_count = exploration_analysis.get('standard_count', 0)
        bars = ax1.bar(['标准交易', '探索交易'], [std_count, exp_count], color=['#3498db', '#e74c3c'])
        ax1.set_title('交易数量对比', fontsize=14)
        ax1.set_ylabel('交易数量')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12)
                    
        # 2. 成功率对比
        ax2 = fig.add_subplot(gs[0, 1])
        exp_rate = exploration_analysis.get('exploration_success_rate', 0)
        std_rate = exploration_analysis.get('standard_success_rate', 0)
        bars = ax2.bar(['标准交易', '探索交易'], [std_rate, exp_rate], color=['#3498db', '#e74c3c'])
        ax2.set_title('成功率对比', fontsize=14)
        ax2.set_ylabel('成功率')
        ax2.set_ylim(0, max(exp_rate, std_rate) * 1.2)
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}',
                    ha='center', va='bottom', fontsize=12)
                    
        # 3. 平均利润对比
        ax3 = fig.add_subplot(gs[0, 2])
        exp_profit = exploration_analysis.get('exploration_avg_profit', 0)
        std_profit = exploration_analysis.get('standard_avg_profit', 0)
        bars = ax3.bar(['标准交易', '探索交易'], [std_profit, exp_profit], color=['#3498db', '#e74c3c'])
        ax3.set_title('平均利润对比', fontsize=14)
        ax3.set_ylabel('平均利润 (%)')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            color = 'green' if height > 0 else 'red'
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height > 0 else height - 0.5,
                    f'{height:.2f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=12, color=color)
                    
        # 4. 探索参数类型效果
        ax4 = fig.add_subplot(gs[1, :2])
        
        param_types = []
        success_rates = []
        avg_profits = []
        counts = []
        
        # 获取探索参数类型数据
        exploration_by_param = exploration_analysis.get('exploration_by_param', {})
        for param_type, data in exploration_by_param.items():
            if data.get('count', 0) > 0:
                param_types.append(param_type)
                success_rates.append(data.get('success_rate', 0))
                avg_profits.append(data.get('avg_profit', 0))
                counts.append(data.get('count', 0))
                
        # 绘制探索参数类型效果
        x = np.arange(len(param_types))
        width = 0.35
        
        if param_types:
            rects1 = ax4.bar(x - width/2, success_rates, width, label='成功率', color='#2ecc71')
            ax4.set_ylabel('成功率', color='#2ecc71')
            ax4.tick_params(axis='y', labelcolor='#2ecc71')
            ax4.set_ylim(0, max(success_rates) * 1.2)
            ax4.yaxis.set_major_formatter(PercentFormatter(1.0))
            
            # 添加数值标签
            for i, rect in enumerate(rects1):
                height = rect.get_height()
                ax4.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                        f'{height:.1%}',
                        ha='center', va='bottom', fontsize=10, color='#2ecc71')
                        
            # 创建第二个y轴
            ax4_2 = ax4.twinx()
            rects2 = ax4_2.bar(x + width/2, avg_profits, width, label='平均利润', color='#f39c12')
            ax4_2.set_ylabel('平均利润 (%)', color='#f39c12')
            ax4_2.tick_params(axis='y', labelcolor='#f39c12')
            
            # 添加数值标签
            for i, rect in enumerate(rects2):
                height = rect.get_height()
                ax4_2.text(rect.get_x() + rect.get_width()/2., height + 0.1 if height > 0 else height - 0.5,
                        f'{height:.1f}%',
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=10, color='#f39c12')
                        
            # 添加交易数量文本
            for i, count in enumerate(counts):
                ax4.text(x[i], -0.05, f'n={count}', ha='center', fontsize=9)
                
            # 设置x轴标签
            ax4.set_xticks(x)
            ax4.set_xticklabels([t.replace('_', '\n') for t in param_types])
            ax4.set_title('不同探索参数类型效果', fontsize=14)
            
            # 添加图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_2.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax4.text(0.5, 0.5, '没有足够的探索参数类型数据', 
                    ha='center', va='center', fontsize=14)
                    
        # 5. 探索SBS特征效果
        ax5 = fig.add_subplot(gs[1, 2])
        
        feature_types = []
        success_rates = []
        counts = []
        
        # 获取探索SBS特征类型数据
        exploration_by_feature = exploration_analysis.get('exploration_by_feature', {})
        for feature_type, data in exploration_by_feature.items():
            if data.get('count', 0) > 0:
                feature_types.append(feature_type)
                success_rates.append(data.get('success_rate', 0))
                counts.append(data.get('count', 0))
                
        # 绘制探索SBS特征效果
        if feature_types:
            # 创建饼图
            wedges, texts, autotexts = ax5.pie(
                counts, 
                labels=[f"{t}\n({c}笔)" for t, c in zip(feature_types, counts)],
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.tab10.colors[:len(feature_types)]
            )
            ax5.set_title('SBS特征分布', fontsize=14)
            ax5.axis('equal')  # 确保饼图为圆形
            
            # 修改文本大小
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(8)
        else:
            ax5.text(0.5, 0.5, '没有足够的SBS特征数据', 
                    ha='center', va='center', fontsize=14)
                    
        # 6. 探索率变化曲线图
        ax6 = fig.add_subplot(gs[2, 0])
        
        # 获取探索历史数据 (这需要添加一个记录探索率历史的功能)
        exploration_history = self.get_exploration_history()
        
        if exploration_history:
            # 提取数据
            timestamps = [entry.get('timestamp', i) for i, entry in enumerate(exploration_history)]
            rates = [entry.get('rate', 0) for entry in exploration_history]
            
            # 绘制曲线
            ax6.plot(timestamps, rates, marker='o', linestyle='-', color='#9b59b6')
            ax6.set_ylabel('探索率')
            ax6.set_title('探索率变化历史', fontsize=14)
            
            # 设置x轴标签
            if len(timestamps) > 10:
                ax6.set_xticks([timestamps[i] for i in range(0, len(timestamps), len(timestamps)//10)])
                
            ax6.grid(True)
        else:
            ax6.text(0.5, 0.5, '没有探索率历史数据', 
                    ha='center', va='center', fontsize=14)
                    
        # 7. 成功率随探索时间变化
        ax7 = fig.add_subplot(gs[2, 1:])
        
        # 这需要添加一个记录成功率历史的功能
        success_rate_history = self.get_success_rate_history()
        
        if success_rate_history:
            # 提取数据
            timestamps = [entry.get('timestamp', i) for i, entry in enumerate(success_rate_history)]
            exp_rates = [entry.get('exploration_rate', 0) for entry in success_rate_history]
            std_rates = [entry.get('standard_rate', 0) for entry in success_rate_history]
            
            # 绘制曲线
            ax7.plot(timestamps, exp_rates, marker='o', linestyle='-', color='#e74c3c', label='探索交易')
            ax7.plot(timestamps, std_rates, marker='s', linestyle='-', color='#3498db', label='标准交易')
            ax7.set_ylabel('成功率')
            ax7.set_title('成功率随时间变化', fontsize=14)
            ax7.legend()
            
            # 设置y轴为百分比
            ax7.yaxis.set_major_formatter(PercentFormatter(1.0))
            
            # 设置x轴标签
            if len(timestamps) > 10:
                ax7.set_xticks([timestamps[i] for i in range(0, len(timestamps), len(timestamps)//10)])
                
            ax7.grid(True)
        else:
            ax7.text(0.5, 0.5, '没有成功率历史数据', 
                    ha='center', va='center', fontsize=14)
        
        # 调整布局
        plt.tight_layout()
        
        # 添加总标题
        fig.suptitle('探索策略效果分析', fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.9)
        
        # 保存或显示图表
        if filepath:
            try:
                # 确保目录存在
                directory = os.path.dirname(filepath)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                    
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                self.logger.info(f"探索效果可视化已保存至: {filepath}")
                plt.close(fig)
                return filepath
            except Exception as e:
                self.logger.error(f"保存可视化图表出错: {e}")
                plt.close(fig)
                return ""
        else:
            plt.show()
            return ""
            
    def get_exploration_history(self) -> List[Dict]:
        """
        获取探索率变化历史
        
        Returns:
            探索率历史记录列表
        """
        # 如果没有预先存储探索率历史，此处返回空列表
        if not hasattr(self, '_exploration_rate_history'):
            return []
            
        return self._exploration_rate_history
        
    def get_success_rate_history(self) -> List[Dict]:
        """
        获取成功率历史
        
        Returns:
            成功率历史记录列表
        """
        # 如果没有预先存储成功率历史，此处返回空列表
        if not hasattr(self, '_success_rate_history'):
            return []
            
        return self._success_rate_history
        
    def optimize_exploration_parameters(self, 
                                      num_trials: int = 30, 
                                      optimization_metric: str = 'profit_factor',
                                      test_trades_count: int = 100,
                                      save_best_params: bool = True) -> Dict:
        """
        使用Optuna优化探索参数
        
        Args:
            num_trials: 优化试验次数
            optimization_metric: 优化目标，可选 'profit_factor', 'win_rate', 'expectancy', 'overall_effect'
            test_trades_count: 每次试验执行的交易数量
            save_best_params: 是否自动保存最佳参数
            
        Returns:
            优化结果字典
        """
        try:
            import optuna
        except ImportError:
            self.logger.error("未安装Optuna库，请使用 'pip install optuna' 安装")
            return {"error": "未安装Optuna"}
            
        # 备份当前参数
        original_params = {
            'exploration_enabled': self.exploration_enabled,
            'exploration_rate': self.exploration_rate,
            'min_exploration_rate': self.min_exploration_rate,
            'exploration_decay': self.exploration_decay,
            'exploration_boost_interval': self.exploration_boost_interval,
            'exploration_boost_factor': self.exploration_boost_factor,
            'exploration_success_threshold': self.exploration_success_threshold,
            'exploration_failure_threshold': self.exploration_failure_threshold,
            'exploration_success_rate_adjust': self.exploration_success_rate_adjust,
            'exploration_failure_rate_adjust': self.exploration_failure_rate_adjust
        }
        
        # 创建一个模拟环境用于测试参数
        # 这里需要为每次试验复制一个测试环境
        trade_templates = self._prepare_trade_templates()
        
        if not trade_templates:
            self.logger.error("无法准备交易模板，至少需要一些历史交易记录")
            return {"error": "无法准备交易模板"}
            
        self.logger.info(f"准备了{len(trade_templates)}个交易模板用于优化")
        
        # 定义优化目标函数
        def objective(trial):
            # 为当前试验设置参数
            params = {
                'exploration_enabled': True,
                'exploration_rate': trial.suggest_float('exploration_rate', 0.05, 0.4),
                'min_exploration_rate': trial.suggest_float('min_exploration_rate', 0.01, 0.1),
                'exploration_decay': trial.suggest_float('exploration_decay', 0.9, 0.999),
                'exploration_success_threshold': trial.suggest_float('exploration_success_threshold', 0.4, 0.8),
                'exploration_failure_threshold': trial.suggest_float('exploration_failure_threshold', 0.1, 0.4),
                'exploration_success_rate_adjust': trial.suggest_float('exploration_success_rate_adjust', 0.01, 0.2),
                'exploration_failure_rate_adjust': trial.suggest_float('exploration_failure_rate_adjust', 0.01, 0.2)
            }
            
            # 创建测试环境
            test_tracker = self._create_test_environment(params)
            
            # 执行测试交易
            results = self._simulate_trades(test_tracker, trade_templates, test_trades_count)
            
            # 返回优化指标
            if optimization_metric == 'profit_factor':
                # 盈亏比，越高越好
                return results.get('profit_factor', 0.0)
            elif optimization_metric == 'win_rate':
                # 胜率，越高越好
                return results.get('win_rate', 0.0)
            elif optimization_metric == 'expectancy':
                # 期望值，越高越好
                return results.get('expectancy', 0.0)
            elif optimization_metric == 'overall_effect':
                # 整体效果，综合考虑探索和标准交易的效果差异
                exploration_analysis = test_tracker._analyze_exploration_effectiveness()
                return exploration_analysis.get('overall_effect', 0.0)
            else:
                # 默认返回盈亏比
                return results.get('profit_factor', 0.0)
                
        # 创建研究
        study_name = f"exploration_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # 开始优化
        self.logger.info(f"开始参数优化，目标：{optimization_metric}，试验次数：{num_trials}")
        
        try:
            study.optimize(objective, n_trials=num_trials)
        except KeyboardInterrupt:
            self.logger.warning("参数优化被用户中断")
        except Exception as e:
            self.logger.error(f"参数优化过程出错: {e}")
            return {"error": str(e)}
            
        # 获取最佳参数
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"优化完成，最佳{optimization_metric}值: {best_value}")
        self.logger.info(f"最佳参数: {best_params}")
        
        # 如果请求保存最佳参数，则更新系统参数
        if save_best_params:
            # 创建完整的参数字典
            complete_params = {
                'exploration_enabled': True,
                'exploration_rate': best_params.get('exploration_rate'),
                'min_exploration_rate': best_params.get('min_exploration_rate'),
                'exploration_decay': best_params.get('exploration_decay'),
                'exploration_success_threshold': best_params.get('exploration_success_threshold'),
                'exploration_failure_threshold': best_params.get('exploration_failure_threshold'),
                'exploration_success_rate_adjust': best_params.get('exploration_success_rate_adjust'),
                'exploration_failure_rate_adjust': best_params.get('exploration_failure_rate_adjust')
            }
            
            # 更新配置
            self.update_config(complete_params)
            self.logger.info("已自动应用最佳参数")
            
        # 准备返回结果
        result = {
            "best_params": best_params,
            "best_value": best_value,
            "optimization_metric": optimization_metric,
            "num_trials": num_trials,
            "original_params": original_params,
            "study_name": study_name,
            "trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params
                }
                for t in study.trials
            ]
        }
        
        # 保存完整结果到文件
        try:
            result_path = self.storage_dir / f"optuna_results_{study_name}.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"优化结果已保存至: {result_path}")
            result["result_file"] = str(result_path)
        except Exception as e:
            self.logger.error(f"保存优化结果出错: {e}")
            
        return result
        
    def _prepare_trade_templates(self) -> List[Dict]:
        """
        准备用于参数优化的交易模板
        
        Returns:
            交易模板列表
        """
        if not self.completed_trades:
            return []
            
        # 找出所有包含SBS数据的交易
        sbs_trades = [t for t in self.completed_trades if 'metadata' in t and 'sbs_data' in t.get('metadata', {})]
        
        if not sbs_trades:
            # 如果没有SBS交易，使用所有已完成交易
            all_trades = self.completed_trades
        else:
            # 优先使用SBS交易
            all_trades = sbs_trades
            
        # 创建模板，保留必要的信息但去除结果相关字段
        templates = []
        
        for trade in all_trades:
            template = {
                'symbol': trade.get('symbol'),
                'direction': trade.get('direction'),
                'entry_price': trade.get('entry_price'),
                'stop_loss': trade.get('stop_loss'),
                'take_profit': trade.get('take_profit'),
                'timeframe': trade.get('timeframe'),
                'risk_percentage': trade.get('risk_percentage'),
            }
            
            # 保留有用的元数据，如SBS数据
            if 'metadata' in trade and 'sbs_data' in trade['metadata']:
                template['metadata'] = {
                    'sbs_data': trade['metadata']['sbs_data']
                }
                
            # 保留交易质量信息
            if 'trade_quality' in trade:
                template['trade_quality'] = trade['trade_quality']
                
            templates.append(template)
            
        return templates
        
    def _create_test_environment(self, params: Dict) -> 'TradeResultTracker':
        """
        创建测试环境用于参数优化
        
        Args:
            params: 参数字典
            
        Returns:
            测试用TradeResultTracker实例
        """
        # 创建新的配置，包含原配置和新参数
        test_config = self.config.copy()
        test_config.update(params)
        
        # 创建干净的测试环境
        test_tracker = TradeResultTracker(test_config)
        
        return test_tracker
        
    def _simulate_trades(self, 
                       test_tracker: 'TradeResultTracker', 
                       templates: List[Dict], 
                       trade_count: int) -> Dict:
        """
        在测试环境中模拟交易
        
        Args:
            test_tracker: 测试用TradeResultTracker实例
            templates: 交易模板列表
            trade_count: 要模拟的交易数量
            
        Returns:
            模拟结果统计
        """
        if not templates:
            return {}
            
        # 随机选择模板执行交易
        for i in range(trade_count):
            template = random.choice(templates)
            
            # 添加交易
            trade_id = f"sim_{i+1}"
            
            # 稍微变化一些参数，增加多样性
            variation = 0.97 + random.random() * 0.06  # 0.97-1.03
            
            # 创建交易
            test_tracker.add_trade(
                trade_id=trade_id,
                symbol=template.get('symbol', 'SIM'),
                direction=template.get('direction', '多'),
                entry_price=template.get('entry_price', 100) * variation,
                stop_loss=template.get('stop_loss', 95) * variation,
                take_profit=template.get('take_profit', 110) * variation,
                timeframe=template.get('timeframe', '1h'),
                risk_percentage=template.get('risk_percentage', 1.0),
                metadata=template.get('metadata'),
                sequence_points=template.get('metadata', {}).get('sbs_data', {}) if 'metadata' in template else None
            )
            
            # 模拟交易结果
            self._simulate_trade_result(test_tracker, trade_id, template, variation)
            
        # 获取统计结果
        return test_tracker.get_trade_stats()
        
    def _simulate_trade_result(self, 
                            test_tracker: 'TradeResultTracker', 
                            trade_id: str, 
                            template: Dict,
                            variation: float = 1.0) -> None:
        """
        模拟单笔交易结果
        
        Args:
            test_tracker: 测试用TradeResultTracker实例
            trade_id: 交易ID
            template: 交易模板
            variation: 结果变异系数
        """
        # 获取活跃交易
        trade = test_tracker.get_trade(trade_id)
        if not trade:
            return
            
        # 确定是否有探索
        is_exploration = 'metadata' in trade and 'exploration' in trade['metadata'] and trade['metadata']['exploration'].get('is_exploration', False)
        
        # 根据模板和探索情况模拟结果
        if is_exploration:
            # 探索交易的结果概率略有不同
            win_prob = 0.48  # 默认胜率略低
            
            # 如果是方向反转，成功率更低
            if ('direction' in trade['metadata']['exploration'].get('original_params', {}) and 
                trade['metadata']['exploration']['original_params']['direction'] != trade['direction']):
                win_prob = 0.4
            # 止损调整的成功率适中
            elif ('stop_loss' in trade['metadata']['exploration'].get('original_params', {})):
                win_prob = 0.5
            # 止盈调整的成功率较高
            elif ('take_profit' in trade['metadata']['exploration'].get('original_params', {})):
                win_prob = 0.55
        else:
            # 标准交易的基础胜率
            win_prob = 0.52
            
            # 如果是高质量交易，提高胜率
            if 'trade_quality' in template:
                quality_score = template['trade_quality'].get('total_score', 5.0)
                # 质量评分在0-10之间，将其映射到额外胜率0-0.2
                quality_bonus = (quality_score - 5.0) / 25.0  # 每高1分，胜率+0.04
                win_prob += quality_bonus
                
        # 随机决定是否盈利
        is_win = random.random() < win_prob
        
        # 设置盈亏
        direction = trade['direction']
        entry_price = trade['entry_price']
        
        if is_win:
            # 盈利交易
            if direction == '多':
                exit_price = trade['take_profit'] * random.uniform(0.8, 1.0)  # 达到止盈的80%-100%
            else:
                exit_price = trade['take_profit'] * random.uniform(1.0, 1.2)  # 达到止盈的100%-120%
            exit_reason = 'take_profit'
        else:
            # 亏损交易
            if direction == '多':
                exit_price = trade['stop_loss'] * random.uniform(1.0, 1.2)  # 达到止损的100%-120%
            else:
                exit_price = trade['stop_loss'] * random.uniform(0.8, 1.0)  # 达到止损的80%-100%
            exit_reason = 'stop_loss'
            
        # 关闭交易
        test_tracker.close_trade(
            trade_id=trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            exit_time=datetime.now().isoformat()
        )

    def get_exploration_metrics(self) -> Dict[str, Any]:
        """
        获取探索指标
        
        返回:
            探索指标字典
        """
        if not hasattr(self, 'exploration_manager'):
            return {
                'enabled': False,
                'message': '探索管理器未初始化'
            }
        
        return self.exploration_manager.get_metrics()

    def analyze_exploration_effectiveness(self) -> Dict[str, Any]:
        """
        分析探索策略的有效性
        
        返回:
            探索效果分析结果
        """
        if not hasattr(self, 'exploration_manager'):
            return {
                'enabled': False,
                'message': '探索管理器未初始化'
            }
        
        # 基础探索分析
        exploration_analysis = self.exploration_manager.analyze_exploration_effect()
        
        # 进一步分析交易结果
        exploration_trades = []
        standard_trades = []
        
        for trade in self.completed_trades:
            if trade.get('is_exploration', False):
                exploration_trades.append(trade)
            else:
                standard_trades.append(trade)
        
        # 计算统计数据
        exploration_analysis.update({
            'comparison': self._compare_exploration_with_standard(exploration_trades, standard_trades),
            'trade_counts': {
                'exploration': len(exploration_trades),
                'standard': len(standard_trades),
                'total': len(self.completed_trades)
            }
        })
        
        return exploration_analysis

    def _compare_exploration_with_standard(self, exploration_trades: List[Dict], standard_trades: List[Dict]) -> Dict[str, Any]:
        """
        比较探索交易和标准交易的效果
        
        参数:
            exploration_trades: 探索交易列表
            standard_trades: 标准交易列表
            
        返回:
            比较结果字典
        """
        # 如果任一列表为空，返回空结果
        if not exploration_trades or not standard_trades:
            return {
                'message': '没有足够的交易数据进行比较',
                'exploration_count': len(exploration_trades),
                'standard_count': len(standard_trades)
            }
        
        # 计算胜率
        exploration_wins = sum(1 for t in exploration_trades if t.get('profit_percentage', 0) > 0)
        standard_wins = sum(1 for t in standard_trades if t.get('profit_percentage', 0) > 0)
        
        exploration_win_rate = exploration_wins / len(exploration_trades) if exploration_trades else 0
        standard_win_rate = standard_wins / len(standard_trades) if standard_trades else 0
        
        # 计算平均利润
        exploration_profits = [t.get('profit_percentage', 0) for t in exploration_trades]
        standard_profits = [t.get('profit_percentage', 0) for t in standard_trades]
        
        exploration_avg_profit = sum(exploration_profits) / len(exploration_profits) if exploration_profits else 0
        standard_avg_profit = sum(standard_profits) / len(standard_profits) if standard_profits else 0
        
        # 计算夏普比率
        exploration_sharpe = self._calculate_sharpe_ratio_for_trades(exploration_trades)
        standard_sharpe = self._calculate_sharpe_ratio_for_trades(standard_trades)
        
        # 按交易类型分类
        exploration_by_type = {}
        for trade in exploration_trades:
            if 'metadata' in trade and 'exploration_type' in trade['metadata']:
                e_type = trade['metadata']['exploration_type']
                if e_type not in exploration_by_type:
                    exploration_by_type[e_type] = {
                        'trades': [],
                        'wins': 0,
                        'total_profit': 0,
                        'count': 0
                    }
                
                record = exploration_by_type[e_type]
                record['trades'].append(trade)
                record['count'] += 1
                
                profit = trade.get('profit_percentage', 0)
                record['total_profit'] += profit
                if profit > 0:
                    record['wins'] += 1
        
        # 计算每种探索类型的统计数据
        exploration_type_stats = {}
        for e_type, record in exploration_by_type.items():
            if record['count'] > 0:
                exploration_type_stats[e_type] = {
                    'count': record['count'],
                    'win_rate': record['wins'] / record['count'] if record['count'] > 0 else 0,
                    'avg_profit': record['total_profit'] / record['count'] if record['count'] > 0 else 0
                }
        
        return {
            'win_rate': {
                'exploration': exploration_win_rate,
                'standard': standard_win_rate,
                'difference': exploration_win_rate - standard_win_rate
            },
            'avg_profit': {
                'exploration': exploration_avg_profit,
                'standard': standard_avg_profit,
                'difference': exploration_avg_profit - standard_avg_profit
            },
            'sharpe_ratio': {
                'exploration': exploration_sharpe,
                'standard': standard_sharpe,
                'difference': exploration_sharpe - standard_sharpe
            },
            'exploration_by_type': exploration_type_stats,
            'exploration_benefit': {
                'win_rate_improvement': (exploration_win_rate / standard_win_rate - 1) * 100 if standard_win_rate > 0 else 0,
                'profit_improvement': (exploration_avg_profit / standard_avg_profit - 1) * 100 if standard_avg_profit > 0 else 0,
                'overall_benefit': (exploration_win_rate * exploration_avg_profit) / (standard_win_rate * standard_avg_profit) if (standard_win_rate * standard_avg_profit) > 0 else 0
            }
        }
 
    def _calculate_sharpe_ratio_for_trades(self, trades: List[Dict], risk_free_rate: float = 0.0) -> float:
        """
        计算交易列表的夏普比率
        
        参数:
            trades: 交易列表
            risk_free_rate: 无风险利率
            
        返回:
            夏普比率
        """
        if not trades:
            return 0.0
        
        profits = [t.get('profit_percentage', 0) for t in trades]
        mean_return = sum(profits) / len(profits)
        std_dev = np.std(profits) if len(profits) > 1 else 1e-6
        
        # 避免除以零
        if std_dev == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / std_dev

    def visualize_exploration_effectiveness(self, 
                                         output_path: Optional[str] = None,
                                         title: str = "探索策略效果分析",
                                         show_plot: bool = True) -> None:
        """
        可视化探索策略的效果
        
        参数:
            output_path: 输出文件路径，如果提供则保存图表
            title: 图表标题
            show_plot: 是否显示图表
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 准备数据
        analysis = self.analyze_exploration_effectiveness()
        
        # 如果没有探索交易，则显示提示信息
        if not analysis or analysis.get('enabled') is False or 'comparison' not in analysis:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, '没有足够的探索数据进行分析', 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=16)
            plt.title(title)
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=100)
            if show_plot:
                plt.show()
            else:
                plt.close()
            return
        
        # 创建图表
        fig = plt.figure(figsize=(16, 12))
        plt.suptitle(title, fontsize=16)
        
        # 1. 胜率和平均利润对比
        ax1 = fig.add_subplot(2, 2, 1)
        comparison = analysis['comparison']
        
        labels = ['探索交易', '常规交易']
        win_rates = [comparison['win_rate']['exploration'], comparison['win_rate']['standard']]
        
        ax1.bar(labels, win_rates, color=['#5DA5DA', '#F15854'])
        ax1.set_ylim(0, max(1, max(win_rates) * 1.2))
        ax1.set_title('胜率对比')
        ax1.set_ylabel('胜率')
        
        # 在柱状图上标注胜率值
        for i, v in enumerate(win_rates):
            ax1.text(i, v + 0.02, f'{v:.2%}', ha='center')
        
        # 2. 平均利润对比
        ax2 = fig.add_subplot(2, 2, 2)
        avg_profits = [comparison['avg_profit']['exploration'], comparison['avg_profit']['standard']]
        
        ax2.bar(labels, avg_profits, color=['#5DA5DA', '#F15854'])
        ax2.set_title('平均利润对比')
        ax2.set_ylabel('平均利润 (%)')
        
        # 在柱状图上标注平均利润值
        for i, v in enumerate(avg_profits):
            ax2.text(i, v + 0.1, f'{v:.2f}%', ha='center')
        
        # 3. 探索率和成功率变化趋势
        ax3 = fig.add_subplot(2, 2, 3)
        
        # 检查是否有探索成功率历史记录
        if hasattr(self.exploration_manager.state, 'success_rate_history') and self.exploration_manager.state.success_rate_history:
            history = self.exploration_manager.state.success_rate_history
            x = list(range(len(history)))
            recent_rates = [entry['recent_rate'] for entry in history]
            overall_rates = [entry['overall_rate'] for entry in history]
            
            ax3.plot(x, recent_rates, 'b-', label='近期成功率')
            ax3.plot(x, overall_rates, 'g--', label='总体成功率')
            ax3.set_title('探索成功率趋势')
            ax3.set_xlabel('交易序号')
            ax3.set_ylabel('成功率')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, '没有历史记录数据', ha='center', va='center', fontsize=12)
            ax3.set_title('探索成功率趋势')
        
        # 4. 探索类型分析
        ax4 = fig.add_subplot(2, 2, 4)
        
        # 检查是否有不同类型的探索
        if 'exploration_by_type' in comparison and comparison['exploration_by_type']:
            types = list(comparison['exploration_by_type'].keys())
            type_win_rates = [comparison['exploration_by_type'][t]['win_rate'] for t in types]
            type_avg_profits = [comparison['exploration_by_type'][t]['avg_profit'] for t in types]
            
            x = np.arange(len(types))
            width = 0.35
            
            ax4.bar(x - width/2, type_win_rates, width, label='胜率')
            ax4.bar(x + width/2, [p/10 for p in type_avg_profits], width, label='平均利润/10%')
            
            ax4.set_xticks(x)
            ax4.set_xticklabels(types)
            ax4.set_title('不同探索类型的效果')
            ax4.set_ylabel('比率')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, '没有探索类型数据', ha='center', va='center', fontsize=12)
            ax4.set_title('不同探索类型的效果')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局，留出顶部标题空间
        
        if output_path:
            plt.savefig(output_path, dpi=100)
        if show_plot:
            plt.show()
        else:
            plt.close()

    def optimize_exploration_parameters(self, 
                                     num_trials: int = 30, 
                                     optimization_metric: str = 'profit_factor',
                                     test_trades_count: int = 100,
                                     save_best_params: bool = True) -> Dict[str, Any]:
        """
        优化探索参数
        
        参数:
            num_trials: 优化试验次数
            optimization_metric: 优化目标指标，可选 'profit_factor', 'win_rate', 'sharpe_ratio', 'combined_score'
            test_trades_count: 每次试验的测试交易数量
            save_best_params: 是否保存最佳参数
            
        返回:
            优化结果字典
        """
        # 检查是否安装了optuna
        try:
            import optuna
        except ImportError:
            self.logger.error("优化探索参数需要安装optuna库，请使用 'pip install optuna' 安装")
            return {
                'success': False,
                'message': "未安装optuna库"
            }
        
        self.logger.info(f"开始优化探索参数，将进行{num_trials}次试验，每次使用{test_trades_count}笔交易")
        
        # 准备交易模板
        trade_templates = self._prepare_trade_templates()
        if not trade_templates:
            return {
                'success': False,
                'message': "没有足够的交易数据创建模板"
            }
        
        # 定义目标函数
        def objective(trial):
            # 定义优化参数范围
            params = {
                'exploration_rate': trial.suggest_float('exploration_rate', 0.05, 0.5),
                'min_exploration_rate': trial.suggest_float('min_exploration_rate', 0.01, 0.1),
                'exploration_decay': trial.suggest_float('exploration_decay', 0.9, 0.99),
                'exploration_reward_threshold': trial.suggest_float('exploration_reward_threshold', 0.5, 2.0),
                'success_threshold': trial.suggest_float('success_threshold', 0.4, 0.8),
                'failure_threshold': trial.suggest_float('failure_threshold', 0.1, 0.4),
                'boost_interval': trial.suggest_int('boost_interval', 50, 200),
                'boost_factor': trial.suggest_float('boost_factor', 1.01, 1.2)
            }
            
            # 创建测试环境
            test_tracker = self._create_test_environment(params)
            
            # 执行交易模拟
            results = self._simulate_trades(test_tracker, trade_templates, test_trades_count)
            
            # 计算优化指标
            if optimization_metric == 'profit_factor':
                return results.get('profit_factor', 0)
            elif optimization_metric == 'win_rate':
                return results.get('win_rate', 0)
            elif optimization_metric == 'sharpe_ratio':
                return results.get('sharpe_ratio', 0)
            elif optimization_metric == 'combined_score':
                # 综合指标：利用多个指标的加权和
                profit_factor = results.get('profit_factor', 0)
                win_rate = results.get('win_rate', 0)
                sharpe_ratio = results.get('sharpe_ratio', 0)
                avg_profit = results.get('avg_profit', 0) / 5  # 缩放平均利润
                
                return (profit_factor * 0.4) + (win_rate * 0.3) + (sharpe_ratio * 0.2) + (avg_profit * 0.1)
            else:
                # 默认使用利润因子
                return results.get('profit_factor', 0)
        
        # 创建优化研究
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=num_trials)
        
        # 获取最佳参数
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"优化完成，最佳参数: {best_params}, 最佳值: {best_value:.4f}")
        
        # 如果需要，应用最佳参数
        if save_best_params:
            # 更新探索管理器配置
            self.exploration_manager.exploration_config.update(**best_params)
            self.logger.info("已应用优化后的最佳参数到探索管理器")
        
        # 格式化优化历史
        optimization_history = []
        for i, trial in enumerate(study.trials):
            optimization_history.append({
                'trial': i,
                'params': trial.params,
                'value': trial.value
            })
        
        return {
            'success': True,
            'best_params': best_params,
            'best_value': best_value,
            'optimization_metric': optimization_metric,
            'optimization_history': optimization_history
        }

    def _prepare_trade_templates(self) -> List[Dict]:
        """
        准备交易模板，用于模拟测试
        
        返回:
            交易模板列表
        """
        templates = []
        
        # 至少需要10个已完成的交易
        if len(self.completed_trades) < 10:
            self.logger.warning("没有足够的已完成交易用于创建模板")
            return templates
        
        # 提取常见交易属性和结果分布
        symbols = set()
        directions = {'多': 0, '空': 0}
        timeframes = set()
        profit_values = []
        durations = []
        risk_reward_ratios = []
        
        for trade in self.completed_trades:
            symbols.add(trade.get('symbol', 'UNKNOWN'))
            directions[trade.get('direction', '多')] = directions.get(trade.get('direction', '多'), 0) + 1
            if 'timeframe' in trade:
                timeframes.add(trade.get('timeframe'))
            
            profit_values.append(trade.get('profit_percentage', 0))
            
            if 'duration' in trade:
                durations.append(trade.get('duration', 0))
            
            # 计算风险回报比
            entry_price = trade.get('entry_price', 0)
            stop_loss = trade.get('stop_loss', 0)
            take_profit = trade.get('take_profit', 0)
            direction = trade.get('direction', '多')
            
            if entry_price > 0 and stop_loss > 0 and take_profit > 0:
                if direction == '多':
                    risk = entry_price - stop_loss if stop_loss < entry_price else 0
                    reward = take_profit - entry_price if take_profit > entry_price else 0
                else:  # 空
                    risk = stop_loss - entry_price if stop_loss > entry_price else 0
                    reward = entry_price - take_profit if take_profit < entry_price else 0
                
                if risk > 0:
                    risk_reward_ratios.append(reward / risk)
        
        # 确保有足够的数据
        if not symbols or not timeframes or not profit_values:
            self.logger.warning("交易数据不完整，无法创建模板")
            return templates
        
        # 计算分布统计
        symbols = list(symbols)
        timeframes = list(timeframes)
        preferred_direction = '多' if directions.get('多', 0) >= directions.get('空', 0) else '空'
        
        profit_mean = sum(profit_values) / len(profit_values) if profit_values else 0
        profit_std = np.std(profit_values) if len(profit_values) > 1 else 1
        
        duration_mean = sum(durations) / len(durations) if durations else 300
        duration_std = np.std(durations) if len(durations) > 1 else 60
        
        rr_mean = sum(risk_reward_ratios) / len(risk_reward_ratios) if risk_reward_ratios else 2
        rr_std = np.std(risk_reward_ratios) if len(risk_reward_ratios) > 1 else 0.5
        
        # 创建交易模板
        # 为每个品种和时间框架组合创建模板
        for symbol in symbols:
            for timeframe in timeframes:
                # 为多空方向各创建一个模板
                for direction in ['多', '空']:
                    # 基础入场价
                    entry_price = 100 * (1 + random.uniform(-0.1, 0.1))
                    
                    # 根据方向和平均风险回报比设置止损止盈
                    rr_ratio = max(1, random.normalvariate(rr_mean, rr_std))
                    
                    if direction == '多':
                        stop_loss = entry_price * (1 - random.uniform(0.01, 0.05))
                        take_profit = entry_price * (1 + random.uniform(0.01, 0.05) * rr_ratio)
                    else:
                        stop_loss = entry_price * (1 + random.uniform(0.01, 0.05))
                        take_profit = entry_price * (1 - random.uniform(0.01, 0.05) * rr_ratio)
                    
                    # 创建模板
                    template = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'direction': direction,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_percentage': random.uniform(0.5, 2.0),
                        'expected_profit_mean': profit_mean,
                        'expected_profit_std': profit_std,
                        'expected_duration_mean': duration_mean,
                        'expected_duration_std': duration_std,
                        'probability_multiplier': 1.2 if direction == preferred_direction else 0.8
                    }
                    
                    templates.append(template)
        
        self.logger.info(f"已创建{len(templates)}个交易模板用于测试")
        return templates

    def _create_test_environment(self, params: Dict) -> 'TradeResultTracker':
        """
        创建测试环境
        
        参数:
            params: 配置参数
            
        返回:
            交易跟踪器实例
        """
        # 构建测试配置
        test_config = {
            'exploration_enabled': True,
            'memory_management_enabled': False
        }
        
        # 添加探索参数
        for key, value in params.items():
            test_config[f'exploration_{key}'] = value
        
        # 创建新实例
        test_tracker = TradeResultTracker(test_config)
        
        return test_tracker

    def _simulate_trades(self, 
                       test_tracker: 'TradeResultTracker', 
                       templates: List[Dict], 
                       trade_count: int) -> Dict:
        """
        模拟执行交易
        
        参数:
            test_tracker: 测试用的交易跟踪器
            templates: 交易模板列表
            trade_count: 要模拟的交易数量
            
        返回:
            模拟结果
        """
        # 确保至少有一个模板
        if not templates:
            return {'error': 'No templates available'}
        
        # 模拟交易
        for i in range(trade_count):
            # 随机选择一个模板
            template = random.choice(templates)
            
            # 创建唯一ID
            trade_id = f"test_{i+1:04d}"
            
            # 决定是否是探索交易
            market_state = {
                'price': template['entry_price'],
                'volatility': random.uniform(0.01, 0.05),
                'trend': random.uniform(-1, 1),
                'volume': random.uniform(100, 1000)
            }
            
            is_exploration = test_tracker.should_explore(market_state)
            
            # 添加交易
            trade = test_tracker.add_trade(
                trade_id=trade_id,
                symbol=template['symbol'],
                direction=template['direction'],
                entry_price=template['entry_price'],
                stop_loss=template['stop_loss'],
                take_profit=template['take_profit'],
                timeframe=template['timeframe'],
                risk_percentage=template['risk_percentage'],
                is_exploration=is_exploration,
                market_data=market_state
            )
            
            # 模拟交易结果
            self._simulate_trade_result(test_tracker, trade_id, template, 
                                       1.0 if is_exploration else template['probability_multiplier'])
        
        # 获取统计数据
        stats = test_tracker.get_trade_stats()
        
        return stats

    def _simulate_trade_result(self, 
                             test_tracker: 'TradeResultTracker', 
                             trade_id: str, 
                             template: Dict,
                             variation: float = 1.0) -> None:
        """
        模拟交易结果
        
        参数:
            test_tracker: 测试用的交易跟踪器
            trade_id: 交易ID
            template: 交易模板
            variation: 变异系数
        """
        if trade_id not in test_tracker.active_trades:
            return
        
        trade = test_tracker.active_trades[trade_id]
        direction = trade['direction']
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        
        # 根据模板生成结果
        expected_profit_mean = template['expected_profit_mean'] * variation
        expected_profit_std = template['expected_profit_std'] * variation
        
        # 生成随机利润
        profit_percentage = random.normalvariate(expected_profit_mean, expected_profit_std)
        
        # 计算出场价格
        if direction == '多':
            if profit_percentage > 0:
                # 盈利交易，价格向上移动
                exit_price = entry_price * (1 + profit_percentage / 100)
                exit_reason = 'take_profit'
            else:
                # 亏损交易，价格向下移动
                exit_price = entry_price * (1 + profit_percentage / 100)
                exit_reason = 'stop_loss'
        else:  # 空
            if profit_percentage > 0:
                # 盈利交易，价格向下移动
                exit_price = entry_price * (1 - profit_percentage / 100)
                exit_reason = 'take_profit'
            else:
                # 亏损交易，价格向上移动
                exit_price = entry_price * (1 - profit_percentage / 100)
                exit_reason = 'stop_loss'
        
        # 模拟持续时间
        duration = max(1, int(random.normalvariate(
            template['expected_duration_mean'], 
            template['expected_duration_std']
        )))
        
        # 出场时间
        entry_time = datetime.now() - timedelta(minutes=duration)
        exit_time = datetime.now()
        
        # 更新交易
        trade['entry_time'] = entry_time.isoformat()
        
        # 关闭交易
        test_tracker.close_trade(
            trade_id=trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            exit_time=exit_time.isoformat()
        )

    def process_trades_in_batches(self, trades: List[Dict], batch_size: int = 100, processing_func=None, parallel: bool = False, num_workers: int = 4) -> List[Any]:
        """
        分批处理大量交易数据，支持并行处理
        
        参数:
            trades: 要处理的交易列表
            batch_size: 每批处理的交易数量
            processing_func: 处理函数
            parallel: 是否启用并行处理
            num_workers: 并行处理的工作进程数
            
        返回:
            处理结果列表
        """
        if not trades:
            return []
            
        total_trades = len(trades)
        num_batches = (total_trades + batch_size - 1) // batch_size
        
        # 创建批次
        batches = [
            trades[i * batch_size : min((i + 1) * batch_size, total_trades)]
            for i in range(num_batches)
        ]
        
        if parallel and processing_func:
            # 并行处理
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 使用tqdm显示进度
                futures = [executor.submit(processing_func, batch) for batch in batches]
                
                results = []
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="处理交易批次"
                ):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"批次处理出错: {e}")
        else:
            # 串行处理
            results = []
            for i, batch in enumerate(tqdm(batches, desc="处理交易批次")):
                try:
                    if processing_func:
                        result = processing_func(batch)
                        if result is not None:
                            results.append(result)
                    else:
                        results.append(batch)
                except Exception as e:
                    logger.error(f"批次 {i+1}/{num_batches} 处理出错: {e}")
                    
                # 定期执行垃圾回收
                if (i + 1) % 10 == 0:
                    gc.collect()
        
        # 合并结果
        if results:
            if all(isinstance(r, list) for r in results):
                # 合并列表类型结果
                return [item for sublist in results for item in sublist]
            elif all(isinstance(r, dict) for r in results):
                # 合并字典类型结果
                merged = {}
                for r in results:
                    merged.update(r)
                return merged
                
        return results