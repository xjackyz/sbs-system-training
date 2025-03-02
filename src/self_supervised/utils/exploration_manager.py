#!/usr/bin/env python
"""
探索管理器模块
负责处理探索决策和基于强化学习的优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import deque
import logging
import json
from datetime import datetime
from pathlib import Path

from .exploration_config import ExplorationConfig, ExplorationState

class ExplorationManager:
    """探索管理器，使用强化学习优化探索决策"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化探索管理器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.exploration_config = ExplorationConfig.from_dict(self.config)
        self.state = ExplorationState()
        
        # 初始化记录器
        self.logger = logging.getLogger('exploration_manager')
        
        # 初始化RL模型
        self._init_rl_model()
        
        # 经验回放缓冲区
        self.replay_buffer_size = self.config.get('replay_buffer_size', 10000)
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        
        # 探索追踪
        self.last_decision_was_exploration = False
        self.decision_market_state = None
        
    def _init_rl_model(self) -> None:
        """初始化强化学习模型"""
        # 只有当启用强化学习决策时才创建模型
        if self.config.get('use_rl_for_exploration', False):
            # 输入特征数量
            input_size = 12  # 市场特征数量
            
            # 简单的DQN模型
            self.dqn = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2)  # [常规行动价值, 探索行动价值]
            )
            
            # 优化器
            self.optimizer = torch.optim.Adam(
                self.dqn.parameters(), 
                lr=self.config.get('rl_learning_rate', 0.001)
            )
            
            # 目标网络（用于稳定学习）
            self.target_dqn = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
            
            # 复制参数到目标网络
            self.target_dqn.load_state_dict(self.dqn.state_dict())
            
            self.logger.info("已初始化强化学习探索决策模型")
            
    def should_explore(self, market_state: Optional[Dict[str, Any]] = None) -> bool:
        """
        决定是否应该进行探索
        
        参数:
            market_state: 当前市场状态，可选
            
        返回:
            是否应该探索
        """
        # 如果未启用探索，直接返回False
        if not self.exploration_config.enabled:
            return False
            
        # 保存当前市场状态，用于后续学习
        self.decision_market_state = market_state
        
        # 使用强化学习决策
        if self.config.get('use_rl_for_exploration', False) and market_state is not None:
            decision = self._rl_decision(market_state)
            self.last_decision_was_exploration = decision
            return decision
        
        # 使用ε-greedy策略
        if random.random() < self.exploration_config.exploration_rate:
            self.last_decision_was_exploration = True
            return True
            
        self.last_decision_was_exploration = False
        return False
        
    def _rl_decision(self, market_state: Dict[str, Any]) -> bool:
        """
        使用强化学习模型做决策
        
        参数:
            market_state: 市场状态
            
        返回:
            是否探索
        """
        # ε-greedy策略
        if random.random() < self.exploration_config.exploration_rate:
            return random.choice([True, False])
            
        # 使用模型预测
        with torch.no_grad():
            state_tensor = self._prepare_state_tensor(market_state)
            q_values = self.dqn(state_tensor)
            action = torch.argmax(q_values).item()
            
        return action == 1  # 1表示探索，0表示常规
    
    def update_from_result(self, 
                         was_exploration: bool, 
                         profit: float,
                         market_state: Optional[Dict[str, Any]] = None, 
                         next_market_state: Optional[Dict[str, Any]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        根据交易结果更新探索机制
        
        参数:
            was_exploration: 是否为探索交易
            profit: 获得的利润
            market_state: 决策时的市场状态
            next_market_state: 当前市场状态
            metadata: 附加元数据
        """
        # 更新交易计数
        self.state.total_trades += 1
        
        # 如果是探索交易，更新探索记录
        if was_exploration:
            # 判断探索是否成功
            successful = profit > 0
            
            # 添加探索记录
            self.state.add_exploration(successful, profit, metadata)
            
            # 如果启用了强化学习，存储经验
            if self.config.get('use_rl_for_exploration', False) and market_state and next_market_state:
                self._store_experience(market_state, was_exploration, profit, next_market_state)
                
                # 定期学习
                if len(self.replay_buffer) >= self.config.get('batch_size', 32):
                    self._learn()
                    
                    # 定期更新目标网络
                    if self.state.total_trades % self.config.get('target_update_freq', 100) == 0:
                        self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        # 更新探索率
        self._update_exploration_rate()
        
    def _store_experience(self, 
                        state: Dict[str, Any], 
                        action: bool, 
                        reward: float, 
                        next_state: Dict[str, Any]) -> None:
        """
        存储经验到回放缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作(是否探索)
            reward: 获得的奖励
            next_state: 下一个状态
        """
        # 转换为张量
        state_tensor = self._prepare_state_tensor(state).numpy()
        next_state_tensor = self._prepare_state_tensor(next_state).numpy()
        
        # 存储经验
        self.replay_buffer.append((
            state_tensor, 
            1 if action else 0,  # 转换为整数
            reward, 
            next_state_tensor
        ))
    
    def _learn(self) -> None:
        """从经验中学习"""
        # 如果缓冲区太小，跳过学习
        batch_size = self.config.get('batch_size', 32)
        if len(self.replay_buffer) < batch_size:
            return
            
        # 采样批次
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # 转换为张量
        states = torch.tensor(np.vstack(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.vstack(next_states), dtype=torch.float32)
        
        # 计算当前Q值
        current_q = self.dqn(states).gather(1, actions)
        
        # 计算下一个状态的最大Q值(使用目标网络)
        with torch.no_grad():
            max_next_q = self.target_dqn(next_states).max(1)[0].unsqueeze(1)
            
        # 计算目标Q值
        gamma = self.config.get('gamma', 0.99)  # 折扣因子
        target_q = rewards + gamma * max_next_q
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _prepare_state_tensor(self, market_state: Dict[str, Any]) -> torch.Tensor:
        """
        将市场状态转换为模型输入张量
        
        参数:
            market_state: 市场状态字典
            
        返回:
            状态张量
        """
        # 提取市场特征
        features = []
        
        # 核心特征 - 只保留K线基本数据和SMA
        features.extend([
            market_state.get('price', 0),
            market_state.get('volume', 0),
            market_state.get('volatility', 0),
            market_state.get('sma20', 0),  # SMA20
            market_state.get('sma200', 0), # SMA200
            market_state.get('price_to_sma20', 0),  # 价格相对于SMA20的位置
            market_state.get('price_to_sma200', 0),  # 价格相对于SMA200的位置
            market_state.get('sma20_to_sma200', 0),  # SMA20相对于SMA200的位置
        ])
        
        # 当前回合探索状态
        exploration_rate = self.exploration_config.exploration_rate
        success_rate = self.state.calculate_success_rate()
        features.extend([
            exploration_rate,
            success_rate,
            self.state.current_streak,
            self.state.total_explorations / max(1, self.state.total_trades),
        ])
        
        # 确保特征数量正确(填充到12个特征)
        while len(features) < 12:
            features.append(0.0)
            
        # 截断特征到12个
        features = features[:12]
        
        # 转换为张量
        return torch.tensor([features], dtype=torch.float32)
    
    def _update_exploration_rate(self) -> None:
        """更新探索率"""
        # 基本衰减
        current_rate = self.exploration_config.exploration_rate
        new_rate = max(
            self.exploration_config.min_exploration_rate,
            current_rate * self.exploration_config.exploration_decay
        )
        
        # 根据成功率动态调整
        if self.state.total_explorations > 20:
            success_rate = self.state.calculate_success_rate()
            
            # 如果成功率高，可以增加探索
            if success_rate > self.exploration_config.success_threshold:
                adjust_factor = 1.0 + self.exploration_config.success_rate_adjust
                new_rate = min(0.5, new_rate * adjust_factor)  # 上限0.5
                
            # 如果成功率低，减少探索
            elif success_rate < self.exploration_config.failure_threshold:
                adjust_factor = 1.0 - self.exploration_config.failure_rate_adjust
                new_rate = max(self.exploration_config.min_exploration_rate, new_rate * adjust_factor)
                
        # 周期性地提升探索率，避免陷入局部最优
        if (self.state.total_trades % self.exploration_config.boost_interval == 0 and 
            self.state.total_trades > 0):
            new_rate = min(0.5, new_rate * self.exploration_config.boost_factor)
            self.logger.info(f"周期性提升探索率至: {new_rate:.4f}")
        
        # 更新探索率
        self.exploration_config.exploration_rate = new_rate
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取探索指标
        
        返回:
            探索指标字典
        """
        metrics = self.state.to_dict()
        metrics.update({
            'exploration_rate': self.exploration_config.exploration_rate,
            'configured_min_rate': self.exploration_config.min_exploration_rate,
            'success_threshold': self.exploration_config.success_threshold,
            'failure_threshold': self.exploration_config.failure_threshold,
        })
        
        return metrics
        
    def analyze_exploration_effect(self) -> Dict[str, Any]:
        """
        分析探索效果
        
        返回:
            探索效果分析结果
        """
        if self.state.total_explorations == 0:
            return {
                'analysis': '没有足够的探索数据进行分析',
                'exploration_count': 0
            }
            
        # 基本指标
        analysis = {
            'exploration_count': self.state.total_explorations,
            'success_rate': self.state.successful_explorations / self.state.total_explorations,
            'current_exploration_rate': self.exploration_config.exploration_rate,
            'best_streak': self.state.best_streak,
        }
        
        # 市场条件分析
        if self.state.market_condition_records:
            analysis['market_condition_analysis'] = self.state.get_market_condition_stats()
            
        # 最近趋势分析
        if len(self.state.success_rate_history) > 0:
            recent_rates = [entry['recent_rate'] for entry in self.state.success_rate_history[-10:]]
            if recent_rates:
                analysis['recent_success_trend'] = {
                    'mean': sum(recent_rates) / len(recent_rates),
                    'increasing': recent_rates[-1] > recent_rates[0] if len(recent_rates) > 1 else False,
                    'values': recent_rates
                }
                
        return analysis
    
    def save_state(self, filepath: str) -> bool:
        """
        保存探索状态到文件
        
        参数:
            filepath: 文件路径
            
        返回:
            是否保存成功
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            state_dict = {
                'exploration_config': self.exploration_config.to_dict(),
                'state': self.state.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'total_trades': self.state.total_trades,
                'exploration_history': self.state.exploration_history[-100:] if self.state.exploration_history else []
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=4, ensure_ascii=False)
                
            return True
        except Exception as e:
            self.logger.error(f"保存探索状态失败: {e}")
            return False
            
    def load_state(self, filepath: str) -> bool:
        """
        从文件加载探索状态
        
        参数:
            filepath: 文件路径
            
        返回:
            是否加载成功
        """
        try:
            path = Path(filepath)
            if not path.exists():
                self.logger.warning(f"探索状态文件不存在: {filepath}")
                return False
                
            with open(path, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)
                
            # 加载配置
            if 'exploration_config' in state_dict:
                self.exploration_config = ExplorationConfig(**state_dict['exploration_config'])
                
            # 重建探索状态
            if 'state' in state_dict:
                state_data = state_dict['state']
                self.state.total_trades = state_data.get('total_trades', 0)
                self.state.total_explorations = state_data.get('total_explorations', 0)
                self.state.successful_explorations = state_data.get('successful_explorations', 0)
                self.state.current_streak = state_data.get('current_streak', 0)
                self.state.best_streak = state_data.get('best_streak', 0)
                
            # 加载历史记录
            if 'exploration_history' in state_dict:
                self.state.exploration_history = state_dict['exploration_history']
                
            self.logger.info(f"成功从{filepath}加载探索状态")
            return True
        except Exception as e:
            self.logger.error(f"加载探索状态失败: {e}")
            return False 