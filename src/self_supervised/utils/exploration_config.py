#!/usr/bin/env python
"""
探索配置模块
提供用于管理探索机制参数的数据类
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path
import json

@dataclass
class ExplorationConfig:
    """探索机制配置数据类"""
    # 基本参数
    enabled: bool = False
    exploration_rate: float = 0.2
    min_exploration_rate: float = 0.05
    exploration_decay: float = 0.995
    reward_threshold: float = 1.0
    
    # 高级参数
    boost_interval: int = 100
    boost_factor: float = 1.05
    success_threshold: float = 0.6
    failure_threshold: float = 0.3
    success_rate_adjust: float = 0.05
    failure_rate_adjust: float = 0.05
    
    # 记录参数
    history_size: int = 1000
    save_history: bool = True
    history_path: Optional[str] = None
    
    # 分析参数
    market_aware: bool = False
    volatility_scaling: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExplorationConfig':
        """从字典创建配置"""
        # 提取探索相关的配置
        exploration_dict = {}
        for k, v in config_dict.items():
            # 处理带前缀的参数
            if k.startswith('exploration_'):
                key = k.replace('exploration_', '')
                exploration_dict[key] = v
            # 处理直接的参数
            elif k in cls.__annotations__:
                exploration_dict[k] = v
        
        return cls(**exploration_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items()}
    
    def save(self, filepath: str) -> bool:
        """保存配置到文件"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"保存探索配置失败: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> Optional['ExplorationConfig']:
        """从文件加载配置"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except Exception as e:
            logging.error(f"加载探索配置失败: {e}")
            return None
    
    def update(self, **kwargs) -> None:
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass
class ExplorationState:
    """探索状态跟踪数据类"""
    # 计数器
    total_trades: int = 0
    total_explorations: int = 0
    successful_explorations: int = 0
    
    # 序列跟踪
    current_streak: int = 0
    best_streak: int = 0
    
    # 记录历史
    exploration_history: List[Dict[str, Any]] = field(default_factory=list)
    success_rate_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # 市场状态关联
    market_condition_records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_exploration(self, successful: bool, profit: float, metadata: Dict[str, Any] = None) -> None:
        """添加探索记录"""
        self.total_explorations += 1
        
        if successful:
            self.successful_explorations += 1
            self.current_streak += 1
            if self.current_streak > self.best_streak:
                self.best_streak = self.current_streak
        else:
            self.current_streak = 0
        
        # 添加到历史记录
        history_entry = {
            'successful': successful,
            'profit': profit,
            'total_trades': self.total_trades,
            'timestamp': metadata.get('timestamp') if metadata else None,
            'metadata': metadata
        }
        
        # 记录市场条件相关的探索结果
        if metadata and 'market_condition' in metadata:
            condition = metadata['market_condition']
            if condition not in self.market_condition_records:
                self.market_condition_records[condition] = {
                    'total': 0,
                    'successful': 0,
                    'total_profit': 0.0,
                    'explorations': []
                }
            
            record = self.market_condition_records[condition]
            record['total'] += 1
            record['total_profit'] += profit
            if successful:
                record['successful'] += 1
            record['explorations'].append(history_entry)
        
        self.exploration_history.append(history_entry)
        
        # 维护历史记录大小
        max_history = 1000  # 可配置
        if len(self.exploration_history) > max_history:
            self.exploration_history = self.exploration_history[-max_history:]
    
    def calculate_success_rate(self, window: int = 50) -> float:
        """计算探索成功率"""
        if self.total_explorations == 0:
            return 0.0
            
        # 全部历史的成功率
        overall_rate = self.successful_explorations / self.total_explorations
        
        # 如果有足够多的探索记录，计算近期窗口的成功率
        if len(self.exploration_history) >= window:
            recent = self.exploration_history[-window:]
            successful_count = sum(1 for entry in recent if entry['successful'])
            recent_rate = successful_count / window
            
            # 记录到成功率历史
            self.success_rate_history.append({
                'overall_rate': overall_rate,
                'recent_rate': recent_rate,
                'total_explorations': self.total_explorations
            })
            
            return recent_rate
        
        return overall_rate
    
    def get_market_condition_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """获取不同市场条件下的探索统计"""
        stats = {}
        
        for condition, record in self.market_condition_records.items():
            win_rate = record['successful'] / record['total'] if record['total'] > 0 else 0
            avg_profit = record['total_profit'] / record['total'] if record['total'] > 0 else 0
            
            stats[condition] = {
                'count': record['total'],
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'total_profit': record['total_profit']
            }
            
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_trades': self.total_trades,
            'total_explorations': self.total_explorations,
            'successful_explorations': self.successful_explorations,
            'current_streak': self.current_streak,
            'best_streak': self.best_streak,
            'success_rate': self.calculate_success_rate(),
            'market_condition_stats': self.get_market_condition_stats()
        } 