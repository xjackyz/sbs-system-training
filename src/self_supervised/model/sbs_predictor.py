"""
SBS预测器
基于K线图识别SBS序列和交易信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
import json
from pathlib import Path
import pandas as pd

from .base_model import BaseModel
from ..utils.logger import setup_logger

logger = setup_logger('sbs_predictor')

class SBSPredictor(BaseModel):
    """SBS序列预测器类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化SBS预测器
        
        Args:
            config: 配置字典
        """
        super().__init__()
        self.config = config or {}
        
        # 模型参数
        self.input_channels = config.get('input_channels', 7)  # OHLCV + SMA20 + SMA200
        self.hidden_size = config.get('hidden_size', 128)
        self.num_points = 5  # SBS序列的5个关键点
        self.sequence_length = config.get('sequence_length', 100)
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # 序列状态预测器
        self.sequence_status_predictor = nn.Sequential(
            nn.Linear(128 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 3种状态: 未形成, 形成中, 已完成
        )
        
        # LSTM层，捕捉时间序列依赖
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # 点位预测层
        self.point_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 256),  # 双向LSTM
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_points)
        )
        
        # 置信度预测层
        self.confidence_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_points),
            nn.Sigmoid()  # 输出0-1之间的置信度
        )
        
        # 交易方向预测层
        self.trade_direction_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # 3个方向: 多, 空, 无信号
        )
        
        self.logger = setup_logger('sbs_predictor')
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_channels, sequence_length]
            
        Returns:
            预测结果字典
        """
        batch_size = x.size(0)
        
        # 特征提取
        features = self.feature_extractor(x)  # [batch_size, 128, 32]
        
        # 序列状态预测
        flat_features = features.view(batch_size, -1)
        sequence_status_logits = self.sequence_status_predictor(flat_features)
        sequence_status_probs = F.softmax(sequence_status_logits, dim=1)
        
        # LSTM处理
        lstm_input = features.permute(0, 2, 1)  # [batch_size, 32, 128]
        lstm_output, _ = self.lstm(lstm_input)
        
        # 获取最后一个时间步的输出
        lstm_output = lstm_output[:, -1, :]  # [batch_size, hidden_size*2]
        
        # 点位预测
        point_logits = self.point_predictor(lstm_output)  # [batch_size, num_points]
        
        # 置信度预测
        confidence_scores = self.confidence_predictor(lstm_output)  # [batch_size, num_points]
        
        # 交易方向预测
        trade_direction_logits = self.trade_direction_predictor(lstm_output)
        trade_direction_probs = F.softmax(trade_direction_logits, dim=1)
        
        return {
            'point_logits': point_logits,
            'confidence_scores': confidence_scores,
            'sequence_status_logits': sequence_status_logits,
            'sequence_status_probs': sequence_status_probs,
            'trade_direction_logits': trade_direction_logits,
            'trade_direction_probs': trade_direction_probs
        }
        
    def predict(self, kline_data: Dict) -> Dict:
        """
        预测SBS序列点位
        
        Args:
            kline_data: K线数据
            
        Returns:
            预测结果
        """
        try:
            # 准备输入数据
            x = self._prepare_input(kline_data)
            
            # 设置为评估模式
            self.eval()
            
            # 无梯度计算
            with torch.no_grad():
                # 前向传播
                output = self.forward(x)
                
                # 获取点位预测
                point_logits = output['point_logits'].squeeze(0).cpu().numpy()
                
                # 获取置信度
                confidence_scores = output['confidence_scores'].squeeze(0).cpu().numpy()
                
                # 获取序列状态预测
                sequence_status_probs = output['sequence_status_probs'].squeeze(0).cpu().numpy()
                sequence_status_idx = np.argmax(sequence_status_probs)
                sequence_status_labels = ["未形成", "形成中", "已完成"]
                sequence_status = sequence_status_labels[sequence_status_idx]
                sequence_status_confidence = float(sequence_status_probs[sequence_status_idx])
                
                # 获取交易方向预测
                trade_direction_probs = output['trade_direction_probs'].squeeze(0).cpu().numpy()
                trade_direction_idx = np.argmax(trade_direction_probs)
                trade_direction_labels = ["多", "空", "无信号"]
                trade_direction = trade_direction_labels[trade_direction_idx]
                trade_direction_confidence = float(trade_direction_probs[trade_direction_idx])
                
                # 规范化点位预测
                normalized_points = self._normalize_points(point_logits, len(kline_data['open']))
                
                # 验证预测结果
                validated_points = self._validate_points(normalized_points, confidence_scores)
                
                # 构建结果
                is_active = sequence_status != "未形成"
                
                result = {
                    'model_version': self.config.get('model_version', 'v1.0'),
                    'sequence_status': {
                        'label': sequence_status,
                        'is_active': is_active,
                        'confidence': sequence_status_confidence
                    },
                    'points': {
                        'point1': validated_points[0],
                        'point2': validated_points[1],
                        'point3': validated_points[2],
                        'point4': validated_points[3],
                        'point5': validated_points[4]
                    },
                    'confidence_scores': {
                        'point1': float(confidence_scores[0]),
                        'point2': float(confidence_scores[1]),
                        'point3': float(confidence_scores[2]),
                        'point4': float(confidence_scores[3]),
                        'point5': float(confidence_scores[4])
                    },
                    'trade_setup': {
                        'direction': trade_direction if is_active else "无信号",
                        'confidence': trade_direction_confidence if is_active else 0.0,
                        'entry_price': self._calculate_entry_price(kline_data, validated_points) if is_active else None,
                        'stop_loss': self._calculate_stop_loss(kline_data, validated_points, trade_direction) if is_active else None,
                        'take_profit': self._calculate_take_profit(kline_data, validated_points, trade_direction) if is_active else None
                    }
                }
                
                return result
                
        except Exception as e:
            logger.error(f"预测过程中出错: {str(e)}")
            # 返回默认结果
            return {
                'sequence_status': {'label': "未形成", 'is_active': False, 'confidence': 0.0},
                'points': {}, 
                'confidence_scores': {},
                'trade_setup': {'direction': "无信号", 'confidence': 0.0}
            }
        
    def _prepare_input(self, kline_data: Dict) -> torch.Tensor:
        """
        准备模型输入数据
        
        Args:
            kline_data: K线数据
            
        Returns:
            模型输入张量
        """
        # 提取OHLCV数据
        open_prices = np.array(kline_data['open'], dtype=np.float32)
        high_prices = np.array(kline_data['high'], dtype=np.float32)
        low_prices = np.array(kline_data['low'], dtype=np.float32)
        close_prices = np.array(kline_data['close'], dtype=np.float32)
        volumes = np.array(kline_data['volume'], dtype=np.float32) if 'volume' in kline_data else np.zeros_like(open_prices)
        
        # 使用pandas的rolling计算SMA，提高性能
        close_series = pd.Series(close_prices)
        sma20 = close_series.rolling(window=20, min_periods=1).mean().fillna(method='bfill').values
        sma200 = close_series.rolling(window=200, min_periods=1).mean().fillna(method='bfill').values
        
        # 归一化数据
        open_norm = self._normalize_data(open_prices)
        high_norm = self._normalize_data(high_prices)
        low_norm = self._normalize_data(low_prices)
        close_norm = self._normalize_data(close_prices)
        volume_norm = self._normalize_data(volumes) if 'volume' in kline_data else np.zeros_like(open_norm)
        sma20_norm = self._normalize_data(sma20)
        sma200_norm = self._normalize_data(sma200)
        
        # 截取或补齐序列长度
        seq_length = self.sequence_length
        if len(open_norm) > seq_length:
            # 截取最后的seq_length个数据点
            open_norm = open_norm[-seq_length:]
            high_norm = high_norm[-seq_length:]
            low_norm = low_norm[-seq_length:]
            close_norm = close_norm[-seq_length:]
            volume_norm = volume_norm[-seq_length:]
            sma20_norm = sma20_norm[-seq_length:]
            sma200_norm = sma200_norm[-seq_length:]
        elif len(open_norm) < seq_length:
            # 前面补零
            padding_length = seq_length - len(open_norm)
            open_norm = np.pad(open_norm, (padding_length, 0), 'constant')
            high_norm = np.pad(high_norm, (padding_length, 0), 'constant')
            low_norm = np.pad(low_norm, (padding_length, 0), 'constant')
            close_norm = np.pad(close_norm, (padding_length, 0), 'constant')
            volume_norm = np.pad(volume_norm, (padding_length, 0), 'constant')
            sma20_norm = np.pad(sma20_norm, (padding_length, 0), 'constant')
            sma200_norm = np.pad(sma200_norm, (padding_length, 0), 'constant')
            
        # 组合数据 - 包含OHLCV和SMA
        data = np.stack([open_norm, high_norm, low_norm, close_norm, volume_norm, sma20_norm, sma200_norm], axis=0)
        
        # 转换为torch张量
        x = torch.from_numpy(data).float().unsqueeze(0)  # 添加batch维度 [1, input_channels, seq_length]
        
        # 移至GPU（如果可用）
        if torch.cuda.is_available():
            x = x.cuda()
            
        return x
        
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        归一化数据
        
        Args:
            data: 输入数据
            
        Returns:
            归一化后的数据
        """
        if np.max(data) == np.min(data) or len(data) == 0:
            return np.zeros_like(data)
            
        return (data - np.min(data)) / (np.max(data) - np.min(data))
        
    def _normalize_points(self, points: np.ndarray, kline_length: int) -> List[int]:
        """
        规范化点位预测
        
        Args:
            points: 原始点位预测
            kline_length: K线长度
            
        Returns:
            规范化后的点位索引
        """
        # 将预测转换为K线中的索引位置
        normalized = []
        for p in points:
            # 限制在合理范围内 [0, kline_length-1]
            point_idx = max(0, min(kline_length - 1, int(p * kline_length / self.sequence_length)))
            normalized.append(point_idx)
            
        return normalized
        
    def _validate_points(self, points: List[int], confidence_scores: np.ndarray) -> List[Optional[int]]:
        """
        验证预测点位
        
        Args:
            points: 点位列表
            confidence_scores: 置信度分数
            
        Returns:
            验证后的点位列表
        """
        validated_points = []
        min_confidence = self.config.get('min_confidence', 0.3)
        
        for i, (point, conf) in enumerate(zip(points, confidence_scores)):
            # 检查置信度
            if conf > min_confidence:
                validated_points.append(point)
            else:
                validated_points.append(None)
                
        return validated_points
        
    def _calculate_entry_price(self, kline_data: Dict, points: List[Optional[int]]) -> Optional[float]:
        """
        计算入场价格
        
        Args:
            kline_data: K线数据
            points: 点位列表
            
        Returns:
            入场价格
        """
        # 入场点通常在点位4之后
        point4 = points[3]
        if point4 is None or point4 >= len(kline_data['close']) - 1:
            return None
            
        # 用点位4之后的收盘价作为入场价
        return float(kline_data['close'][point4])
        
    def _calculate_stop_loss(self, 
                          kline_data: Dict, 
                          points: List[Optional[int]], 
                          direction: str) -> Optional[float]:
        """
        计算止损价格
        
        Args:
            kline_data: K线数据
            points: 点位列表
            direction: 交易方向
            
        Returns:
            止损价格
        """
        point2 = points[1]  # 极值点
        point3 = points[2]  # 流动性获取点
        
        if point2 is None or point3 is None:
            return None
            
        if direction == "多":
            # 对于多单，止损设在点位3的低点下方
            return float(kline_data['low'][point3]) * 0.995
        else:  # 空单
            # 对于空单，止损设在点位3的高点上方
            return float(kline_data['high'][point3]) * 1.005
            
    def _calculate_take_profit(self, 
                            kline_data: Dict, 
                            points: List[Optional[int]], 
                            direction: str) -> Optional[float]:
        """
        计算止盈价格
        
        Args:
            kline_data: K线数据
            points: 点位列表
            direction: 交易方向
            
        Returns:
            止盈价格
        """
        point4 = points[3]  # 确认点
        point2 = points[1]  # 极值点
        
        if point2 is None or point4 is None:
            return None
            
        # 计算点位2到点位4的价格差
        if direction == "多":
            # 计算点位4的收盘价到点位2的低点的距离
            price_diff = float(kline_data['close'][point4]) - float(kline_data['low'][point2])
            # 按1.5倍设置止盈
            return float(kline_data['close'][point4]) + price_diff * 1.5
        else:  # 空单
            # 计算点位2的高点到点位4的收盘价的距离
            price_diff = float(kline_data['high'][point2]) - float(kline_data['close'][point4])
            # 按1.5倍设置止盈
            return float(kline_data['close'][point4]) - price_diff * 1.5
            
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型参数
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config
        }, model_path)
        
        logger.info(f"模型已保存到 {filepath}")
        
    @classmethod
    def load_model(cls, filepath: str) -> 'SBSPredictor':
        """
        加载模型
        
        Args:
            filepath: 模型路径
            
        Returns:
            加载的模型
        """
        # 加载保存的状态
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        # 创建模型实例
        model = cls(config=checkpoint['config'])
        
        # 加载模型参数
        model.load_state_dict(checkpoint['state_dict'])
        
        logger.info(f"已从 {filepath} 加载模型")
        return model 

    def supports_self_supervised(self):
        """
        检查模型是否支持自监督学习
        
        Returns:
            是否支持自监督学习
        """
        # 检查模型配置中是否启用了自监督学习功能
        return self.config.get('self_supervised', False) or hasattr(self, 'calculate_self_supervised_loss')
    
    def supports_rl(self):
        """
        检查模型是否支持强化学习
        
        Returns:
            是否支持强化学习
        """
        # 检查模型配置中是否启用了强化学习功能
        return self.config.get('rl_mode', False) or hasattr(self, 'calculate_rl_loss')
    
    def get_output_shape(self):
        """
        获取模型输出形状
        
        Returns:
            输出形状元组(batch_size, sequence_length, feature_dim)
        """
        # 根据模型架构返回输出形状
        if hasattr(self, 'output_shape'):
            return self.output_shape
        
        # 如果没有显式定义，使用配置中的输出大小推断
        feature_dim = self.config.get('output_size', 5)
        sequence_length = self.config.get('sequence_length', 1)
        
        # 标准输出形状：(batch_size, sequence_length, feature_dim)
        # 这里batch_size留空(None)，因为它是动态的
        return (None, sequence_length, feature_dim) 