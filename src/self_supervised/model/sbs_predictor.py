"""
SBS序列预测器
负责预测K线图中的SBS序列关键点位
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..utils.logger import setup_logger
from ...models.base_model import BaseModel

logger = setup_logger('sbs_predictor')

class SBSPredictor(BaseModel):
    """SBS序列预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化SBS预测器
        
        Args:
            config: 配置字典，包含模型参数
        """
        super().__init__(config)
        self.config.update({
            'confidence_threshold': 0.6,
            'max_points': 5,
            'window_size': 100,
            'min_pattern_bars': 10,
            'max_pattern_bars': 50
        })
        
        # 初始化模型组件
        self._init_model_components()
        
    def _init_model_components(self):
        """初始化模型组件"""
        # 特征提取器
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv1d(5, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2)
        )
        
        # 点位预测器
        self.point_predictor = torch.nn.Sequential(
            torch.nn.Linear(64 * 25, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 5)  # 5个点位
        )
        
        # 置信度预测器
        self.confidence_predictor = torch.nn.Sequential(
            torch.nn.Linear(64 * 25, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 5),  # 每个点位的置信度
            torch.nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, 5, sequence_length]
                5个通道分别是: open, high, low, close, volume
                
        Returns:
            包含点位预测和置信度的字典
        """
        # 特征提取
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # 预测点位
        points = self.point_predictor(features)
        
        # 预测置信度
        confidence = self.confidence_predictor(features)
        
        return {
            'points': points,
            'confidence': confidence
        }
        
    def predict_points(self, kline_data: pd.DataFrame) -> Dict[str, Any]:
        """
        预测SBS序列点位
        
        Args:
            kline_data: K线数据DataFrame
            
        Returns:
            预测结果字典
        """
        try:
            # 准备输入数据
            x = self._prepare_input(kline_data)
            x = x.to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output = self(x)
                
            # 处理预测结果
            points = output['points'].cpu().numpy()[0]
            confidence = output['confidence'].cpu().numpy()[0]
            
            # 构建预测结果
            predictions = {
                'points': {
                    'point1': int(points[0]),  # 第一次回调
                    'point2': int(points[1]),  # 极值点
                    'point3': int(points[2]),  # 流动性获取
                    'point4': int(points[3]),  # 确认点
                    'point5': int(points[4])   # 目标点
                },
                'confidence_scores': {
                    'point1': float(confidence[0]),
                    'point2': float(confidence[1]),
                    'point3': float(confidence[2]),
                    'point4': float(confidence[3]),
                    'point5': float(confidence[4])
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # 验证预测结果
            predictions = self._validate_predictions(predictions, len(kline_data))
            
            return predictions
            
        except Exception as e:
            logger.error(f"预测点位时发生错误: {str(e)}")
            raise
            
    def _prepare_input(self, kline_data: pd.DataFrame) -> torch.Tensor:
        """
        准备模型输入数据
        
        Args:
            kline_data: K线数据
            
        Returns:
            张量形式的输入数据
        """
        # 提取OHLCV数据
        ohlcv = kline_data[['open', 'high', 'low', 'close', 'volume']].values
        
        # 数据标准化
        ohlcv_normalized = self._normalize_data(ohlcv)
        
        # 转换为张量
        x = torch.FloatTensor(ohlcv_normalized)
        x = x.transpose(0, 1).unsqueeze(0)  # [1, 5, sequence_length]
        
        return x
        
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        标准化数据
        
        Args:
            data: 原始数据
            
        Returns:
            标准化后的数据
        """
        # OHLC使用对数收益率
        data[:, :4] = np.log(data[:, :4] / data[:, :4].mean(axis=1, keepdims=True))
        
        # 成交量使用标准化
        data[:, 4] = (data[:, 4] - data[:, 4].mean()) / data[:, 4].std()
        
        return data
        
    def _validate_predictions(self, 
                            predictions: Dict[str, Any], 
                            data_length: int) -> Dict[str, Any]:
        """
        验证并修正预测结果
        
        Args:
            predictions: 预测结果
            data_length: 数据长度
            
        Returns:
            修正后的预测结果
        """
        # 确保点位在有效范围内
        for point_name in predictions['points']:
            point_idx = predictions['points'][point_name]
            if point_idx is not None:
                predictions['points'][point_name] = max(0, min(point_idx, data_length - 1))
                
        # 确保点位顺序正确
        points = [(name, idx) for name, idx in predictions['points'].items() 
                 if idx is not None]
        points.sort(key=lambda x: x[1])
        
        for i, (name, _) in enumerate(points):
            predictions['points'][name] = points[i][1]
            
        return predictions 