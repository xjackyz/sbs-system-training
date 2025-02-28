"""
验证工具模块
实现模型验证和评估功能
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import torch
from torch.utils.data import DataLoader

from .logger import setup_logger
from .exceptions import ValidationError

logger = setup_logger('validation')

class ModelValidator:
    """模型验证器"""
    
    def __init__(self, config: Dict):
        """
        初始化验证器
        
        Args:
            config: 验证配置
        """
        self.config = config
        self.metrics_history = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
    def validate_sequence(self, 
                         model: torch.nn.Module,
                         dataset: torch.utils.data.Dataset,
                         n_splits: int = 5) -> Dict:
        """
        序列识别验证
        
        Args:
            model: 待验证的模型
            dataset: 数据集
            n_splits: 分割数量
            
        Returns:
            验证结果
        """
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(dataset)):
                logger.info(f"验证折叠 {fold + 1}/{n_splits}")
                
                # 创建数据加载器
                train_loader = DataLoader(
                    torch.utils.data.Subset(dataset, train_idx),
                    batch_size=self.config['batch_size'],
                    shuffle=False
                )
                val_loader = DataLoader(
                    torch.utils.data.Subset(dataset, val_idx),
                    batch_size=self.config['batch_size'],
                    shuffle=False
                )
                
                # 验证当前折叠
                metrics = self._validate_fold(model, train_loader, val_loader)
                fold_metrics.append(metrics)
                
            # 计算平均指标
            avg_metrics = self._calculate_average_metrics(fold_metrics)
            
            # 更新历史记录
            self._update_metrics_history(avg_metrics)
            
            return avg_metrics
            
        except Exception as e:
            logger.error(f"序列验证失败: {str(e)}")
            raise ValidationError(f"序列验证失败: {str(e)}")
            
    def _validate_fold(self,
                      model: torch.nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader) -> Dict:
        """
        验证单个折叠
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            验证指标
        """
        model.eval()
        metrics = {
            'loss': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # 前向传播
                outputs = model(batch)
                
                # 计算指标
                batch_metrics = self._calculate_metrics(outputs, batch)
                
                # 更新累积指标
                for key in metrics:
                    metrics[key] += batch_metrics[key]
                    
        # 计算平均值
        num_batches = len(val_loader)
        for key in metrics:
            metrics[key] /= num_batches
            
        return metrics
        
    def _calculate_metrics(self, outputs: torch.Tensor, batch: Dict) -> Dict:
        """
        计算批次指标
        
        Args:
            outputs: 模型输出
            batch: 批次数据
            
        Returns:
            批次指标
        """
        try:
            # 获取预测和真实标签
            predictions = torch.argmax(outputs, dim=1)
            targets = batch['labels']
            
            # 计算各项指标
            correct = (predictions == targets).float()
            accuracy = correct.mean().item()
            
            # 计算精确率、召回率和F1分数
            tp = ((predictions == 1) & (targets == 1)).float().sum()
            fp = ((predictions == 1) & (targets == 0)).float().sum()
            fn = ((predictions == 0) & (targets == 1)).float().sum()
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            
            # 计算损失
            loss = torch.nn.functional.cross_entropy(outputs, targets).item()
            
            return {
                'loss': loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
        except Exception as e:
            logger.error(f"计算指标失败: {str(e)}")
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
    def _calculate_average_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """
        计算平均指标
        
        Args:
            fold_metrics: 各折叠的指标
            
        Returns:
            平均指标
        """
        avg_metrics = {
            'loss': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
        
        n_folds = len(fold_metrics)
        for metrics in fold_metrics:
            for key in avg_metrics:
                avg_metrics[key] += metrics[key]
                
        for key in avg_metrics:
            avg_metrics[key] /= n_folds
            
        return avg_metrics
        
    def _update_metrics_history(self, metrics: Dict) -> None:
        """
        更新指标历史记录
        
        Args:
            metrics: 当前指标
        """
        for key in self.metrics_history:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
                
    def get_metrics_history(self) -> Dict:
        """
        获取指标历史记录
        
        Returns:
            指标历史记录
        """
        return self.metrics_history 