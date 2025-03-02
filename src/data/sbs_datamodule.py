#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SBS数据模块
使用PyTorch Lightning的DataModule处理数据加载
"""

import torch
from typing import Optional, Dict, List
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SBSDataset(Dataset):
    """SBS数据集"""
    def __init__(self, 
                market_data: List[Dict], 
                labels: List[Dict],
                transform=None,
                target_transform=None):
        self.market_data = market_data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.scaler = StandardScaler()
        
        # 初始化数据预处理
        self._init_preprocessing()
        
    def __len__(self):
        return len(self.market_data)
        
    def __getitem__(self, idx):
        """获取单个样本"""
        market_item = self.market_data[idx]
        label_item = self.labels[idx]
        
        # 转换市场数据为张量
        market_tensor = self._convert_market_data_to_tensor(market_item)
        
        # 应用数据增强
        if self.transform is not None:
            market_tensor = self.transform(market_tensor)
        
        # 转换标签为张量
        label_tensor = self._convert_label_to_tensor(label_item)
        
        # 应用标签转换
        if self.target_transform is not None:
            label_tensor = self.target_transform(label_tensor)
        
        return market_tensor, label_tensor
        
    def _init_preprocessing(self):
        """初始化数据预处理"""
        # 提取所有特征用于拟合StandardScaler
        all_features = []
        for item in self.market_data:
            features = self._extract_features(item)
            all_features.append(features)
            
        # 拟合StandardScaler
        self.scaler.fit(all_features)
        
    def _extract_features(self, market_item: Dict) -> np.ndarray:
        """提取特征"""
        # 基础特征
        features = [
            market_item['open'],
            market_item['high'],
            market_item['low'],
            market_item['close'],
            market_item['volume']
        ]
        
        # 技术指标
        if 'indicators' in market_item:
            for indicator in market_item['indicators'].values():
                if isinstance(indicator, (list, np.ndarray)):
                    features.extend(indicator)
                else:
                    features.append(indicator)
                    
        return np.array(features).reshape(1, -1)
        
    def _convert_market_data_to_tensor(self, market_item: Dict) -> torch.Tensor:
        """将市场数据转换为张量"""
        # 提取特征
        features = self._extract_features(market_item)
        
        # 标准化
        normalized_features = self.scaler.transform(features)
        
        # 转换为张量
        return torch.tensor(normalized_features.flatten(), dtype=torch.float32)
        
    def _convert_label_to_tensor(self, label_item: Dict) -> torch.Tensor:
        """将标签转换为张量"""
        # 根据标注类型转换
        if 'class' in label_item:
            # 分类标签
            return torch.tensor(label_item['class'], dtype=torch.long)
        elif 'value' in label_item:
            # 回归标签
            return torch.tensor(label_item['value'], dtype=torch.float32)
        else:
            raise ValueError("未知的标签格式")

class SBSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        market_data: List[Dict],
        labels: List[Dict],
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        transform=None,
        target_transform=None
    ):
        """初始化数据模块"""
        super().__init__()
        self.market_data = market_data
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.transform = transform
        self.target_transform = target_transform
        
        # 数据增强设置
        self.train_transform = None
        self.val_transform = None
        self._setup_transforms()
        
    def _setup_transforms(self):
        """设置数据增强"""
        # 训练集数据增强
        self.train_transform = torch.nn.Sequential(
            # 添加噪声
            lambda x: x + torch.randn_like(x) * 0.01,
            # 随机缩放
            lambda x: x * (1 + torch.randn(1) * 0.1),
        )
        
        # 验证集不需要数据增强
        self.val_transform = None
        
    def setup(self, stage: Optional[str] = None):
        """准备数据集"""
        # 创建完整数据集
        full_dataset = SBSDataset(
            self.market_data, 
            self.labels,
            transform=self.transform,
            target_transform=self.target_transform
        )
        
        # 划分训练集和验证集
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 设置不同的转换
        self.train_dataset.dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.val_transform
        
        logger.info(f"训练集大小: {len(self.train_dataset)}")
        logger.info(f"验证集大小: {len(self.val_dataset)}")
        
    def train_dataloader(self):
        """返回训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        
    def val_dataloader(self):
        """返回验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        
    def get_input_size(self) -> int:
        """获取输入特征维度"""
        sample_item = self.market_data[0]
        sample_tensor = self._convert_market_data_to_tensor(sample_item)
        return sample_tensor.shape[0]
        
    def _convert_market_data_to_tensor(self, market_item: Dict) -> torch.Tensor:
        """将单个市场数据转换为张量"""
        return SBSDataset._convert_market_data_to_tensor(None, market_item)
        
    def save_scaler(self, filepath: str):
        """保存StandardScaler"""
        import joblib
        scaler_path = Path(filepath)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.train_dataset.dataset.scaler, scaler_path)
        logger.info(f"已保存StandardScaler到: {filepath}")
        
    def load_scaler(self, filepath: str):
        """加载StandardScaler"""
        import joblib
        self.train_dataset.dataset.scaler = joblib.load(filepath)
        self.val_dataset.dataset.scaler = self.train_dataset.dataset.scaler
        logger.info(f"已加载StandardScaler: {filepath}")
        
    @property
    def scaler(self):
        """获取StandardScaler"""
        return self.train_dataset.dataset.scaler if self.train_dataset else None 