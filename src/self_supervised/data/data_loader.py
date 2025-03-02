#!/usr/bin/env python
"""
SBS数据加载器
为不同训练模式提供数据加载功能
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import logging
from tqdm import tqdm

from ..utils.logger import setup_logger

logger = setup_logger('data_loader')

class SBSDataset(Dataset):
    """
    SBS序列数据集基类
    """
    
    def __init__(self, data_path: str, config: Dict = None, transform=None):
        """
        初始化SBS数据集
        
        参数:
            data_path: 数据文件或目录路径
            config: 配置参数
            transform: 数据转换函数
        """
        self.config = config or {}
        self.data_path = data_path
        self.transform = transform
        self.data = []
        self.labels = []
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载数据，子类应重写此方法"""
        raise NotImplementedError("子类必须实现_load_data方法")
        
    def __len__(self):
        """返回数据集长度"""
        return len(self.data)
        
    def __getitem__(self, idx):
        """获取指定索引的数据项"""
        item = self.data[idx]
        label = self.labels[idx] if self.labels else None
        
        if self.transform:
            item = self.transform(item)
            
        if label is not None:
            return item, label
        else:
            return item


class StandardDataset(SBSDataset):
    """标准训练数据集，用于监督学习"""
    
    def _load_data(self):
        """加载标准数据集"""
        logger.info(f"加载标准数据集: {self.data_path}")
        
        try:
            # 检查数据路径类型
            data_path = Path(self.data_path)
            if data_path.is_file():
                # 单个文件
                if data_path.suffix == '.csv':
                    self._load_from_csv(data_path)
                elif data_path.suffix == '.npz':
                    self._load_from_npz(data_path)
                else:
                    logger.error(f"不支持的文件格式: {data_path.suffix}")
            elif data_path.is_dir():
                # 目录，加载所有支持的文件
                self._load_from_directory(data_path)
            else:
                logger.error(f"数据路径不存在: {self.data_path}")
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            
    def _load_from_csv(self, file_path: Path):
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(file_path)
            
            # 预处理CSV数据，提取特征和标签
            # 这里需要根据实际CSV格式调整列名
            feature_cols = self.config.get('feature_columns', ['open', 'high', 'low', 'close', 'volume'])
            label_cols = self.config.get('label_columns', ['point1', 'point2', 'point3', 'point4', 'point5', 'direction'])
            
            # 检查列是否存在
            missing_feature_cols = [col for col in feature_cols if col not in df.columns]
            if missing_feature_cols:
                logger.warning(f"CSV中缺少特征列: {missing_feature_cols}")
                
            missing_label_cols = [col for col in label_cols if col not in df.columns]
            if missing_label_cols:
                logger.warning(f"CSV中缺少标签列: {missing_label_cols}")
            
            # 提取特征和标签
            features = df[feature_cols].values
            labels = df[label_cols].values if all(col in df.columns for col in label_cols) else None
            
            # 转换为张量并添加到数据集
            for i in range(len(features)):
                self.data.append(torch.tensor(features[i], dtype=torch.float32))
                if labels is not None:
                    self.labels.append(torch.tensor(labels[i], dtype=torch.float32))
                    
            logger.info(f"从CSV加载了{len(self.data)}个数据项")
        except Exception as e:
            logger.error(f"从CSV加载数据失败: {e}")
            
    def _load_from_npz(self, file_path: Path):
        """从NPZ文件加载数据"""
        try:
            data = np.load(file_path)
            
            # 检查NPZ文件中的数组
            if 'features' in data and 'labels' in data:
                features = data['features']
                labels = data['labels']
                
                # 转换为张量并添加到数据集
                for i in range(len(features)):
                    self.data.append(torch.tensor(features[i], dtype=torch.float32))
                    self.labels.append(torch.tensor(labels[i], dtype=torch.float32))
                    
                logger.info(f"从NPZ加载了{len(self.data)}个数据项")
            else:
                logger.error(f"NPZ文件缺少必要的数组: features或labels")
        except Exception as e:
            logger.error(f"从NPZ加载数据失败: {e}")
            
    def _load_from_directory(self, dir_path: Path):
        """从目录加载数据"""
        # 支持的文件格式
        supported_exts = ['.csv', '.npz']
        
        # 查找所有支持的文件
        files = []
        for ext in supported_exts:
            files.extend(list(dir_path.glob(f'*{ext}')))
            
        if not files:
            logger.warning(f"目录中没有找到支持的数据文件: {dir_path}")
            return
            
        # 加载每个文件
        for file in tqdm(files, desc="加载数据文件"):
            if file.suffix == '.csv':
                self._load_from_csv(file)
            elif file.suffix == '.npz':
                self._load_from_npz(file)


class SelfSupervisedDataset(SBSDataset):
    """自监督学习数据集，用于无标签数据"""
    
    def __init__(self, data_path: str, config: Dict = None, transform=None, labeled_path: str = None):
        """
        初始化自监督数据集
        
        参数:
            data_path: 无标签数据文件或目录路径
            config: 配置参数
            transform: 数据转换函数
            labeled_path: 有标签数据路径（可选）
        """
        self.labeled_path = labeled_path
        self.labeled_data = []
        self.labeled_labels = []
        
        super().__init__(data_path, config, transform)
        
    def _load_data(self):
        """加载自监督数据集"""
        logger.info(f"加载自监督数据集: {self.data_path}")
        
        # 加载无标签数据
        try:
            data_path = Path(self.data_path)
            if data_path.is_file():
                self._load_unlabeled_from_file(data_path)
            elif data_path.is_dir():
                self._load_unlabeled_from_directory(data_path)
            else:
                logger.error(f"无标签数据路径不存在: {self.data_path}")
        except Exception as e:
            logger.error(f"加载无标签数据失败: {e}")
            
        # 加载有标签数据（如果提供）
        if self.labeled_path:
            try:
                labeled_path = Path(self.labeled_path)
                if labeled_path.is_file():
                    self._load_labeled_from_file(labeled_path)
                elif labeled_path.is_dir():
                    self._load_labeled_from_directory(labeled_path)
                else:
                    logger.error(f"有标签数据路径不存在: {self.labeled_path}")
            except Exception as e:
                logger.error(f"加载有标签数据失败: {e}")
                
    def _load_unlabeled_from_file(self, file_path: Path):
        """从文件加载无标签数据"""
        if file_path.suffix == '.csv':
            try:
                df = pd.read_csv(file_path)
                feature_cols = self.config.get('feature_columns', ['open', 'high', 'low', 'close', 'volume'])
                
                # 检查列是否存在
                missing_cols = [col for col in feature_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"CSV中缺少列: {missing_cols}")
                    
                # 提取特征
                features = df[feature_cols].values
                
                # 转换为张量并添加到数据集
                for i in range(len(features)):
                    self.data.append(torch.tensor(features[i], dtype=torch.float32))
                    
                logger.info(f"从CSV加载了{len(self.data)}个无标签数据项")
            except Exception as e:
                logger.error(f"从CSV加载无标签数据失败: {e}")
        elif file_path.suffix == '.npz':
            try:
                data = np.load(file_path)
                
                if 'features' in data:
                    features = data['features']
                    
                    # 转换为张量并添加到数据集
                    for i in range(len(features)):
                        self.data.append(torch.tensor(features[i], dtype=torch.float32))
                        
                    logger.info(f"从NPZ加载了{len(self.data)}个无标签数据项")
                else:
                    logger.error(f"NPZ文件缺少必要的数组: features")
            except Exception as e:
                logger.error(f"从NPZ加载无标签数据失败: {e}")
                
    def _load_unlabeled_from_directory(self, dir_path: Path):
        """从目录加载无标签数据"""
        # 支持的文件格式
        supported_exts = ['.csv', '.npz']
        
        # 查找所有支持的文件
        files = []
        for ext in supported_exts:
            files.extend(list(dir_path.glob(f'*{ext}')))
            
        if not files:
            logger.warning(f"目录中没有找到支持的数据文件: {dir_path}")
            return
            
        # 加载每个文件
        for file in tqdm(files, desc="加载无标签数据文件"):
            self._load_unlabeled_from_file(file)
            
    def _load_labeled_from_file(self, file_path: Path):
        """从文件加载有标签数据"""
        # 实现类似StandardDataset的加载逻辑
        if file_path.suffix == '.csv':
            try:
                df = pd.read_csv(file_path)
                
                feature_cols = self.config.get('feature_columns', ['open', 'high', 'low', 'close', 'volume'])
                label_cols = self.config.get('label_columns', ['point1', 'point2', 'point3', 'point4', 'point5', 'direction'])
                
                # 检查列是否存在
                missing_feature_cols = [col for col in feature_cols if col not in df.columns]
                if missing_feature_cols:
                    logger.warning(f"CSV中缺少特征列: {missing_feature_cols}")
                    
                missing_label_cols = [col for col in label_cols if col not in df.columns]
                if missing_label_cols:
                    logger.warning(f"CSV中缺少标签列: {missing_label_cols}")
                
                # 提取特征和标签
                features = df[feature_cols].values
                labels = df[label_cols].values if all(col in df.columns for col in label_cols) else None
                
                # 转换为张量并添加到数据集
                for i in range(len(features)):
                    self.labeled_data.append(torch.tensor(features[i], dtype=torch.float32))
                    if labels is not None:
                        self.labeled_labels.append(torch.tensor(labels[i], dtype=torch.float32))
                        
                logger.info(f"从CSV加载了{len(self.labeled_data)}个有标签数据项")
            except Exception as e:
                logger.error(f"从CSV加载有标签数据失败: {e}")
        elif file_path.suffix == '.npz':
            # NPZ文件加载逻辑
            pass
            
    def _load_labeled_from_directory(self, dir_path: Path):
        """从目录加载有标签数据"""
        # 支持的文件格式
        supported_exts = ['.csv', '.npz']
        
        # 查找所有支持的文件
        files = []
        for ext in supported_exts:
            files.extend(list(dir_path.glob(f'*{ext}')))
            
        if not files:
            logger.warning(f"目录中没有找到支持的数据文件: {dir_path}")
            return
            
        # 加载每个文件
        for file in tqdm(files, desc="加载有标签数据文件"):
            self._load_labeled_from_file(file)
            
    def __len__(self):
        """返回数据集长度"""
        return len(self.data)
        
    def __getitem__(self, idx):
        """获取指定索引的数据项"""
        # 对于自监督学习，可能需要生成不同视角的数据或数据增强
        item = self.data[idx]
        
        if self.transform:
            item_augmented = self.transform(item)
            return item, item_augmented
        else:
            # 如果没有提供转换函数，返回相同项两次（原始方法）
            return item, item


class RLDataset(SBSDataset):
    """强化学习数据集，用于与环境交互的数据"""
    
    def __init__(self, data_path: str, config: Dict = None, transform=None, window_size: int = 100):
        """
        初始化强化学习数据集
        
        参数:
            data_path: 市场数据文件或目录路径
            config: 配置参数
            transform: 数据转换函数
            window_size: 窗口大小，表示每个状态的时间步长
        """
        self.window_size = window_size
        self.market_data = None
        super().__init__(data_path, config, transform)
        
    def _load_data(self):
        """加载市场数据"""
        logger.info(f"加载市场数据: {self.data_path}")
        
        try:
            data_path = Path(self.data_path)
            if data_path.suffix == '.csv':
                self._load_from_csv(data_path)
            elif data_path.suffix == '.npz':
                self._load_from_npz(data_path)
            else:
                logger.error(f"不支持的文件格式: {data_path.suffix}")
        except Exception as e:
            logger.error(f"加载市场数据失败: {e}")
            
    def _load_from_csv(self, file_path: Path):
        """从CSV文件加载市场数据"""
        try:
            df = pd.read_csv(file_path)
            
            # 预处理CSV数据
            feature_cols = self.config.get('feature_columns', ['open', 'high', 'low', 'close', 'volume'])
            
            # 检查列是否存在
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"CSV中缺少列: {missing_cols}")
                
            # 提取特征
            self.market_data = df[feature_cols].values
            
            # 创建滑动窗口作为状态
            for i in range(len(self.market_data) - self.window_size):
                window = self.market_data[i:i+self.window_size]
                self.data.append(torch.tensor(window, dtype=torch.float32))
                
            logger.info(f"从CSV加载了{len(self.market_data)}条市场数据，生成了{len(self.data)}个状态")
        except Exception as e:
            logger.error(f"从CSV加载市场数据失败: {e}")
            
    def _load_from_npz(self, file_path: Path):
        """从NPZ文件加载市场数据"""
        try:
            data = np.load(file_path)
            
            if 'market_data' in data:
                self.market_data = data['market_data']
                
                # 创建滑动窗口作为状态
                for i in range(len(self.market_data) - self.window_size):
                    window = self.market_data[i:i+self.window_size]
                    self.data.append(torch.tensor(window, dtype=torch.float32))
                    
                logger.info(f"从NPZ加载了{len(self.market_data)}条市场数据，生成了{len(self.data)}个状态")
            else:
                logger.error(f"NPZ文件缺少必要的数组: market_data")
        except Exception as e:
            logger.error(f"从NPZ加载市场数据失败: {e}")
            
    def get_state(self, index: int):
        """获取指定索引的状态"""
        if index < 0 or index >= len(self.data):
            logger.error(f"索引超出范围: {index}")
            return None
            
        return self.data[index]
        
    def get_next_state(self, index: int):
        """获取指定索引的下一个状态"""
        next_index = index + 1
        if next_index >= len(self.data):
            return None
            
        return self.data[next_index]


class ActiveLearningDataset(SBSDataset):
    """主动学习数据集，管理有标签和无标签数据池"""
    
    def __init__(self, unlabeled_path: str, labeled_path: str = None, config: Dict = None, transform=None):
        """
        初始化主动学习数据集
        
        参数:
            unlabeled_path: 无标签数据文件或目录路径
            labeled_path: 有标签数据文件或目录路径（可选）
            config: 配置参数
            transform: 数据转换函数
        """
        self.unlabeled_path = unlabeled_path
        self.labeled_path = labeled_path
        
        self.unlabeled_data = []  # 无标签数据池
        self.unlabeled_indices = []  # 无标签数据的原始索引
        
        self.labeled_data = []  # 有标签数据
        self.labeled_labels = []  # 有标签数据的标签
        
        super().__init__(unlabeled_path, config, transform)
        
    def _load_data(self):
        """加载数据"""
        # 加载无标签数据
        self._load_unlabeled_data()
        
        # 加载有标签数据（如果提供）
        if self.labeled_path:
            self._load_labeled_data()
            
        # 初始化数据和标签
        self._update_data()
        
    def _load_unlabeled_data(self):
        """加载无标签数据"""
        logger.info(f"加载无标签数据: {self.unlabeled_path}")
        
        try:
            path = Path(self.unlabeled_path)
            if path.is_file():
                self._load_unlabeled_from_file(path)
            elif path.is_dir():
                self._load_unlabeled_from_directory(path)
            else:
                logger.error(f"无标签数据路径不存在: {self.unlabeled_path}")
        except Exception as e:
            logger.error(f"加载无标签数据失败: {e}")
            
    def _load_unlabeled_from_file(self, file_path: Path):
        """从文件加载无标签数据"""
        if file_path.suffix == '.csv':
            try:
                df = pd.read_csv(file_path)
                
                feature_cols = self.config.get('feature_columns', ['open', 'high', 'low', 'close', 'volume'])
                
                # 检查列是否存在
                missing_cols = [col for col in feature_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"CSV中缺少列: {missing_cols}")
                    
                # 提取特征
                features = df[feature_cols].values
                
                # 转换为张量并添加到数据集
                for i in range(len(features)):
                    self.unlabeled_data.append(torch.tensor(features[i], dtype=torch.float32))
                    self.unlabeled_indices.append(i)
                    
                logger.info(f"从CSV加载了{len(self.unlabeled_data)}个无标签数据项")
            except Exception as e:
                logger.error(f"从CSV加载无标签数据失败: {e}")
        elif file_path.suffix == '.npz':
            # NPZ文件加载逻辑
            pass
            
    def _load_unlabeled_from_directory(self, dir_path: Path):
        """从目录加载无标签数据"""
        # 支持的文件格式
        supported_exts = ['.csv', '.npz']
        
        # 查找所有支持的文件
        files = []
        for ext in supported_exts:
            files.extend(list(dir_path.glob(f'*{ext}')))
            
        if not files:
            logger.warning(f"目录中没有找到支持的数据文件: {dir_path}")
            return
            
        # 加载每个文件
        for file in tqdm(files, desc="加载无标签数据文件"):
            self._load_unlabeled_from_file(file)
            
    def _load_labeled_data(self):
        """加载有标签数据"""
        logger.info(f"加载有标签数据: {self.labeled_path}")
        
        try:
            path = Path(self.labeled_path)
            if path.is_file():
                self._load_labeled_from_file(path)
            elif path.is_dir():
                self._load_labeled_from_directory(path)
            else:
                logger.error(f"有标签数据路径不存在: {self.labeled_path}")
        except Exception as e:
            logger.error(f"加载有标签数据失败: {e}")
            
    def _load_labeled_from_file(self, file_path: Path):
        """从文件加载有标签数据"""
        if file_path.suffix == '.csv':
            try:
                df = pd.read_csv(file_path)
                
                feature_cols = self.config.get('feature_columns', ['open', 'high', 'low', 'close', 'volume'])
                label_cols = self.config.get('label_columns', ['point1', 'point2', 'point3', 'point4', 'point5', 'direction'])
                
                # 检查列是否存在
                missing_feature_cols = [col for col in feature_cols if col not in df.columns]
                if missing_feature_cols:
                    logger.warning(f"CSV中缺少特征列: {missing_feature_cols}")
                    
                missing_label_cols = [col for col in label_cols if col not in df.columns]
                if missing_label_cols:
                    logger.warning(f"CSV中缺少标签列: {missing_label_cols}")
                
                # 提取特征和标签
                features = df[feature_cols].values
                labels = df[label_cols].values if all(col in df.columns for col in label_cols) else None
                
                # 转换为张量并添加到数据集
                for i in range(len(features)):
                    self.labeled_data.append(torch.tensor(features[i], dtype=torch.float32))
                    if labels is not None:
                        self.labeled_labels.append(torch.tensor(labels[i], dtype=torch.float32))
                        
                logger.info(f"从CSV加载了{len(self.labeled_data)}个有标签数据项")
            except Exception as e:
                logger.error(f"从CSV加载有标签数据失败: {e}")
        elif file_path.suffix == '.npz':
            # NPZ文件加载逻辑
            pass
            
    def _load_labeled_from_directory(self, dir_path: Path):
        """从目录加载有标签数据"""
        # 支持的文件格式
        supported_exts = ['.csv', '.npz']
        
        # 查找所有支持的文件
        files = []
        for ext in supported_exts:
            files.extend(list(dir_path.glob(f'*{ext}')))
            
        if not files:
            logger.warning(f"目录中没有找到支持的数据文件: {dir_path}")
            return
            
        # 加载每个文件
        for file in tqdm(files, desc="加载有标签数据文件"):
            self._load_labeled_from_file(file)
            
    def _update_data(self):
        """更新数据和标签"""
        # 合并有标签数据
        self.data = self.labeled_data.copy()
        self.labels = self.labeled_labels.copy()
        
    def get_unlabeled_pool(self):
        """获取无标签数据池"""
        return self.unlabeled_data, self.unlabeled_indices
        
    def add_labeled_data(self, indices: List[int], labels: List[torch.Tensor]):
        """
        将无标签数据移至有标签数据池
        
        参数:
            indices: 需要标注的无标签数据索引列表
            labels: 对应的标签列表
        """
        if len(indices) != len(labels):
            logger.error(f"索引数量({len(indices)})与标签数量({len(labels)})不匹配")
            return
            
        # 添加到有标签数据池
        for idx, label in zip(indices, labels):
            if 0 <= idx < len(self.unlabeled_data):
                data_item = self.unlabeled_data[idx]
                self.labeled_data.append(data_item)
                self.labeled_labels.append(label)
                
        # 从无标签池中删除
        new_unlabeled_data = []
        new_unlabeled_indices = []
        for i, item in enumerate(self.unlabeled_data):
            if i not in indices:
                new_unlabeled_data.append(item)
                new_unlabeled_indices.append(self.unlabeled_indices[i])
                
        self.unlabeled_data = new_unlabeled_data
        self.unlabeled_indices = new_unlabeled_indices
        
        # 更新数据和标签
        self._update_data()
        
        logger.info(f"已添加{len(indices)}个标注样本，现有{len(self.labeled_data)}个有标签样本，{len(self.unlabeled_data)}个无标签样本")


class SBSDataLoader:
    """SBS数据加载器，用于创建不同训练模式的数据加载器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化SBS数据加载器
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        
    def get_standard_dataloaders(self, train_path: str, val_path: str = None, test_path: str = None, 
                             batch_size: int = 32, shuffle: bool = True, num_workers: int = 4):
        """
        创建标准训练的数据加载器
        
        参数:
            train_path: 训练数据路径
            val_path: 验证数据路径（可选）
            test_path: 测试数据路径（可选）
            batch_size: 批处理大小
            shuffle: 是否打乱数据
            num_workers: 数据加载的工作线程数
            
        返回:
            训练数据加载器、验证数据加载器、测试数据加载器
        """
        train_dataset = StandardDataset(train_path, self.config)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
        
        val_loader = None
        if val_path:
            val_dataset = StandardDataset(val_path, self.config)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers
            )
            
        test_loader = None
        if test_path:
            test_dataset = StandardDataset(test_path, self.config)
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers
            )
            
        return train_loader, val_loader, test_loader
        
    def get_self_supervised_dataloaders(self, unlabeled_path: str, labeled_path: str = None, val_path: str = None,
                                    batch_size: int = 32, shuffle: bool = True, num_workers: int = 4):
        """
        创建自监督训练的数据加载器
        
        参数:
            unlabeled_path: 无标签数据路径
            labeled_path: 有标签数据路径（可选）
            val_path: 验证数据路径（可选）
            batch_size: 批处理大小
            shuffle: 是否打乱数据
            num_workers: 数据加载的工作线程数
            
        返回:
            自监督数据加载器、监督数据加载器、验证数据加载器
        """
        # 创建自监督数据集
        self_supervised_dataset = SelfSupervisedDataset(unlabeled_path, self.config, labeled_path=labeled_path)
        self_supervised_loader = DataLoader(
            self_supervised_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
        
        # 如果有标注数据，创建监督数据加载器
        supervised_loader = None
        if labeled_path:
            supervised_dataset = StandardDataset(labeled_path, self.config)
            supervised_loader = DataLoader(
                supervised_dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=num_workers
            )
            
        # 如果有验证数据，创建验证数据加载器
        val_loader = None
        if val_path:
            val_dataset = StandardDataset(val_path, self.config)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers
            )
            
        return self_supervised_loader, supervised_loader, val_loader
        
    def get_rl_dataloader(self, market_path: str, window_size: int = 100, batch_size: int = 32, shuffle: bool = False, 
                     num_workers: int = 4):
        """
        创建强化学习的数据加载器
        
        参数:
            market_path: 市场数据路径
            window_size: 窗口大小
            batch_size: 批处理大小
            shuffle: 是否打乱数据
            num_workers: 数据加载的工作线程数
            
        返回:
            市场数据加载器
        """
        rl_dataset = RLDataset(market_path, self.config, window_size=window_size)
        rl_loader = DataLoader(
            rl_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
        
        return rl_loader
        
    def get_active_learning_dataloaders(self, unlabeled_path: str, labeled_path: str = None, batch_size: int = 32, 
                                    shuffle: bool = True, num_workers: int = 4):
        """
        创建主动学习的数据加载器
        
        参数:
            unlabeled_path: 无标签数据路径
            labeled_path: 有标签数据路径（可选）
            batch_size: 批处理大小
            shuffle: 是否打乱数据
            num_workers: 数据加载的工作线程数
            
        返回:
            主动学习数据集、有标签数据加载器
        """
        # 创建主动学习数据集
        active_dataset = ActiveLearningDataset(unlabeled_path, labeled_path, self.config)
        
        # 创建有标签数据加载器
        labeled_loader = DataLoader(
            active_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
        
        return active_dataset, labeled_loader 