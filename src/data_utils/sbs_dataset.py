#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SBS图像文本数据集类
用于加载和处理交易图表图像及相应的训练提示文本
"""

import os
import json
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path


class SBSImageTextDataset(Dataset):
    """
    SBS交易系统的图像-文本数据集
    用于加载图表图像和相应的提示文本对，适用于LLaVA模型训练
    """
    
    def __init__(self, data_dir, processor, image_size=(336, 336), max_length=512):
        """
        初始化数据集
        
        参数:
            data_dir: 数据目录，包含图像和标注文件
            processor: LLaVA处理器，用于处理图像和文本
            image_size: 图像大小，默认为336x336
            max_length: 文本最大长度，默认为512
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.image_size = image_size
        self.max_length = max_length
        
        # 加载数据集信息
        self.samples = self._load_dataset_info()
        logging.info(f"加载了{len(self.samples)}个训练样本")
        
    def _load_dataset_info(self):
        """
        加载数据集信息，包括图像路径和对应的提示文本
        
        返回:
            样本列表，每个样本包含图像路径和提示文本
        """
        # 加载元数据文件
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 如果元数据文件不存在，则遍历目录收集数据
        samples = []
        image_dir = self.data_dir / "images"
        text_dir = self.data_dir / "texts"
        
        if not image_dir.exists() or not text_dir.exists():
            raise ValueError(f"数据目录结构不正确，请确保{image_dir}和{text_dir}目录存在")
        
        # 遍历图像目录
        for img_file in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
            img_name = img_file.stem
            txt_file = text_dir / f"{img_name}.txt"
            
            # 检查对应的文本文件是否存在
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    prompt_text = f.read().strip()
                
                samples.append({
                    "image_path": str(img_file.relative_to(self.data_dir)),
                    "prompt": prompt_text
                })
            else:
                logging.warning(f"图像{img_file}没有对应的文本文件{txt_file}")
        
        # 保存元数据以供将来使用
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
            
        return samples
    
    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        参数:
            idx: 样本索引
            
        返回:
            处理后的样本，包括图像特征和文本特征
        """
        sample = self.samples[idx]
        
        # 加载图像
        image_path = self.data_dir / sample["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            # 调整图像大小
            if self.image_size:
                image = image.resize(self.image_size, Image.LANCZOS)
        except Exception as e:
            logging.error(f"加载图像{image_path}失败: {e}")
            raise
        
        # 获取提示文本
        prompt_text = sample["prompt"]
        
        # 使用处理器处理图像和文本
        try:
            inputs = self.processor(
                text=prompt_text,
                images=image,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # 移除批次维度
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
            
            return inputs
        except Exception as e:
            logging.error(f"处理样本{idx}时发生错误: {e}")
            raise


class SBSActiveLearningDataset(SBSImageTextDataset):
    """
    用于主动学习的SBS数据集
    扩展基本数据集以支持不确定性采样和主动学习策略
    """
    
    def __init__(self, data_dir, processor, image_size=(336, 336), max_length=512):
        """
        初始化主动学习数据集
        
        参数:
            data_dir: 数据目录
            processor: LLaVA处理器
            image_size: 图像大小
            max_length: 文本最大长度
        """
        super().__init__(data_dir, processor, image_size, max_length)
        
        # 初始化样本权重
        self.sample_weights = torch.ones(len(self.samples))
        
        # 加载已标记的样本索引
        self.labeled_indices = set()
        labeled_file = self.data_dir / "labeled_indices.txt"
        if labeled_file.exists():
            with open(labeled_file, 'r') as f:
                self.labeled_indices = set(int(line.strip()) for line in f if line.strip())
        
    def update_sample_weights(self, indices, weights):
        """
        更新样本权重用于主动学习采样
        
        参数:
            indices: 要更新的样本索引
            weights: 对应的权重值
        """
        for idx, weight in zip(indices, weights):
            if 0 <= idx < len(self.sample_weights):
                self.sample_weights[idx] = weight
    
    def mark_as_labeled(self, indices):
        """
        将指定索引的样本标记为已标记
        
        参数:
            indices: 已标记样本的索引列表
        """
        self.labeled_indices.update(indices)
        
        # 保存已标记的索引
        labeled_file = self.data_dir / "labeled_indices.txt"
        with open(labeled_file, 'w') as f:
            for idx in sorted(self.labeled_indices):
                f.write(f"{idx}\n")
    
    def get_unlabeled_indices(self):
        """
        获取未标记样本的索引
        
        返回:
            未标记样本的索引列表
        """
        all_indices = set(range(len(self.samples)))
        return list(all_indices - self.labeled_indices)
    
    def get_sample_metadata(self, idx):
        """
        获取样本的元数据信息
        
        参数:
            idx: 样本索引
            
        返回:
            样本的元数据字典
        """
        sample = self.samples[idx].copy()
        sample['is_labeled'] = idx in self.labeled_indices
        sample['weight'] = float(self.sample_weights[idx])
        return sample 