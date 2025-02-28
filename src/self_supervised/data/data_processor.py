import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime
import cv2
from PIL import Image
import torch


class DataProcessor:
    """数据处理器
    
    用于准备自监督学习的训练数据，包括图表数据处理和标签生成。
    """
    
    def __init__(self, config: Dict = None):
        """初始化数据处理器
        
        Args:
            config: 配置参数
        """
        # 默认配置
        default_config = {
            'image_size': (224, 224),
            'sequence_length': 100,
            'window_size': 100,
            'stride': 20,
            'min_sequence_length': 50,
            'max_sequence_length': 200,
            'augmentation': {
                'enabled': True,
                'time_scaling': True,
                'price_scaling': True,
                'noise_injection': True
            }
        }
        
        # 更新配置
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.logger = logging.getLogger('data_processor')
    
    def process_chart_image(self, image_path: str) -> np.ndarray:
        """处理图表图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            processed_image: 处理后的图像数组
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        image = cv2.resize(image, self.config['image_size'])
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def process_chart_sequence(self, image_paths: List[str]) -> np.ndarray:
        """处理图表序列
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            processed_sequence: 处理后的序列数组
        """
        # 处理每个图像
        processed_images = []
        for path in image_paths:
            try:
                image = self.process_chart_image(path)
                processed_images.append(image)
            except Exception as e:
                self.logger.warning(f"处理图像 {path} 失败: {e}")
        
        # 确保序列长度一致
        if len(processed_images) < self.config['sequence_length']:
            # 序列太短，填充
            padding = [processed_images[-1]] * (self.config['sequence_length'] - len(processed_images))
            processed_images.extend(padding)
        elif len(processed_images) > self.config['sequence_length']:
            # 序列太长，截断
            processed_images = processed_images[:self.config['sequence_length']]
        
        # 转换为数组
        sequence = np.array(processed_images)
        
        return sequence
    
    def augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """增强序列数据
        
        Args:
            sequence: 原始序列
            
        Returns:
            augmented_sequence: 增强后的序列
        """
        if not self.config['augmentation']['enabled']:
            return sequence
        
        augmented = sequence.copy()
        
        # 时间缩放
        if self.config['augmentation']['time_scaling'] and np.random.random() < 0.5:
            scale_factor = np.random.uniform(0.8, 1.2)
            seq_len = len(augmented)
            scaled_len = int(seq_len * scale_factor)
            
            if scaled_len < seq_len:
                # 缩短序列
                indices = np.linspace(0, seq_len - 1, scaled_len).astype(int)
                augmented = augmented[indices]
                
                # 填充到原始长度
                padding = [augmented[-1]] * (seq_len - scaled_len)
                augmented = np.concatenate([augmented, padding])
            elif scaled_len > seq_len:
                # 延长序列
                indices = np.linspace(0, seq_len - 1, scaled_len).astype(int)
                augmented = augmented[indices]
                
                # 截断到原始长度
                augmented = augmented[:seq_len]
        
        # 价格缩放
        if self.config['augmentation']['price_scaling'] and np.random.random() < 0.5:
            scale_factor = np.random.uniform(0.9, 1.1)
            augmented = augmented * scale_factor
            
            # 裁剪到有效范围
            augmented = np.clip(augmented, 0.0, 1.0)
        
        # 噪声注入
        if self.config['augmentation']['noise_injection'] and np.random.random() < 0.5:
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, augmented.shape)
            augmented = augmented + noise
            
            # 裁剪到有效范围
            augmented = np.clip(augmented, 0.0, 1.0)
        
        return augmented
    
    def prepare_training_data(self, 
                             chart_data: Dict[str, Any], 
                             labels: Dict[str, Any] = None) -> Dict[str, Any]:
        """准备训练数据
        
        Args:
            chart_data: 图表数据
            labels: 标签数据
            
        Returns:
            training_data: 训练数据
        """
        # 处理图表数据
        if 'image_paths' in chart_data:
            # 处理图像序列
            sequence = self.process_chart_sequence(chart_data['image_paths'])
            
            # 数据增强
            sequence = self.augment_sequence(sequence)
            
            processed_data = {
                'sequence': sequence
            }
        elif 'price_data' in chart_data:
            # 处理价格数据
            price_data = chart_data['price_data']
            
            # 转换为数组
            if isinstance(price_data, list):
                price_array = np.array(price_data)
            elif isinstance(price_data, dict):
                # 假设价格数据是字典格式，包含 'open', 'high', 'low', 'close' 等字段
                ohlc = np.array([
                    price_data.get('open', []),
                    price_data.get('high', []),
                    price_data.get('low', []),
                    price_data.get('close', [])
                ]).T  # 转置为 [time, features]
                
                price_array = ohlc
            else:
                raise ValueError("不支持的价格数据格式")
            
            processed_data = {
                'price_data': price_array
            }
        else:
            raise ValueError("图表数据必须包含 'image_paths' 或 'price_data'")
        
        # 处理标签
        processed_labels = {}
        if labels:
            # 序列点位
            if 'sequence_points' in labels:
                sequence_points = np.array(labels['sequence_points'])
                processed_labels['sequence_points'] = sequence_points
            
            # 信号方向
            if 'direction' in labels:
                direction = labels['direction']
                direction_one_hot = np.zeros(2)
                if direction == 'long':
                    direction_one_hot[0] = 1
                elif direction == 'short':
                    direction_one_hot[1] = 1
                
                processed_labels['signal'] = direction_one_hot
            
            # 价格目标
            price_targets = []
            if 'entry_price' in labels:
                price_targets.append(labels['entry_price'])
            if 'stop_loss' in labels:
                price_targets.append(labels['stop_loss'])
            if 'target_price' in labels:
                price_targets.append(labels['target_price'])
            
            if price_targets:
                processed_labels['prices'] = np.array(price_targets)
        
        # 组合数据和标签
        training_data = {
            'data': processed_data,
            'labels': processed_labels
        }
        
        return training_data
    
    def save_training_data(self, training_data: Dict[str, Any], output_dir: str, filename: str = None) -> str:
        """保存训练数据
        
        Args:
            training_data: 训练数据
            output_dir: 输出目录
            filename: 文件名，如果为None则自动生成
            
        Returns:
            保存的文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.npz"
        
        file_path = output_path / filename
        
        # 保存为NumPy压缩文件
        np.savez_compressed(
            file_path,
            data=training_data['data'],
            labels=training_data['labels']
        )
        
        self.logger.info(f"训练数据已保存到 {file_path}")
        
        return str(file_path)
    
    def load_training_data(self, file_path: str) -> Dict[str, Any]:
        """加载训练数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            training_data: 训练数据
        """
        # 加载NumPy压缩文件
        loaded = np.load(file_path, allow_pickle=True)
        
        training_data = {
            'data': loaded['data'].item(),
            'labels': loaded['labels'].item()
        }
        
        return training_data
    
    def prepare_batch(self, batch_data: List[Dict[str, Any]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """准备批次数据
        
        Args:
            batch_data: 批次数据列表
            
        Returns:
            batch_x: 输入数据张量
            batch_y: 标签数据字典
        """
        # 提取数据和标签
        data_list = []
        labels_dict = {
            'sequence_points': [],
            'signal': [],
            'prices': []
        }
        
        for item in batch_data:
            # 数据
            if 'sequence' in item['data']:
                data_list.append(item['data']['sequence'])
            elif 'price_data' in item['data']:
                data_list.append(item['data']['price_data'])
            
            # 标签
            labels = item['labels']
            for key in labels_dict.keys():
                if key in labels:
                    labels_dict[key].append(labels[key])
                else:
                    labels_dict[key].append(None)
        
        # 转换为张量
        batch_x = torch.tensor(np.array(data_list), dtype=torch.float32)
        
        # 处理标签
        batch_y = {}
        for key, values in labels_dict.items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                batch_y[key] = torch.tensor(np.array(valid_values), dtype=torch.float32)
        
        return batch_x, batch_y
    
    def create_validation_set(self, data_dir: str, output_dir: str, validation_ratio: float = 0.2):
        """创建验证集
        
        从训练数据中分离出一部分作为验证集
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            validation_ratio: 验证集比例
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有训练数据文件
        data_files = list(data_path.glob('*.npz'))
        
        if not data_files:
            self.logger.warning(f"在 {data_dir} 中没有找到训练数据文件")
            return
        
        # 随机打乱文件列表
        np.random.shuffle(data_files)
        
        # 计算验证集大小
        val_size = int(len(data_files) * validation_ratio)
        
        # 分离验证集
        val_files = data_files[:val_size]
        
        # 复制文件到验证集目录
        for file_path in val_files:
            # 加载数据
            training_data = self.load_training_data(str(file_path))
            
            # 保存到验证集目录
            output_file = output_path / file_path.name
            self.save_training_data(training_data, str(output_path.parent), output_file.name)
        
        self.logger.info(f"创建了包含 {len(val_files)} 个样本的验证集")
    
    def generate_pseudo_labels(self, 
                              unlabeled_data_dir: str, 
                              model, 
                              output_dir: str, 
                              confidence_threshold: float = 0.8):
        """生成伪标签
        
        使用模型为无标签数据生成伪标签
        
        Args:
            unlabeled_data_dir: 无标签数据目录
            model: 预训练模型
            output_dir: 输出目录
            confidence_threshold: 置信度阈值
        """
        unlabeled_path = Path(unlabeled_data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有无标签数据文件
        data_files = list(unlabeled_path.glob('*.npz'))
        
        if not data_files:
            self.logger.warning(f"在 {unlabeled_data_dir} 中没有找到无标签数据文件")
            return
        
        # 设置模型为评估模式
        model.eval()
        device = next(model.parameters()).device
        
        # 处理每个文件
        for file_path in data_files:
            try:
                # 加载数据
                data = np.load(str(file_path), allow_pickle=True)
                
                # 准备输入数据
                input_data = torch.tensor(data['data'].item()['sequence'], dtype=torch.float32).unsqueeze(0).to(device)
                
                # 模型预测
                with torch.no_grad():
                    outputs = model(input_data)
                
                # 获取置信度
                confidence = torch.max(torch.softmax(outputs['sequence_points'], dim=1)).item()
                
                # 如果置信度高于阈值，生成伪标签
                if confidence >= confidence_threshold:
                    # 获取预测结果
                    sequence_points = torch.argmax(outputs['sequence_points'], dim=1).cpu().numpy()
                    
                    if 'signal' in outputs:
                        signal = torch.argmax(outputs['signal'], dim=1).cpu().numpy()
                        signal_label = 'long' if signal[0] == 0 else 'short'
                    else:
                        signal_label = None
                    
                    if 'prices' in outputs:
                        prices = outputs['prices'].cpu().numpy()[0]
                    else:
                        prices = None
                    
                    # 创建伪标签
                    pseudo_labels = {}
                    pseudo_labels['sequence_points'] = sequence_points.tolist()
                    
                    if signal_label:
                        pseudo_labels['direction'] = signal_label
                    
                    if prices is not None and len(prices) >= 3:
                        pseudo_labels['entry_price'] = float(prices[0])
                        pseudo_labels['stop_loss'] = float(prices[1])
                        pseudo_labels['target_price'] = float(prices[2])
                    
                    # 创建训练数据
                    training_data = {
                        'data': data['data'].item(),
                        'labels': pseudo_labels,
                        'confidence': confidence
                    }
                    
                    # 保存到输出目录
                    output_file = output_path / f"pseudo_{file_path.stem}.npz"
                    self.save_training_data(training_data, str(output_path), output_file.name)
                    
                    self.logger.info(f"为 {file_path.name} 生成了伪标签，置信度: {confidence:.4f}")
            
            except Exception as e:
                self.logger.warning(f"处理文件 {file_path} 失败: {e}")
        
        self.logger.info(f"伪标签生成完成") 