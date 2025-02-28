"""数据增强工具"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import random
from scipy.signal import resample
from scipy.interpolate import interp1d

class TimeSeriesAugmentation:
    """时间序列数据增强"""
    
    def __init__(self, 
                 window_size: int = 100,
                 noise_level: float = 0.01,
                 time_warp_factor: float = 0.2,
                 crop_ratio: float = 0.8):
        """初始化
        
        Args:
            window_size: 窗口大小
            noise_level: 噪声水平
            time_warp_factor: 时间扭曲因子
            crop_ratio: 裁剪比例
        """
        self.window_size = window_size
        self.noise_level = noise_level
        self.time_warp_factor = time_warp_factor
        self.crop_ratio = crop_ratio
        
    def add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声
        
        Args:
            x: 输入张量 [batch_size, sequence_length, features]
            
        Returns:
            添加噪声后的张量
        """
        noise = torch.randn_like(x) * self.noise_level
        return x + noise
        
    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """时间扭曲
        
        Args:
            x: 输入张量 [batch_size, sequence_length, features]
            
        Returns:
            时间扭曲后的张量
        """
        batch_size, seq_len, features = x.shape
        
        # 生成扭曲网格
        orig_steps = np.arange(seq_len)
        warp_steps = orig_steps + np.random.normal(
            loc=0, scale=self.time_warp_factor, size=(batch_size, seq_len)
        )
        warp_steps = np.sort(warp_steps, axis=1)
        
        # 应用扭曲
        warped = torch.zeros_like(x)
        for i in range(batch_size):
            for j in range(features):
                warped[i, :, j] = torch.tensor(
                    interp1d(orig_steps, x[i, :, j].cpu(),
                            kind='linear', bounds_error=False, fill_value='extrapolate')
                    (warp_steps[i])
                )
        return warped
        
    def random_crop(self, x: torch.Tensor) -> torch.Tensor:
        """随机裁剪
        
        Args:
            x: 输入张量 [batch_size, sequence_length, features]
            
        Returns:
            裁剪后的张量
        """
        batch_size, seq_len, features = x.shape
        crop_len = int(seq_len * self.crop_ratio)
        
        # 随机选择起始位置
        starts = torch.randint(0, seq_len - crop_len + 1, (batch_size,))
        cropped = torch.zeros((batch_size, crop_len, features))
        
        for i in range(batch_size):
            cropped[i] = x[i, starts[i]:starts[i]+crop_len, :]
            
        return cropped
        
    def random_scale(self, x: torch.Tensor, min_scale: float = 0.8, max_scale: float = 1.2) -> torch.Tensor:
        """随机缩放
        
        Args:
            x: 输入张量 [batch_size, sequence_length, features]
            min_scale: 最小缩放因子
            max_scale: 最大缩放因子
            
        Returns:
            缩放后的张量
        """
        scales = torch.rand(x.size(0), 1, 1) * (max_scale - min_scale) + min_scale
        return x * scales
        
    def mixup(self, x: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Mixup增强
        
        Args:
            x: 输入张量 [batch_size, sequence_length, features]
            alpha: beta分布参数
            
        Returns:
            混合后的张量，原始张量的混合权重
        """
        batch_size = x.size(0)
        
        # 生成混合权重
        lam = np.random.beta(alpha, alpha)
        
        # 随机打乱索引
        index = torch.randperm(batch_size)
        
        # 混合数据
        mixed_x = lam * x + (1 - lam) * x[index]
        
        return mixed_x, x[index], lam
        
    def cutmix(self, x: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """CutMix增强
        
        Args:
            x: 输入张量 [batch_size, sequence_length, features]
            alpha: beta分布参数
            
        Returns:
            混合后的张量，切片位置，混合比例
        """
        batch_size, seq_len, features = x.size()
        
        # 生成混合权重
        lam = np.random.beta(alpha, alpha)
        
        # 随机打乱索引
        index = torch.randperm(batch_size)
        
        # 计算切片位置
        cut_len = int(seq_len * lam)
        cut_start = torch.randint(0, seq_len - cut_len + 1, (batch_size,))
        
        # 创建混合数据
        mixed_x = x.clone()
        for i in range(batch_size):
            mixed_x[i, cut_start[i]:cut_start[i]+cut_len, :] = \
                x[index[i], cut_start[i]:cut_start[i]+cut_len, :]
        
        return mixed_x, x[index], lam
        
    def apply_augmentations(self, x: torch.Tensor, num_augments: int = 2) -> List[torch.Tensor]:
        """应用多个数据增强
        
        Args:
            x: 输入张量 [batch_size, sequence_length, features]
            num_augments: 增强次数
            
        Returns:
            增强后的张量列表
        """
        augmentations = [
            self.add_gaussian_noise,
            self.time_warp,
            self.random_crop,
            self.random_scale
        ]
        
        results = []
        for _ in range(num_augments):
            # 随机选择增强方法
            selected_augments = random.sample(augmentations, k=2)
            augmented = x
            
            # 依次应用选中的增强
            for aug in selected_augments:
                augmented = aug(augmented)
            
            results.append(augmented)
            
        return results


class ContrastiveLearning:
    """对比学习实现"""
    
    def __init__(self, 
                 temperature: float = 0.07,
                 base_momentum: float = 0.999,
                 queue_size: int = 65536):
        """初始化
        
        Args:
            temperature: 温度参数
            base_momentum: 动量编码器基础动量
            queue_size: 负样本队列大小
        """
        self.temperature = temperature
        self.base_momentum = base_momentum
        self.queue_size = queue_size
        self.current_momentum = base_momentum
        
    def info_nce_loss(self, 
                      query: torch.Tensor,
                      key: torch.Tensor,
                      queue: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算InfoNCE损失
        
        Args:
            query: 查询特征 [batch_size, feature_dim]
            key: 正样本特征 [batch_size, feature_dim]
            queue: 负样本队列 [queue_size, feature_dim]
            
        Returns:
            对比损失值
        """
        # L2标准化
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        
        # 计算正样本相似度
        pos_logit = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)
        
        # 如果有负样本队列
        if queue is not None:
            queue = F.normalize(queue, dim=1)
            neg_logits = torch.einsum('nc,kc->nk', [query, queue])
            logits = torch.cat([pos_logit, neg_logits], dim=1)
        else:
            # 使用batch内其他样本作为负样本
            neg_logits = torch.einsum('nc,kc->nk', [query, key])
            neg_logits = neg_logits - torch.eye(query.shape[0], device=query.device) * 1e9
            logits = torch.cat([pos_logit, neg_logits], dim=1)
        
        # 应用温度缩放
        logits = logits / self.temperature
        
        # 计算交叉熵损失
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
        
    def momentum_update(self, online_net: torch.nn.Module, target_net: torch.nn.Module):
        """动量更新目标网络
        
        Args:
            online_net: 在线网络
            target_net: 目标网络
        """
        for online_params, target_params in zip(online_net.parameters(), target_net.parameters()):
            target_params.data = target_params.data * self.current_momentum + \
                               online_params.data * (1. - self.current_momentum)
                               
    def adjust_momentum(self, epoch: int, total_epochs: int):
        """调整动量值
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
        """
        # 余弦退火调整动量
        self.current_momentum = 1. - (1. - self.base_momentum) * \
            (np.cos(np.pi * epoch / total_epochs) + 1) / 2 