"""
基础模型模块
定义所有模型的基类
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from ..utils.logger import setup_logger
from ..utils.exceptions import ModelError

logger = setup_logger('base_model')

class BaseModel(nn.Module, ABC):
    """模型基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        前向传播
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            模型输出
        """
        raise NotImplementedError
        
    def save(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        try:
            state_dict = {
                'model_state': self.state_dict(),
                'config': self.config
            }
            torch.save(state_dict, filepath)
            logger.info(f'模型已保存到: {filepath}')
        except Exception as e:
            logger.error(f'保存模型失败: {str(e)}')
            raise ModelError(f'保存模型失败: {str(e)}')
            
    def load(self, filepath: str) -> None:
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        try:
            state_dict = torch.load(filepath, map_location=self.device)
            self.load_state_dict(state_dict['model_state'])
            self.config.update(state_dict['config'])
            logger.info(f'模型已从 {filepath} 加载')
        except Exception as e:
            logger.error(f'加载模型失败: {str(e)}')
            raise ModelError(f'加载模型失败: {str(e)}')
            
    def to_device(self, device: Optional[torch.device] = None) -> None:
        """
        将模型移动到指定设备
        
        Args:
            device: 目标设备，如果为None则使用默认设备
        """
        try:
            target_device = device or self.device
            self.to(target_device)
            self.device = target_device
            logger.info(f'模型已移动到设备: {target_device}')
        except Exception as e:
            logger.error(f'移动模型到设备失败: {str(e)}')
            raise ModelError(f'移动模型到设备失败: {str(e)}')
            
    def count_parameters(self) -> int:
        """
        统计模型参数数量
        
        Returns:
            参数总数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def freeze(self) -> None:
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
        logger.info('所有模型参数已冻结')
        
    def unfreeze(self) -> None:
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info('所有模型参数已解冻')
        
    def get_parameter_groups(self) -> Dict[str, Any]:
        """
        获取参数组，用于优化器
        
        Returns:
            参数组字典
        """
        return [
            {
                'params': [p for n, p in self.named_parameters() if p.requires_grad],
                'weight_decay': self.config.get('weight_decay', 0.01)
            }
        ]
        
    @property
    def num_parameters(self) -> int:
        """
        获取模型参数数量
        
        Returns:
            参数总数
        """
        return self.count_parameters()
        
    def __str__(self) -> str:
        """
        返回模型描述
        
        Returns:
            模型描述字符串
        """
        return f"{self.__class__.__name__}(num_parameters={self.num_parameters:,})" 