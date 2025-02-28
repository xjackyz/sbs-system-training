#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志工具模块
提供统一的日志配置和管理
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logger(name: str, 
                level: str = 'INFO',
                log_dir: str = 'logs',
                max_bytes: int = 10485760,  # 10MB
                backup_count: int = 5) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_dir: 日志目录
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的日志文件数量
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 如果已经有处理器，不重复添加
    if logger.handlers:
        return logger
        
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建文件处理器
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'{name}.log'),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
    
def get_logger(name: str) -> Optional[logging.Logger]:
    """
    获取已存在的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器或None
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    训练日志记录器，用于记录训练过程中的指标
    """
    
    def __init__(self, name, log_dir='logs'):
        """
        初始化训练日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志文件保存目录
        """
        self.logger = setup_logger(name, log_dir)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {},
            'lr': []
        }
    
    def log_epoch(self, epoch, train_loss, val_loss=None, metrics=None, lr=None):
        """
        记录每个训练轮次的信息
        
        Args:
            epoch: 当前轮次
            train_loss: 训练损失
            val_loss: 验证损失
            metrics: 评估指标字典
            lr: 学习率
        """
        # 记录训练损失
        self.history['train_loss'].append(train_loss)
        
        # 构建日志消息
        msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"
        
        # 记录验证损失
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
            msg += f", val_loss={val_loss:.4f}"
        
        # 记录评估指标
        if metrics:
            for name, value in metrics.items():
                if name not in self.history['metrics']:
                    self.history['metrics'][name] = []
                self.history['metrics'][name].append(value)
                msg += f", {name}={value:.4f}"
        
        # 记录学习率
        if lr is not None:
            self.history['lr'].append(lr)
            msg += f", lr={lr:.6f}"
        
        # 输出日志
        self.logger.info(msg)
    
    def log_batch(self, epoch, batch, total_batches, loss, lr=None):
        """
        记录每个批次的信息
        
        Args:
            epoch: 当前轮次
            batch: 当前批次
            total_batches: 总批次数
            loss: 批次损失
            lr: 学习率
        """
        # 构建日志消息
        msg = f"Epoch {epoch}, Batch {batch}/{total_batches}: loss={loss:.4f}"
        
        # 记录学习率
        if lr is not None:
            msg += f", lr={lr:.6f}"
        
        # 输出日志
        self.logger.debug(msg)
    
    def log_info(self, message):
        """
        记录一般信息
        
        Args:
            message: 日志消息
        """
        self.logger.info(message)
    
    def log_warning(self, message):
        """
        记录警告信息
        
        Args:
            message: 日志消息
        """
        self.logger.warning(message)
    
    def log_error(self, message):
        """
        记录错误信息
        
        Args:
            message: 日志消息
        """
        self.logger.error(message)
    
    def get_history(self):
        """
        获取训练历史记录
        
        Returns:
            history: 训练历史记录字典
        """
        return self.history