#!/usr/bin/env python
"""
SBS训练器
统一的SBS序列模型训练器，支持标准训练、自监督学习、强化学习和主动学习。
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import time
import datetime
import json
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from ..utils.logger import setup_logger
from ..model.sbs_predictor import SBSPredictor
from ..utils.reward_calculator import SBSRewardCalculator
from ..utils.trade_tracker import TradeResultTracker
from ..utils.evaluator import SBSEvaluator
from ..data.data_loader import SBSDataLoader

logger = setup_logger('sbs_trainer')

class SBSTrainer:
    """
    SBS序列模型统一训练器，支持标准训练、自监督学习、强化学习和主动学习。
    """
    
    def __init__(self, config: Dict = None, mode: str = 'standard'):
        """
        初始化训练器。
        
        参数:
            config: 训练配置字典
            mode: 训练模式，可选值：standard, self_supervised, reinforcement, active_learning
        """
        self.config = config or {}
        self.mode = mode
        
        # 初始化组件
        self._init_logger()
        self._init_model()
        self._init_optimizer()
        self._init_reward_calculator()
        self._init_trade_tracker()
        self._init_evaluator()
        self._init_data_loader()
        self._init_callbacks()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.early_stop_counter = 0
        self.training_start_time = None
        
        # 学习率调度器
        self.scheduler = None
        if self.config.get('use_lr_scheduler', False):
            scheduler_type = self.config.get('scheduler_type', 'step')
            if scheduler_type == 'step':
                self.scheduler = StepLR(
                    self.optimizer, 
                    step_size=self.config.get('lr_step_size', 10),
                    gamma=self.config.get('lr_gamma', 0.1)
                )
            elif scheduler_type == 'plateau':
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=self.config.get('lr_factor', 0.1),
                    patience=self.config.get('lr_patience', 5),
                    verbose=True
                )
        
        # 加载检查点（如果有）
        checkpoint_path = self.config.get('resume_from')
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            
        # 检查系统一致性
        self._check_system_consistency()
        
    def _init_logger(self):
        """初始化日志系统"""
        log_level = self.config.get('log_level', 'INFO')
        self.logger = setup_logger('sbs_trainer', level=log_level)
        self.logger.info(f"初始化SBS训练器（模式: {self.mode}）")
        
    def _init_model(self):
        """初始化模型"""
        model_config = self.config.get('model', {})
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.get('use_gpu', True) else 'cpu')
        
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化预测器模型
        self.model = SBSPredictor(model_config)
        self.model.to(self.device)
        
        # 如果提供了检查点路径，则加载模型权重
        checkpoint_path = self.config.get('checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        optim_config = self.config.get('optimizer', {})
        lr = optim_config.get('learning_rate', 1e-4)
        weight_decay = optim_config.get('weight_decay', 1e-5)
        
        # 创建优化器
        optimizer_type = optim_config.get('type', 'adam')
        if optimizer_type.lower() == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            momentum = optim_config.get('momentum', 0.9)
            self.optimizer = SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        self.logger.info(f"优化器初始化完成: {optimizer_type}, 学习率: {lr}")
        
    def _init_reward_calculator(self):
        """初始化奖励计算器"""
        reward_config = self.config.get('reward', {})
        if self.config.get('use_reward_calculator', False):
            self.reward_calculator = SBSRewardCalculator(reward_config)
            self.logger.info("奖励计算器初始化完成")
        else:
            self.reward_calculator = None
            
    def _init_trade_tracker(self):
        """初始化交易跟踪器"""
        trade_config = self.config.get('trade', {})
        if self.config.get('use_trade_tracker', False):
            self.trade_tracker = TradeResultTracker(trade_config)
            self.logger.info("交易结果跟踪器初始化完成")
        else:
            self.trade_tracker = None
        
    def _init_evaluator(self):
        """初始化评估器"""
        eval_config = self.config.get('evaluation', {})
        self.evaluator = SBSEvaluator(eval_config)
        self.logger.info("评估器初始化完成")
        
    def _init_data_loader(self):
        """初始化数据加载器"""
        data_config = self.config.get('data', {})
        self.data_loader = SBSDataLoader(data_config)
        self.logger.info("数据加载器初始化完成")
        
    def _init_callbacks(self):
        """初始化回调函数"""
        self.callbacks = []
        callback_configs = self.config.get('callbacks', [])
        
        # 这里可以根据配置添加不同的回调函数
        # 如：TensorBoard、模型检查点保存、早停等
        
    def _load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_metric = checkpoint.get('best_metric', float('-inf'))
            
            self.logger.info(f"成功加载检查点: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            
    def save_checkpoint(self, path, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        save_path = path
        if is_best:
            save_path = os.path.join(os.path.dirname(path), 'best_model.pth')
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        self.logger.info(f"保存检查点到: {save_path}")
        
    def train(self, train_data=None, val_data=None):
        """
        标准训练模式
        
        参数:
            train_data: 训练数据，如果未提供则使用配置中的数据路径
            val_data: 验证数据，如果未提供则使用配置中的数据路径
        """
        self.logger.info("开始标准训练...")
        self.training_start_time = time.time()
        
        # 训练参数
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        val_interval = self.config.get('val_interval', 1)
        save_interval = self.config.get('save_interval', 5)
        early_stop = self.config.get('early_stop', 10)
        num_workers = self.config.get('num_workers', 4)
        shuffle = self.config.get('shuffle', True)
        
        # 如果未提供数据，使用配置中的数据路径创建数据加载器
        if train_data is None or val_data is None:
            train_path = self.config.get('train_data_path')
            val_path = self.config.get('val_data_path')
            
            if not train_path:
                raise ValueError("训练数据路径未提供")
                
            # 创建数据加载器
            train_loader, val_loader, _ = self.data_loader.get_standard_dataloaders(
                train_path=train_path,
                val_path=val_path,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            )
        else:
            # 直接使用提供的数据
            train_loader = train_data
            val_loader = val_data
        
        # 训练循环
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_metrics = self._train_epoch(train_loader)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                
            # 验证
            val_metrics = {}
            if val_loader and epoch % val_interval == 0:
                val_metrics = self._validate(val_loader)
                
                # 检查是否是最佳模型
                current_metric = val_metrics.get('f1', 0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.save_checkpoint(
                        os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), f'epoch_{epoch}.pth'),
                        is_best=True
                    )
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    
            # 保存检查点
            if epoch % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), f'epoch_{epoch}.pth')
                )
                
            # 记录训练信息
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"训练损失: {train_metrics.get('loss', 0):.4f}, "
                f"验证F1: {val_metrics.get('f1', 0):.4f}, "
                f"耗时: {epoch_time:.2f}秒"
            )
            
            # 早停检查
            if early_stop and self.early_stop_counter >= early_stop:
                self.logger.info(f"触发早停: {early_stop}个epoch没有改善")
                break
                
        # 训练结束
        total_time = time.time() - self.training_start_time
        self.logger.info(f"训练完成，总耗时: {datetime.timedelta(seconds=int(total_time))}")
        
        return {'best_metric': self.best_metric}
        
    def _train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据移动到指定设备
            data, target = data.to(self.device), target.to(self.device)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            
            # 计算损失
            loss = self.model.calculate_loss(output, target)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            batch_count += 1
            
            # 更新全局步数
            self.global_step += 1
            
            # 记录训练进度
            if batch_idx % 10 == 0:
                self.logger.debug(
                    f"训练批次: {batch_idx}/{len(train_loader)} "
                    f"损失: {loss.item():.6f}"
                )
        
        metrics = {
            'loss': total_loss / max(1, batch_count)
        }
        
        return metrics
        
    def _validate(self, val_loader):
        """在验证集上评估模型"""
        # 使用评估器评估模型
        metrics = self.evaluator.evaluate_model(
            model=self.model,
            dataloader=val_loader,
            device=self.device
        )
        
        return metrics
        
    def train_self_supervised(self, labeled_data=None, unlabeled_data=None):
        """
        自监督训练模式
        
        参数:
            labeled_data: 已标记数据，如果未提供则使用配置中的数据路径
            unlabeled_data: 未标记数据，如果未提供则使用配置中的数据路径
        """
        self.logger.info("开始自监督训练...")
        self.training_start_time = time.time()
        
        # 训练参数
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        val_interval = self.config.get('val_interval', 1)
        save_interval = self.config.get('save_interval', 5)
        early_stop = self.config.get('early_stop', 10)
        num_workers = self.config.get('num_workers', 4)
        shuffle = self.config.get('shuffle', True)
        supervised_ratio = self.config.get('supervised_ratio', 0.3)  # 监督学习与自监督学习的比例
        
        # 如果未提供数据，使用配置中的数据路径创建数据加载器
        if labeled_data is None or unlabeled_data is None:
            labeled_path = self.config.get('labeled_data_path')
            unlabeled_path = self.config.get('unlabeled_data_path')
            val_path = self.config.get('val_data_path')
            
            if not labeled_path or not unlabeled_path:
                raise ValueError("标记数据或未标记数据路径未提供")
                
            # 创建数据加载器
            labeled_loader, unlabeled_loader, val_loader = self.data_loader.get_self_supervised_dataloaders(
                labeled_path=labeled_path,
                unlabeled_path=unlabeled_path,
                val_path=val_path,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            )
        else:
            # 直接使用提供的数据
            labeled_loader = labeled_data
            unlabeled_loader = unlabeled_data
            val_loader = None
        
        # 训练循环
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_metrics = self._train_self_supervised_epoch(labeled_loader, unlabeled_loader, supervised_ratio)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                
            # 验证
            val_metrics = {}
            if val_loader and epoch % val_interval == 0:
                val_metrics = self._validate(val_loader)
                
                # 检查是否是最佳模型
                current_metric = val_metrics.get('f1', 0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.save_checkpoint(
                        os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), f'epoch_{epoch}.pth'),
                        is_best=True
                    )
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    
            # 保存检查点
            if epoch % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), f'epoch_{epoch}.pth')
                )
                
            # 记录训练信息
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"监督损失: {train_metrics.get('supervised_loss', 0):.4f}, "
                f"自监督损失: {train_metrics.get('self_supervised_loss', 0):.4f}, "
                f"验证F1: {val_metrics.get('f1', 0):.4f}, "
                f"耗时: {epoch_time:.2f}秒"
            )
            
            # 早停检查
            if early_stop and self.early_stop_counter >= early_stop:
                self.logger.info(f"触发早停: {early_stop}个epoch没有改善")
                break
                
        # 训练结束
        total_time = time.time() - self.training_start_time
        self.logger.info(f"训练完成，总耗时: {datetime.timedelta(seconds=int(total_time))}")
        
        return {'best_metric': self.best_metric}
        
    def _train_self_supervised_epoch(self, labeled_loader, unlabeled_loader, supervised_ratio=0.3):
        """训练一个自监督epoch"""
        self.model.train()
        total_supervised_loss = 0
        total_self_supervised_loss = 0
        batch_count = 0
        
        # 创建无限迭代器，以便在不同大小的数据集中循环
        unlabeled_iter = iter(unlabeled_loader)
        
        for batch_idx, (labeled_data, labeled_target) in enumerate(labeled_loader):
            # 将有标签数据移动到设备
            labeled_data, labeled_target = labeled_data.to(self.device), labeled_target.to(self.device)
            
            # 获取无标签数据
            try:
                unlabeled_data, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_data, _ = next(unlabeled_iter)
                
            unlabeled_data = unlabeled_data.to(self.device)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向传播 - 有标签数据
            supervised_output = self.model(labeled_data)
            supervised_loss = self.model.calculate_loss(supervised_output, labeled_target)
            
            # 前向传播 - 无标签数据
            self_supervised_output = self.model(unlabeled_data)
            self_supervised_loss = self.model.calculate_self_supervised_loss(self_supervised_output)
            
            # 组合损失
            combined_loss = supervised_ratio * supervised_loss + (1 - supervised_ratio) * self_supervised_loss
            
            # 反向传播
            combined_loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_supervised_loss += supervised_loss.item()
            total_self_supervised_loss += self_supervised_loss.item()
            batch_count += 1
            
            # 更新全局步数
            self.global_step += 1
            
            # 记录训练进度
            if batch_idx % 10 == 0:
                self.logger.debug(
                    f"训练批次: {batch_idx}/{len(labeled_loader)} "
                    f"监督损失: {supervised_loss.item():.6f}, "
                    f"自监督损失: {self_supervised_loss.item():.6f}"
                )
        
        metrics = {
            'supervised_loss': total_supervised_loss / max(1, batch_count),
            'self_supervised_loss': total_self_supervised_loss / max(1, batch_count)
        }
        
        return metrics
        
    def train_rl(self, market_data=None, val_data=None):
        """
        强化学习训练模式
        
        参数:
            market_data: 市场数据，如果未提供则使用配置中的数据路径
            val_data: 验证数据，如果未提供则使用配置中的数据路径
        """
        self.logger.info("开始强化学习训练...")
        self.training_start_time = time.time()
        
        # 训练参数
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        val_interval = self.config.get('val_interval', 1)
        save_interval = self.config.get('save_interval', 5)
        early_stop = self.config.get('early_stop', 10)
        num_workers = self.config.get('num_workers', 4)
        shuffle = self.config.get('shuffle', True)
        
        # 如果未提供市场数据，使用配置中的数据路径创建数据加载器
        if market_data is None:
            market_data_path = self.config.get('market_data_path')
            val_path = self.config.get('val_data_path')
            
            if not market_data_path:
                raise ValueError("市场数据路径未提供")
                
            # 创建数据加载器
            train_loader = self.data_loader.get_rl_dataloader(
                market_path=market_data_path,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            )
            
            val_loader = None
            if val_path:
                val_loader = self.data_loader.get_standard_dataloaders(
                    train_path=val_path,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers
                )[0]  # 获取第一个返回值（训练加载器）
        else:
            # 直接使用提供的数据
            train_loader = market_data
            val_loader = val_data
        
        # 确保奖励计算器已初始化
        if self.reward_calculator is None:
            self._init_reward_calculator()
            
        # 确保交易结果跟踪器已初始化
        if self.trade_tracker is None:
            self._init_trade_tracker()
            
        # 配置探索机制
        exploration_config = self.config.get('exploration', {})
        if exploration_config.get('enabled', False) and hasattr(self.trade_tracker, 'enable_exploration'):
            self.trade_tracker.enable_exploration(
                enabled=True,
                rate=exploration_config.get('rate', 0.1)
            )
            self.logger.info(f"已启用交易探索机制，初始探索率: {self.trade_tracker.exploration_rate:.4f}")
        
        # 训练循环
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_metrics = self._train_rl_epoch(train_loader)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # 如果是ReduceLROnPlateau，需要提供指标值
                    self.scheduler.step(train_metrics.get('avg_reward', 0))
                else:
                    self.scheduler.step()
                    
            # 验证
            val_metrics = {}
            if val_loader and epoch % val_interval == 0:
                val_metrics = self._validate(val_loader)
                
                # 评估交易结果
                trade_metrics = self.evaluator.evaluate_trades(self.trade_tracker)
                val_metrics.update(trade_metrics)
                
                # 动态调整奖励计算器参数
                if hasattr(self.reward_calculator, 'adaptive_update'):
                    self.reward_calculator.adaptive_update(trade_metrics)
                    self.logger.info("已根据交易表现动态调整奖励计算参数")
                
                # 检查是否是最佳模型
                current_metric = trade_metrics.get('profit_factor', 0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.save_checkpoint(
                        os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), f'epoch_{epoch}.pth'),
                        is_best=True
                    )
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    
            # 保存检查点
            if epoch % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), f'epoch_{epoch}.pth')
                )
                
                # 保存交易历史和探索指标
                trade_history_path = os.path.join(self.config.get('log_dir', 'logs'), f'trade_history_epoch_{epoch}.json')
                if hasattr(self.reward_calculator, 'save_trade_history'):
                    self.reward_calculator.save_trade_history(trade_history_path)
                
                # 输出交易详细报告
                if hasattr(self.trade_tracker, 'generate_detailed_report'):
                    report_path = self.trade_tracker.generate_detailed_report(
                        filepath=os.path.join(self.config.get('log_dir', 'logs'), f'trade_report_epoch_{epoch}.txt')
                    )
                    self.logger.info(f"已生成交易详细报告: {report_path}")
                
                # 导出LabelStudio预标注数据
                if epoch > 0 and epoch % (save_interval * 2) == 0 and hasattr(self.trade_tracker, 'export_for_labelstudio'):
                    label_studio_path = self.trade_tracker.export_for_labelstudio(
                        filepath=os.path.join(self.config.get('log_dir', 'logs'), f'labelstudio_export_epoch_{epoch}.json')
                    )
                    if label_studio_path:
                        self.logger.info(f"已导出LabelStudio预标注数据: {label_studio_path}")
                
            # 更新探索率
            if hasattr(self.trade_tracker, 'update_exploration_rate') and epoch % 2 == 0:
                old_rate = getattr(self.trade_tracker, 'exploration_rate', 0)
                self.trade_tracker.update_exploration_rate(decay=True)
                new_rate = getattr(self.trade_tracker, 'exploration_rate', 0)
                self.logger.info(f"已更新探索率: {old_rate:.4f} -> {new_rate:.4f}")
                
            # 记录训练信息
            epoch_time = time.time() - epoch_start_time
            
            # 获取探索指标
            exploration_metrics = {}
            if hasattr(self.trade_tracker, 'get_exploration_metrics'):
                exploration_metrics = self.trade_tracker.get_exploration_metrics()
                
            # 构建日志消息
            log_msg = (
                f"Epoch {epoch}/{epochs} - "
                f"RL损失: {train_metrics.get('rl_loss', 0):.4f}, "
                f"平均奖励: {train_metrics.get('avg_reward', 0):.4f}, "
                f"总交易: {val_metrics.get('total_trades', 0)}, "
                f"胜率: {val_metrics.get('win_rate', 0):.2f}, "
                f"利润因子: {val_metrics.get('profit_factor', 0):.2f}, "
                f"最大回撤: {val_metrics.get('max_drawdown', 0):.2f}, "
            )
            
            # 添加探索指标到日志
            if exploration_metrics:
                log_msg += (
                    f"探索率: {exploration_metrics.get('exploration_rate', 0):.4f}, "
                    f"探索交易: {exploration_metrics.get('total_explorations', 0)}, "
                    f"探索成功率: {exploration_metrics.get('success_rate', 0):.2f}, "
                )
                
            log_msg += f"耗时: {epoch_time:.2f}秒"
            self.logger.info(log_msg)
            
            # 早停检查
            if early_stop and self.early_stop_counter >= early_stop:
                self.logger.info(f"触发早停: {early_stop}个epoch没有改善")
                break
                
        # 训练结束
        total_time = time.time() - self.training_start_time
        self.logger.info(f"训练完成，总耗时: {datetime.timedelta(seconds=int(total_time))}")
        
        # 保存最终交易历史
        if hasattr(self.reward_calculator, 'save_trade_history'):
            self.reward_calculator.save_trade_history(
                os.path.join(self.config.get('log_dir', 'logs'), 'trade_history_final.json')
            )
        
        # 导出最终交易详细报告
        if hasattr(self.trade_tracker, 'generate_detailed_report'):
            final_report_path = self.trade_tracker.generate_detailed_report(
                filepath=os.path.join(self.config.get('log_dir', 'logs'), 'trade_report_final.txt'),
                include_trade_details=True
            )
            self.logger.info(f"已生成最终交易详细报告: {final_report_path}")
            
        # 导出最终LabelStudio预标注数据
        if hasattr(self.trade_tracker, 'export_for_labelstudio'):
            final_ls_path = self.trade_tracker.export_for_labelstudio(
                filepath=os.path.join(self.config.get('log_dir', 'logs'), 'labelstudio_export_final.json')
            )
            if final_ls_path:
                self.logger.info(f"已导出最终LabelStudio预标注数据: {final_ls_path}")
        
        return {'best_metric': self.best_metric}
        
    def _train_rl_epoch(self, train_loader):
        """训练一个强化学习epoch"""
        self.model.train()
        total_loss = 0
        total_reward = 0
        batch_count = 0
        
        for batch_idx, (states, actions, next_states, rewards, dones, market_data) in enumerate(train_loader):
            # 将数据移动到设备
            states = states.to(self.device)
            actions = actions.to(self.device) if actions is not None else None
            next_states = next_states.to(self.device) if next_states is not None else None
            rewards = rewards.to(self.device) if rewards is not None else None
            dones = dones.to(self.device) if dones is not None else None
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向传播 - 获取模型输出
            outputs = self.model(states)
            
            # 进行交易决策
            predictions = self.model.get_predictions(outputs)
            
            # 计算奖励
            if rewards is None:
                rewards = torch.zeros(states.size(0), device=self.device)
                for i in range(states.size(0)):
                    state = states[i].cpu().numpy()
                    prediction = predictions[i].cpu().numpy()
                    market_info = market_data[i] if market_data is not None else None
                    
                    # 使用奖励计算器计算奖励
                    reward = self.reward_calculator.calculate(
                        prediction=prediction,
                        ground_truth=None,  # 在RL中可能没有真实标签
                        market_data=market_info,
                        trade_result=None
                    )
                    
                    # 记录交易结果
                    if self.trade_tracker is not None:
                        self.trade_tracker.add_trade(
                            prediction=prediction,
                            ground_truth=None,
                            profit=reward,
                            market_data=market_info
                        )
                    
                    rewards[i] = torch.tensor(reward, device=self.device)
            
            # 计算RL损失
            loss = self.model.calculate_rl_loss(outputs, actions, rewards, next_states, dones)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失和奖励
            total_loss += loss.item()
            total_reward += rewards.mean().item()
            batch_count += 1
            
            # 更新全局步数
            self.global_step += 1
            
            # 记录训练进度
            if batch_idx % 10 == 0:
                self.logger.debug(
                    f"训练批次: {batch_idx}/{len(train_loader)} "
                    f"RL损失: {loss.item():.6f}, "
                    f"平均奖励: {rewards.mean().item():.6f}"
                )
        
        metrics = {
            'rl_loss': total_loss / max(1, batch_count),
            'avg_reward': total_reward / max(1, batch_count)
        }
        
        return metrics
        
    def train_active_learning(self, labeled_data=None, unlabeled_data=None, val_data=None):
        """
        主动学习训练模式
        
        参数:
            labeled_data: 已标记数据，如果未提供则使用配置中的数据路径
            unlabeled_data: 未标记数据，如果未提供则使用配置中的数据路径
            val_data: 验证数据，如果未提供则使用配置中的数据路径
        """
        self.logger.info("开始主动学习训练...")
        self.training_start_time = time.time()
        
        # 训练参数
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        val_interval = self.config.get('val_interval', 1)
        save_interval = self.config.get('save_interval', 5)
        early_stop = self.config.get('early_stop', 10)
        num_workers = self.config.get('num_workers', 4)
        shuffle = self.config.get('shuffle', True)
        acquisition_size = self.config.get('acquisition_size', 10)  # 每轮获取的样本数
        acquisition_interval = self.config.get('acquisition_interval', 5)  # 每多少个epoch获取一次新样本
        
        # 如果未提供数据，使用配置中的数据路径创建数据加载器
        if labeled_data is None or unlabeled_data is None:
            labeled_path = self.config.get('labeled_data_path')
            unlabeled_path = self.config.get('unlabeled_data_path')
            val_path = self.config.get('val_data_path')
            
            if not labeled_path or not unlabeled_path:
                raise ValueError("标记数据或未标记数据路径未提供")
                
            # 创建数据加载器
            active_learning_dataset = self.data_loader.get_active_learning_dataset(
                labeled_path=labeled_path,
                unlabeled_path=unlabeled_path,
                val_path=val_path
            )
            
            train_loader = DataLoader(
                active_learning_dataset.get_labeled_dataset(),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            )
            
            unlabeled_loader = DataLoader(
                active_learning_dataset.get_unlabeled_dataset(),
                batch_size=batch_size,
                shuffle=False,  # 不打乱未标记数据，以便准确追踪
                num_workers=num_workers
            )
            
            val_loader = DataLoader(
                active_learning_dataset.get_validation_dataset(),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            ) if active_learning_dataset.has_validation_data() else None
        else:
            # 直接使用提供的数据
            train_loader = labeled_data
            unlabeled_loader = unlabeled_data
            val_loader = val_data
            
            # 构建主动学习数据集用于管理标记/未标记数据
            active_learning_dataset = self.data_loader.create_active_learning_dataset_from_loaders(
                labeled_loader=train_loader,
                unlabeled_loader=unlabeled_loader,
                val_loader=val_loader
            )
        
        # 训练循环
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_metrics = self._train_epoch(train_loader)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                
            # 验证
            val_metrics = {}
            if val_loader and epoch % val_interval == 0:
                val_metrics = self._validate(val_loader)
                
                # 检查是否是最佳模型
                current_metric = val_metrics.get('f1', 0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.save_checkpoint(
                        os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), f'epoch_{epoch}.pth'),
                        is_best=True
                    )
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
            
            # 主动学习: 获取新样本
            if epoch % acquisition_interval == 0 and unlabeled_loader is not None:
                self._acquire_samples(active_learning_dataset, unlabeled_loader, acquisition_size)
                
                # 更新训练数据加载器
                train_loader = DataLoader(
                    active_learning_dataset.get_labeled_dataset(),
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers
                )
                
                # 更新未标记数据加载器
                unlabeled_loader = DataLoader(
                    active_learning_dataset.get_unlabeled_dataset(),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers
                )
                
            # 保存检查点
            if epoch % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), f'epoch_{epoch}.pth')
                )
                
            # 记录训练信息
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"训练损失: {train_metrics.get('loss', 0):.4f}, "
                f"验证F1: {val_metrics.get('f1', 0):.4f}, "
                f"已标记样本: {active_learning_dataset.get_labeled_count()}, "
                f"未标记样本: {active_learning_dataset.get_unlabeled_count()}, "
                f"耗时: {epoch_time:.2f}秒"
            )
            
            # 早停检查
            if early_stop and self.early_stop_counter >= early_stop:
                self.logger.info(f"触发早停: {early_stop}个epoch没有改善")
                break
                
        # 训练结束
        total_time = time.time() - self.training_start_time
        self.logger.info(f"训练完成，总耗时: {datetime.timedelta(seconds=int(total_time))}")
        
        return {'best_metric': self.best_metric}
        
    def _acquire_samples(self, active_learning_dataset, unlabeled_loader, acquisition_size):
        """
        从未标记数据中获取最有价值的样本进行标记
        
        参数:
            active_learning_dataset: 主动学习数据集
            unlabeled_loader: 未标记数据加载器
            acquisition_size: 获取的样本数量
        """
        self.model.eval()
        uncertainties = []
        indices = []
        
        # 计算每个未标记样本的不确定性
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                # 将数据移动到设备
                data = data.to(self.device)
                
                # 前向传播
                outputs = self.model(data)
                
                # 计算不确定性（例如，预测概率的方差或熵）
                uncertainty = self.model.calculate_uncertainty(outputs)
                
                # 记录不确定性和对应索引
                for i, u in enumerate(uncertainty):
                    uncertainties.append(u.item())
                    indices.append(batch_idx * unlabeled_loader.batch_size + i)
        
        # 根据不确定性选择样本（选择不确定性最高的样本）
        if not uncertainties:
            self.logger.warning("没有未标记的样本可获取")
            return
            
        # 将不确定性和索引组合，并按不确定性降序排序
        combined = list(zip(uncertainties, indices))
        combined.sort(reverse=True)
        
        # 选择不确定性最高的样本
        selected_indices = [idx for _, idx in combined[:acquisition_size]]
        
        # 从未标记数据中获取这些样本并添加到已标记数据中
        self.logger.info(f"从未标记数据中获取 {len(selected_indices)} 个样本")
        
        # 更新主动学习数据集
        active_learning_dataset.move_samples_to_labeled(selected_indices)
        
        # 记录当前已标记和未标记的样本数量
        self.logger.info(
            f"已标记样本: {active_learning_dataset.get_labeled_count()}, "
            f"未标记样本: {active_learning_dataset.get_unlabeled_count()}"
        )
        
    def start_training(self, mode='standard', **kwargs):
        """
        开始训练的入口方法
        
        参数:
            mode: 训练模式，可选值为 standard, self_supervised, rl, active_learning
            **kwargs: 传递给具体训练方法的参数
        
        返回:
            训练结果字典
        """
        # 创建必要的目录
        os.makedirs(self.config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
        os.makedirs(self.config.get('log_dir', 'logs'), exist_ok=True)
        
        # 根据模式选择训练方法
        if mode == 'standard':
            return self.train(**kwargs)
        elif mode == 'self_supervised':
            return self.train_self_supervised(**kwargs)
        elif mode == 'rl':
            return self.train_rl(**kwargs)
        elif mode == 'active_learning':
            return self.train_active_learning(**kwargs)
        else:
            raise ValueError(f"不支持的训练模式: {mode}")
            
    def predict(self, data):
        """
        使用训练好的模型进行预测
        
        参数:
            data: 输入数据
            
        返回:
            预测结果
        """
        self.model.eval()
        with torch.no_grad():
            # 确保数据在正确的设备上
            if isinstance(data, torch.Tensor):
                data = data.to(self.device)
            else:
                data = torch.tensor(data, dtype=torch.float32, device=self.device)
                
            # 处理维度
            if len(data.shape) == 1:
                data = data.unsqueeze(0)  # 添加批次维度
                
            # 前向传播
            outputs = self.model(data)
            predictions = self.model.get_predictions(outputs)
            
            return predictions.cpu().numpy()

    def _check_system_consistency(self):
        """检查系统配置的一致性，确保各组件能够协同工作"""
        self.logger.info("开始检查系统一致性...")
        
        # 1. 检查训练模式与必要组件的匹配
        if self.mode == 'reinforcement':
            if self.reward_calculator is None:
                self.logger.warning("强化学习模式需要奖励计算器，但未找到。正在初始化默认奖励计算器。")
                self._init_reward_calculator()
                
            if self.trade_tracker is None:
                self.logger.warning("强化学习模式需要交易跟踪器，但未找到。正在初始化默认交易跟踪器。")
                self._init_trade_tracker()
                
        # 2. 检查探索机制配置
        if self.config.get('exploration', {}).get('enabled', False):
            if self.mode != 'reinforcement':
                self.logger.warning(f"探索机制在{self.mode}模式下可能无效，建议仅在强化学习模式下使用。")
                
            if self.trade_tracker and not hasattr(self.trade_tracker, 'enable_exploration'):
                self.logger.warning("交易跟踪器不支持探索机制，相关配置将被忽略。")
                
        # 3. 检查模型配置
        if self.mode == 'self_supervised' and not self.model.supports_self_supervised():
            self.logger.warning("模型不支持自监督学习，可能导致训练失败。")
            
        if self.mode == 'reinforcement' and not self.model.supports_rl():
            self.logger.warning("模型不支持强化学习，可能导致训练失败。")
            
        # 4. 检查数据配置
        if self.mode == 'standard' and not self.config.get('train_data_path'):
            self.logger.warning("标准训练模式需要训练数据路径，但未找到。")
            
        if self.mode == 'self_supervised' and not self.config.get('unlabeled_data_path'):
            self.logger.warning("自监督训练模式需要无标签数据路径，但未找到。")
            
        if self.mode == 'reinforcement' and not self.config.get('market_data_path'):
            self.logger.warning("强化学习训练模式需要市场数据路径，但未找到。")
            
        if self.mode == 'active_learning' and not (self.config.get('labeled_data_path') and self.config.get('unlabeled_data_path')):
            self.logger.warning("主动学习训练模式需要有标签和无标签数据路径，但未找到。")
            
        # 5. 检查模型输出与奖励机制的兼容性
        if self.mode == 'reinforcement' and self.reward_calculator and self.model:
            output_shape = self.model.get_output_shape()
            if not self.reward_calculator.is_compatible_with_output(output_shape):
                self.logger.warning(f"模型输出形状{output_shape}与奖励计算器不兼容，可能导致计算错误。")
                
        # 6. 检查LabelStudio导出功能
        if self.config.get('tracker_config', {}).get('label_studio', {}).get('enabled', False):
            if not hasattr(self.trade_tracker, 'export_for_labelstudio'):
                self.logger.warning("交易跟踪器不支持LabelStudio导出功能，相关配置将被忽略。")
        
        self.logger.info("系统一致性检查完成") 