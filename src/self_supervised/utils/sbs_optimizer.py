#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SBS优化器
用于SBS模型的超参数自动优化
基于Optuna框架实现自动化超参数搜索
"""

import os
import logging
import yaml
import json
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import optuna
from optuna.trial import Trial
from optuna.integration import PyTorchLightningPruningCallback

from ..model.sbs_predictor import SBSPredictor

# 配置日志
logger = logging.getLogger('sbs_optimizer')

class SBSOptimizer:
    """
    SBS模型超参数优化器
    基于Optuna框架进行超参数搜索和优化
    支持自动批量大小调整、容错处理和结果持久化
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化优化器
        
        参数:
            config: 配置字典，包含优化设置
        """
        self.config = config or {}
        self.logger = logging.getLogger('sbs_optimizer')
        self.best_params = None
        self.study = None
        self.trials_history = []
        
        # 确保输出目录存在
        self.output_dir = Path(self.config.get('paths', {}).get('output_dir', 'outputs/optimization'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志设置
        self._setup_logging()
        
        # 记录初始化信息
        self.logger.info(f"SBSOptimizer初始化完成，配置: {self.config.get('optuna', {})}")
        
    def _setup_logging(self):
        """设置日志记录"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # 如果已经有处理器，不再添加
        if not self.logger.handlers:
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(numeric_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # 创建文件处理器
            log_dir = Path(self.config.get('logging', {}).get('log_dir', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self.logger.setLevel(numeric_level)
        
    def optimize(self, train_data: DataLoader, val_data: DataLoader) -> Dict[str, Any]:
        """
        执行超参数优化
        
        参数:
            train_data: 训练数据加载器
            val_data: 验证数据加载器
            
        返回:
            最佳参数字典
        """
        self.logger.info("开始超参数优化...")
        
        # 定义目标函数
        def objective(trial: Trial) -> float:
            # 记录试验开始
            trial_start_time = time.time()
            self.logger.info(f"开始试验 #{trial.number}")
            
            try:
                # 生成超参数
                params = self._generate_params(trial)
                
                # 记录当前试验参数
                self.logger.info(f"试验 #{trial.number} 参数: {params}")
                
                # 更新数据加载器以适应不同批次大小
                train_loader = self._adjust_dataloader(train_data, params['batch_size'])
                val_loader = self._adjust_dataloader(val_data, params['batch_size'])
                
                # 创建模型
                model = self._create_model(params)
                
                # 创建训练器
                trainer = self._create_trainer(trial, params)
                
                # 训练模型
                trainer.fit(model, train_loader, val_loader)
                
                # 获取验证指标
                metric_name = self.config.get('optuna', {}).get('metric', 'val_acc')
                if metric_name in trainer.callback_metrics:
                    metric_value = trainer.callback_metrics[metric_name].item()
                else:
                    # 如果没有找到指定指标，尝试使用val_loss
                    self.logger.warning(f"未找到指标 {metric_name}，尝试使用val_loss")
                    metric_value = trainer.callback_metrics.get('val_loss', torch.tensor(float('inf'))).item()
                    # 如果是最小化指标但方向是最大化，或反之，则取负值
                    if (self.config.get('optuna', {}).get('direction', 'maximize') == 'maximize' and
                        metric_name == 'val_loss'):
                        metric_value = -metric_value
                
                # 记录试验结果
                trial_duration = time.time() - trial_start_time
                self.logger.info(f"试验 #{trial.number} 完成，{metric_name}: {metric_value:.4f}，"
                               f"耗时: {trial_duration:.2f}秒")
                
                # 保存试验历史
                self.trials_history.append({
                    'trial_number': trial.number,
                    'params': params,
                    'metric': metric_value,
                    'duration': trial_duration
                })
                
                # 每5个试验保存一次历史记录
                if len(self.trials_history) % 5 == 0:
                    self._save_trials_history()
                
                return metric_value
            
            except Exception as e:
                self.logger.error(f"试验 #{trial.number} 失败: {str(e)}", exc_info=True)
                
                # 记录失败的试验
                if hasattr(trial, 'params'):
                    self.trials_history.append({
                        'trial_number': trial.number,
                        'params': trial.params,
                        'error': str(e),
                        'status': 'FAILED'
                    })
                
                # 返回一个表示失败的值
                if self.config.get('optuna', {}).get('direction', 'maximize') == 'maximize':
                    return float('-inf')  # 对于最大化问题，返回负无穷
                else:
                    return float('inf')   # 对于最小化问题，返回正无穷
                
        # 创建或加载Optuna study
        study_name = self.config.get('optuna', {}).get('study_name', 'sbs_model_optimization')
        storage = self.config.get('optuna', {}).get('storage', None)
        direction = self.config.get('optuna', {}).get('direction', 'maximize')
        
        # 选择合适的pruner
        pruner_name = self.config.get('optuna', {}).get('pruner', 'median')
        if pruner_name == 'median':
            pruner = optuna.pruners.MedianPruner()
        elif pruner_name == 'hyperband':
            pruner = optuna.pruners.HyperbandPruner()
        elif pruner_name == 'threshold':
            pruner = optuna.pruners.ThresholdPruner(
                upper=self.config.get('optuna', {}).get('pruner_threshold', 0.5)
            )
        else:
            pruner = optuna.pruners.MedianPruner()
            
        # 创建study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            pruner=pruner,
            load_if_exists=True
        )
        
        # 设置优化参数
        n_trials = self.config.get('optuna', {}).get('n_trials', 30)
        timeout = self.config.get('optuna', {}).get('timeout_seconds', 86400)  # 默认24小时
        n_jobs = self.config.get('optuna', {}).get('n_jobs', 1)
        
        # 开始优化
        self.logger.info(f"开始优化，计划进行{n_trials}次试验，超时时间: {timeout}秒")
        try:
            self.study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                gc_after_trial=True,
                show_progress_bar=True
            )
            
            # 记录优化结果
            self.best_params = self.study.best_params
            self.logger.info(f"优化完成，最佳参数: {self.best_params}")
            self.logger.info(f"最佳性能: {self.study.best_value:.4f}")
            
            # 保存结果
            self._save_optimization_results()
            
            return self.best_params
            
        except KeyboardInterrupt:
            self.logger.warning("优化被用户中断")
            # 保存当前最佳结果
            if hasattr(self.study, 'best_params'):
                self.best_params = self.study.best_params
                self._save_optimization_results(interrupted=True)
            return self.best_params
        
        except Exception as e:
            self.logger.error(f"优化过程出错: {str(e)}", exc_info=True)
            raise
        
    def _generate_params(self, trial: Trial) -> Dict[str, Any]:
        """
        生成超参数配置
        
        参数:
            trial: Optuna试验对象
            
        返回:
            超参数字典
        """
        # 基础模型参数
        model_params = {
            # 模型架构参数
            'input_channels': self.config.get('model', {}).get('input_size', 6),
            'hidden_size': trial.suggest_int('hidden_size', 64, 256, step=32),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
            
            # 训练参数
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'max_epochs': self.config.get('training', {}).get('epochs', 100),
            
            # 序列参数
            'sequence_length': trial.suggest_int('sequence_length', 30, 120, step=10),
            
            # 优化器参数
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw', 'sgd']),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        }
        
        # 条件参数：如果使用SGD优化器，添加动量参数
        if model_params['optimizer_type'] == 'sgd':
            model_params['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
        
        # 添加学习率调度器参数
        scheduler_enabled = trial.suggest_categorical('use_lr_scheduler', [True, False])
        if scheduler_enabled:
            model_params['scheduler_type'] = trial.suggest_categorical(
                'scheduler_type', ['step', 'cosine', 'plateau']
            )
            
            # 根据调度器类型添加特定参数
            if model_params['scheduler_type'] == 'step':
                model_params['lr_step_size'] = trial.suggest_int('lr_step_size', 5, 20)
                model_params['lr_gamma'] = trial.suggest_float('lr_gamma', 0.1, 0.5)
            elif model_params['scheduler_type'] == 'plateau':
                model_params['lr_patience'] = trial.suggest_int('lr_patience', 3, 10)
                model_params['lr_factor'] = trial.suggest_float('lr_factor', 0.1, 0.5)
        
        return model_params
        
    def _adjust_dataloader(self, dataloader: DataLoader, batch_size: int) -> DataLoader:
        """
        根据批次大小调整数据加载器
        
        参数:
            dataloader: 原始数据加载器
            batch_size: 新的批次大小
            
        返回:
            调整后的数据加载器
        """
        # 如果批次大小相同，直接返回原数据加载器
        if hasattr(dataloader, 'batch_size') and dataloader.batch_size == batch_size:
            return dataloader
            
        try:
            # 获取数据集和原始参数
            dataset = dataloader.dataset
            num_workers = getattr(dataloader, 'num_workers', 4)
            shuffle = getattr(dataloader, 'shuffle', False)
            pin_memory = getattr(dataloader, 'pin_memory', False)
            
            # 创建新的数据加载器
            return DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        except Exception as e:
            self.logger.error(f"调整数据加载器失败: {str(e)}")
            # 如果失败，返回原始数据加载器
            return dataloader
        
    def _create_model(self, params: Dict[str, Any]) -> pl.LightningModule:
        """
        创建模型实例
        
        参数:
            params: 模型参数
            
        返回:
            PyTorch Lightning模型
        """
        # 这里需要导入SBSLightningModel或SBSPredictor
        # 根据实际情况修改导入语句
        try:
            from ..model.sbs_predictor import SBSPredictor
            
            model = SBSPredictor(
                input_channels=params['input_channels'],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
            
            # 设置学习率
            model.learning_rate = params['learning_rate']
            
            # 如果需要，设置其他参数
            if 'optimizer_type' in params:
                model.optimizer_type = params['optimizer_type']
            if 'weight_decay' in params:
                model.weight_decay = params['weight_decay']
            if 'momentum' in params:
                model.momentum = params['momentum']
                
            return model
        except ImportError:
            self.logger.error("无法导入SBSPredictor，尝试创建简单的模型包装器")
            
            # 如果无法导入，创建一个简单的模型包装器
            class SimpleModel(pl.LightningModule):
                def __init__(self, params):
                    super().__init__()
                    self.save_hyperparameters()
                    self.params = params
                    
                    # 创建简单的LSTM模型
                    self.lstm = nn.LSTM(
                        input_size=params['input_channels'],
                        hidden_size=params['hidden_size'],
                        num_layers=params['num_layers'],
                        dropout=params['dropout'] if params['num_layers'] > 1 else 0,
                        batch_first=True
                    )
                    
                    self.fc = nn.Sequential(
                        nn.Linear(params['hidden_size'], params['hidden_size'] // 2),
                        nn.ReLU(),
                        nn.Dropout(params['dropout']),
                        nn.Linear(params['hidden_size'] // 2, 3)  # 3个输出类别
                    )
                    
                def forward(self, x):
                    # 假设x形状为[batch_size, seq_len, features]
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])  # 使用最后一个时间步的输出
                    
                def training_step(self, batch, batch_idx):
                    x, y = batch
                    y_hat = self(x)
                    loss = nn.functional.cross_entropy(y_hat, y)
                    self.log('train_loss', loss)
                    return loss
                    
                def validation_step(self, batch, batch_idx):
                    x, y = batch
                    y_hat = self(x)
                    loss = nn.functional.cross_entropy(y_hat, y)
                    acc = (y_hat.argmax(dim=1) == y).float().mean()
                    self.log('val_loss', loss, prog_bar=True)
                    self.log('val_acc', acc, prog_bar=True)
                    return {'val_loss': loss, 'val_acc': acc}
                    
                def configure_optimizers(self):
                    if self.params.get('optimizer_type') == 'adam':
                        optimizer = optim.Adam(
                            self.parameters(),
                            lr=self.params['learning_rate'],
                            weight_decay=self.params.get('weight_decay', 0)
                        )
                    elif self.params.get('optimizer_type') == 'adamw':
                        optimizer = optim.AdamW(
                            self.parameters(),
                            lr=self.params['learning_rate'],
                            weight_decay=self.params.get('weight_decay', 0)
                        )
                    elif self.params.get('optimizer_type') == 'sgd':
                        optimizer = optim.SGD(
                            self.parameters(),
                            lr=self.params['learning_rate'],
                            momentum=self.params.get('momentum', 0.9),
                            weight_decay=self.params.get('weight_decay', 0)
                        )
                    else:
                        optimizer = optim.Adam(
                            self.parameters(),
                            lr=self.params['learning_rate']
                        )
                        
                    # 配置学习率调度器
                    if self.params.get('scheduler_type') == 'step':
                        scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=self.params.get('lr_step_size', 10),
                            gamma=self.params.get('lr_gamma', 0.1)
                        )
                        return [optimizer], [scheduler]
                    elif self.params.get('scheduler_type') == 'cosine':
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=self.params.get('max_epochs', 100)
                        )
                        return [optimizer], [scheduler]
                    elif self.params.get('scheduler_type') == 'plateau':
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            mode='min',
                            factor=self.params.get('lr_factor', 0.1),
                            patience=self.params.get('lr_patience', 5),
                            verbose=True
                        )
                        return {
                            'optimizer': optimizer,
                            'lr_scheduler': {
                                'scheduler': scheduler,
                                'monitor': 'val_loss'
                            }
                        }
                    
                    return optimizer
                    
            return SimpleModel(params)
        
    def _create_trainer(self, trial: Trial, params: Dict[str, Any]) -> pl.Trainer:
        """
        创建训练器
        
        参数:
            trial: Optuna试验对象
            params: 模型参数
            
        返回:
            PyTorch Lightning训练器
        """
        # 创建回调
        callbacks = []
        
        # 添加早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('training', {}).get('early_stopping', {}).get('patience', 10),
            min_delta=self.config.get('training', {}).get('early_stopping', {}).get('min_delta', 0.001),
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stopping)
        
        # 添加模型检查点回调（如果需要）
        if self.config.get('training', {}).get('checkpointing', {}).get('enabled', False):
            checkpoint_dir = Path(self.config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir / f"trial_{trial.number}",
                filename=f"checkpoint-{trial.number}",
                save_top_k=1,
                monitor='val_loss',
                mode='min'
            )
            callbacks.append(checkpoint_callback)
        
        # 添加学习率监控
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        
        # 添加Optuna剪枝回调
        pruning_callback = PyTorchLightningPruningCallback(
            trial,
            monitor='val_loss'
        )
        callbacks.append(pruning_callback)
        
        # 创建日志记录器
        loggers = []
        
        # 添加TensorBoard日志记录器
        if self.config.get('tracking', {}).get('tensorboard', {}).get('enabled', False):
            tensorboard_dir = Path(self.config.get('tracking', {}).get('tensorboard', {}).get('log_dir', 'logs/tensorboard'))
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            
            tensorboard_logger = TensorBoardLogger(
                save_dir=str(tensorboard_dir),
                name=f"trial_{trial.number}",
                version=f"v{trial.number}"
            )
            loggers.append(tensorboard_logger)
        
        # 添加WandB日志记录器
        if self.config.get('tracking', {}).get('wandb', {}).get('enabled', False):
            wandb_logger = WandbLogger(
                project=self.config.get('tracking', {}).get('wandb', {}).get('project', 'sbs-training'),
                name=f"trial_{trial.number}",
                config=params
            )
            loggers.append(wandb_logger)
        
        # 创建训练器
        return pl.Trainer(
            max_epochs=params.get('max_epochs', 100),
            accelerator='auto',
            devices='auto',
            callbacks=callbacks,
            logger=loggers if loggers else False,
            enable_checkpointing=self.config.get('training', {}).get('checkpointing', {}).get('enabled', False),
            log_every_n_steps=self.config.get('logging', {}).get('log_every_n_steps', 10),
            deterministic=True
        )
    
    def _save_trials_history(self):
        """保存试验历史记录"""
        history_path = self.output_dir / f"trials_history_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.trials_history, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已保存试验历史到: {history_path}")
        except Exception as e:
            self.logger.error(f"保存试验历史失败: {str(e)}")
    
    def _save_optimization_results(self, interrupted: bool = False):
        """
        保存优化结果
        
        参数:
            interrupted: 是否因中断而保存
        """
        try:
            # 确保输出目录存在
            results_dir = self.output_dir
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存最佳参数
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            status = "interrupted" if interrupted else "completed"
            
            params_path = results_dir / f"best_params_{status}_{timestamp}.yaml"
            with open(params_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.best_params, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            # 如果有study，保存更多详细信息
            if self.study:
                # 保存所有试验结果
                study_results = {
                    'best_params': self.best_params,
                    'best_value': self.study.best_value,
                    'best_trial': self.study.best_trial.number,
                    'n_trials': len(self.study.trials),
                    'datetime': timestamp,
                    'status': status
                }
                
                results_path = results_dir / f"optimization_results_{status}_{timestamp}.yaml"
                with open(results_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(study_results, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                
                # 保存参数重要性
                try:
                    importances = optuna.importance.get_param_importances(self.study)
                    importance_path = results_dir / f"param_importance_{status}_{timestamp}.yaml"
                    with open(importance_path, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(dict(importances), f, default_flow_style=False, allow_unicode=True)
                except Exception as e:
                    self.logger.warning(f"无法计算参数重要性: {str(e)}")
                
                # 保存所有试验结果的详细信息
                detailed_results = []
                for trial in self.study.trials:
                    if trial.state.is_finished():
                        detailed_results.append({
                            'number': trial.number,
                            'params': trial.params,
                            'value': trial.value,
                            'state': str(trial.state)
                        })
                
                details_path = results_dir / f"trials_details_{status}_{timestamp}.json"
                with open(details_path, 'w', encoding='utf-8') as f:
                    json.dump(detailed_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"已保存优化结果到目录: {results_dir}")
            
        except Exception as e:
            self.logger.error(f"保存优化结果失败: {str(e)}")
    
    def train_with_best_params(self, train_data: DataLoader, val_data: DataLoader, 
                              test_data: Optional[DataLoader] = None) -> Any:
        """
        使用最佳参数训练最终模型
        
        参数:
            train_data: 训练数据加载器
            val_data: 验证数据加载器
            test_data: 测试数据加载器 (可选)
            
        返回:
            训练好的模型
        """
        if not self.best_params:
            raise ValueError("请先运行optimize方法获取最佳参数")
            
        self.logger.info("使用最佳参数训练最终模型...")
        
        # 创建完整参数配置
        best_config = self.best_params.copy()
        
        # 使用更多的训练轮数
        final_epochs = self.config.get('training', {}).get('final_epochs', 200)
        best_config['max_epochs'] = final_epochs
        
        # 确保使用最佳批次大小
        if 'batch_size' in self.best_params:
            train_data = self._adjust_dataloader(train_data, self.best_params['batch_size'])
            val_data = self._adjust_dataloader(val_data, self.best_params['batch_size'])
            if test_data:
                test_data = self._adjust_dataloader(test_data, self.best_params['batch_size'])
        
        # 创建最终模型
        final_model = self._create_model(best_config)
        
        # 创建检查点回调
        checkpoint_dir = Path(self.config.get('paths', {}).get('model_dir', 'models/final'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"sbs_best_model_{timestamp}",
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )
        
        # 创建早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('training', {}).get('early_stopping', {}).get('patience', 15),
            min_delta=self.config.get('training', {}).get('early_stopping', {}).get('min_delta', 0.001),
            verbose=True,
            mode='min'
        )
        
        # 创建学习率监控
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        # 创建日志记录器
        loggers = []
        
        # 添加TensorBoard日志记录器
        tensorboard_dir = Path(self.config.get('paths', {}).get('log_dir', 'logs/final_training'))
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        tensorboard_logger = TensorBoardLogger(
            save_dir=str(tensorboard_dir),
            name="final_model",
            version=timestamp
        )
        loggers.append(tensorboard_logger)
        
        # 添加WandB日志记录器（如果配置了）
        if self.config.get('tracking', {}).get('wandb', {}).get('project', None):
            try:
                wandb_logger = WandbLogger(
                    project=self.config.get('tracking', {}).get('wandb', {}).get('project', 'sbs-training'),
                    name=f"final_model_{timestamp}",
                    config=best_config
                )
                loggers.append(wandb_logger)
            except Exception as e:
                self.logger.warning(f"初始化WandB日志记录器失败: {str(e)}")
        
        # 创建最终训练器
        final_trainer = pl.Trainer(
            max_epochs=final_epochs,
            accelerator='auto',
            devices='auto',
            callbacks=[checkpoint_callback, early_stopping, lr_monitor],
            logger=loggers,
            log_every_n_steps=50,
            deterministic=True
        )
        
        # 训练最终模型
        self.logger.info(f"开始训练最终模型，最大轮数: {final_epochs}")
        try:
            final_trainer.fit(final_model, train_data, val_data)
            
            # 如果有测试数据，进行测试
            if test_data:
                self.logger.info("对最终模型进行测试...")
                test_result = final_trainer.test(final_model, test_data)
                self.logger.info(f"测试结果: {test_result}")
                
            # 保存模型
            model_path = checkpoint_dir / f"sbs_final_model_{timestamp}.pt"
            torch.save(final_model.state_dict(), model_path)
            self.logger.info(f"模型已保存到: {model_path}")
            
            # 保存最佳模型路径
            best_model_info = {
                'best_model_path': str(model_path),
                'best_checkpoint_path': checkpoint_callback.best_model_path,
                'best_val_loss': checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
                'parameters': best_config,
                'training_date': timestamp
            }
            
            info_path = checkpoint_dir / f"model_info_{timestamp}.yaml"
            with open(info_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(best_model_info, f, default_flow_style=False, allow_unicode=True)
                
            self.logger.info(f"最终模型训练完成，最佳验证损失: {best_model_info['best_val_loss']}")
            
            return final_model
            
        except Exception as e:
            self.logger.error(f"最终模型训练失败: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    # 简单的使用示例
    import yaml
    
    # 加载配置
    with open('config/training_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建优化器
    optimizer = SBSOptimizer(config)
    
    # 接下来应该加载数据并进行优化
    # 这里只是示例，实际使用时需要替换为真实数据
    # optimizer.optimize(train_data, val_data) 