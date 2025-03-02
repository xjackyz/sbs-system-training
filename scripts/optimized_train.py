#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用PyTorch Lightning优化的训练脚本
整合了人工标注、主动学习、奖励机制和分布式训练
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any

# 设置WandB API密钥
os.environ["WANDB_API_KEY"] = "135573f7b891c2086c6e99c3c22fa3ec5543445d"

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import ray
from datetime import datetime
import wandb
import torchmetrics

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_collector import DataCollector
from src.analysis.llava_analyzer import LLaVAAnalyzer
from src.signal_generator import SignalGenerator
from src.self_supervised.utils.active_learner import ActiveLearner
from src.self_supervised.utils.reward_calculator import RewardCalculator
from src.self_supervised.utils.label_studio_adapter import LabelStudioAdapter
from src.cache.redis_manager import RedisManager
from src.utils.logger import setup_logger, TrainingLogger

# 设置日志
logger = setup_logger('optimized_training')
training_logger = TrainingLogger('optimized_training')

class SBSLightningModule(pl.LightningModule):
    def __init__(self, config: Dict):
        """初始化Lightning模块"""
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # 初始化模型组件
        self.analyzer = LLaVAAnalyzer(config=config)
        self.signal_generator = SignalGenerator(config=config)
        self.reward_calculator = RewardCalculator(config=config)
        
        # 缓存管理
        self.cache_manager = RedisManager(config=config)
        
        # 指标记录
        self.train_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=3),
            'precision': torchmetrics.Precision(task='multiclass', num_classes=3),
            'recall': torchmetrics.Recall(task='multiclass', num_classes=3),
            'f1': torchmetrics.F1Score(task='multiclass', num_classes=3)
        })
        
        self.val_metrics = self.train_metrics.clone()
        
        # 记录模型架构图
        if hasattr(self.analyzer.model, 'config'):
            wandb.watch(self.analyzer.model)
            
        # 保存验证预测结果
        self.val_predictions = []
        self.val_targets = []
        self.class_names = ['下跌', '盘整', '上涨']
        
    def forward(self, x):
        """模型前向传播"""
        return self.analyzer.model(x)
        
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        market_data, human_labels = batch
        
        # 1. 模型预测
        predictions = self(market_data)
        
        # 2. 生成交易信号
        trade_signals = self.signal_generator.generate_signals(predictions)
        
        # 3. 计算奖励和损失
        rewards = self.reward_calculator.calculate_reward(
            prediction=predictions,
            human_label=human_labels,
            trade_result=trade_signals
        )
        
        loss = self._calculate_loss(predictions, human_labels, rewards)
        
        # 更新指标
        self.train_metrics.update(predictions, human_labels)
        
        # 记录详细指标到WandB
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_reward', torch.mean(rewards), on_step=True, on_epoch=True)
        self.log('train_reward_std', torch.std(rewards), on_step=True, on_epoch=True)
        self.log('train_pred_confidence', torch.mean(predictions.max(dim=1)[0]), on_step=True)
        
        # 记录学习率
        opt = self.optimizers()
        if isinstance(opt, torch.optim.Optimizer):
            self.log('learning_rate', opt.param_groups[0]['lr'], on_step=True)
            
        # 记录梯度范数
        if self.trainer.global_step % 100 == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.log('grad_norm', grad_norm, on_step=True)
            
        return loss
        
    def training_epoch_end(self, outputs):
        """训练轮次结束"""
        # 计算并记录训练指标
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f'train_{name}_epoch', value)
            
        # 重置指标
        self.train_metrics.reset()
        
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        market_data, human_labels = batch
        predictions = self(market_data)
        trade_signals = self.signal_generator.generate_signals(predictions)
        
        rewards = self.reward_calculator.calculate_reward(
            prediction=predictions,
            human_label=human_labels,
            trade_result=trade_signals
        )
        
        loss = self._calculate_loss(predictions, human_labels, rewards)
        
        # 更新指标
        self.val_metrics.update(predictions, human_labels)
        
        # 保存预测结果用于生成混淆矩阵
        self.val_predictions.extend(predictions.argmax(dim=1).cpu().numpy())
        self.val_targets.extend(human_labels.cpu().numpy())
        
        # 记录详细验证指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_reward', torch.mean(rewards), on_epoch=True)
        self.log('val_reward_std', torch.std(rewards), on_epoch=True)
        self.log('val_pred_confidence', torch.mean(predictions.max(dim=1)[0]), on_epoch=True)
        
        return {'val_loss': loss, 'predictions': predictions, 'targets': human_labels}
        
    def validation_epoch_end(self, outputs):
        """验证轮次结束时的处理"""
        # 计算平均指标
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f'val_{name}_epoch', value)
            
        # 创建混淆矩阵
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=self.val_targets,
                preds=self.val_predictions,
                class_names=self.class_names
            )
        })
        
        # 重置存储的预测结果
        self.val_predictions = []
        self.val_targets = []
        
        # 重置指标
        self.val_metrics.reset()
        
    def _calculate_loss(self, predictions, human_labels, rewards):
        """计算加权损失"""
        # 基础预测损失
        pred_loss = torch.nn.functional.cross_entropy(predictions, human_labels)
        
        # 奖励加权
        reward_weight = self.config['training']['reward']['profit_weight']
        weighted_loss = pred_loss * (1 - reward_weight) + (-torch.mean(rewards)) * reward_weight
        
        return weighted_loss
        
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 获取优化器配置
        opt_config = self.config['training']['optimizer']
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=opt_config['weight_decay'],
            betas=(opt_config['beta1'], opt_config['beta2'])
        )
        
        # 获取调度器配置
        sched_config = self.config['training']['scheduler']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_config['factor'],
            patience=sched_config['patience'],
            min_lr=sched_config['min_lr'],
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }
        
    def on_save_checkpoint(self, checkpoint):
        """保存检查点时的回调"""
        # 保存额外信息
        checkpoint['class_names'] = self.class_names
        checkpoint['config'] = self.config
        
    def on_load_checkpoint(self, checkpoint):
        """加载检查点时的回调"""
        # 恢复额外信息
        self.class_names = checkpoint['class_names']
        self.config = checkpoint['config']

class OptimizedTrainer:
    def __init__(self, config_path: str):
        """初始化优化训练器"""
        self.config = self._load_config(config_path)
        self._init_components()
        
    def _load_config(self, config_path: str) -> Dict:
        """加载训练配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def _init_components(self):
        """初始化各个组件"""
        self.data_collector = DataCollector(config=self.config)
        self.active_learner = ActiveLearner(config=self.config)
        self.label_studio = LabelStudioAdapter(config=self.config)
        
    async def train(self):
        """执行优化训练流程"""
        try:
            # 初始化WandB
            wandb_config = self.config.get('wandb', {})
            wandb.init(
                project=wandb_config.get('project', 'sbs-training'),
                name=wandb_config.get('run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=self.config,
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', '自动训练运行')
            )
            
            # 1. 收集市场数据
            logger.info("开始收集市场数据...")
            market_data = self.data_collector.collect_market_data(
                symbol="BTCUSDT",
                interval="1d"
            )
            
            # 2. 主动学习选择样本
            logger.info("选择待标注样本...")
            uncertain_samples = self.active_learner.uncertainty_sampling(
                data=market_data,
                n_samples=self.config['training']['active_learning']['max_samples_per_batch']
            )
            
            # 3. 导入Label Studio进行标注
            logger.info("导入待标注样本到Label Studio...")
            project_id = self.label_studio.create_project(
                name=f"SBS标注项目_{datetime.now().strftime('%Y%m%d')}",
                description="SBS序列标注任务"
            )
            self.label_studio.import_tasks(project_id, uncertain_samples)
            
            # 4. 获取人工标注结果
            logger.info("等待人工标注完成...")
            human_labels = self.label_studio.export_annotations(project_id)
            
            # 5. 准备数据集
            train_data = self._prepare_training_data(market_data, human_labels)
            
            # 6. 创建Lightning模块和训练器
            model = SBSLightningModule(config=self.config)
            
            # 设置回调
            callbacks = [
                ModelCheckpoint(
                    dirpath='checkpoints',
                    filename='sbs-{epoch:02d}-{val_loss:.2f}',
                    save_top_k=3,
                    monitor='val_loss'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    mode='min'
                ),
                LearningRateMonitor(logging_interval='step')
            ]
            
            # 设置WandB Logger
            wandb_logger = WandbLogger(
                project=wandb_config.get('project', 'sbs-training'),
                log_model=True,
                save_dir='wandb_logs'
            )
            
            # 创建训练器
            trainer = pl.Trainer(
                max_epochs=self.config['training']['epochs'],
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices='auto',
                strategy='ddp' if torch.cuda.device_count() > 1 else None,
                callbacks=callbacks,
                logger=wandb_logger,  # 使用WandB Logger
                log_every_n_steps=10,
                val_check_interval=0.25
            )
            
            # 7. 开始训练
            logger.info("开始Lightning训练...")
            trainer.fit(model, train_data)
            
            # 8. 保存训练结果
            results = {
                'best_model_path': trainer.checkpoint_callback.best_model_path,
                'best_model_score': trainer.checkpoint_callback.best_model_score,
                'training_duration': str(trainer.fit_loop.total_batch_idx)
            }
            
            # 完成训练后上传额外文件
            wandb.save('config/training_config.yaml')
            wandb.save('outputs/training_results/*')
            
            # 结束WandB运行
            wandb.finish()
            
            return results
            
        except Exception as e:
            logger.error(f"训练过程出错: {str(e)}")
            if wandb.run is not None:
                wandb.finish(exit_code=1)
            raise
            
    def _prepare_training_data(self, 
                           market_data: List[Dict],
                           human_labels: List[Dict]) -> pl.LightningDataModule:
        """准备训练数据"""
        from src.data.sbs_datamodule import SBSDataModule
        
        return SBSDataModule(
            market_data=market_data,
            labels=human_labels,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['device']['num_workers'],
            val_split=self.config['training']['validation_split']
        )

def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='优化训练脚本')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                      help='训练配置文件路径')
    parser.add_argument('--wandb-key', type=str, help='WandB API密钥')
    args = parser.parse_args()
    
    # 设置WandB API密钥
    if args.wandb_key:
        os.environ['WANDB_API_KEY'] = args.wandb_key
    
    try:
        trainer = OptimizedTrainer(config_path=args.config)
        training_results = asyncio.run(trainer.train())
        
        # 保存训练结果
        output_dir = Path('outputs/training_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(training_results, f, allow_unicode=True)
            
        logger.info(f"训练结果已保存至: {output_path}")
        
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        raise

if __name__ == '__main__':
    main() 