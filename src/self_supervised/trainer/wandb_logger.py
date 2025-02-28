import os
import wandb
import torch
import logging
from typing import Dict, Any, Optional

class WandbLogger:
    def __init__(self, config: Dict[str, Any], project: str = "sbs-trading",
                 name: Optional[str] = None, tags: Optional[list] = None):
        """
        初始化Weights & Biases日志记录器
        
        Args:
            config: 配置字典
            project: 项目名称
            name: 实验名称
            tags: 标签列表
        """
        self.config = config
        self.project = project
        self.name = name or f"experiment_{wandb.util.generate_id()}"
        self.tags = tags or ["production"]
        
        # 初始化wandb
        try:
            wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                tags=self.tags
            )
            self.initialized = True
            logging.info(f"Weights & Biases初始化成功: {self.project}/{self.name}")
        except Exception as e:
            self.initialized = False
            logging.error(f"Weights & Biands初始化失败: {e}")
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        记录训练指标
        
        Args:
            metrics: 指标字典
            step: 当前步数
        """
        if not self.initialized:
            return
            
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logging.error(f"指标记录失败: {e}")
            
    def log_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int):
        """
        保存模型检查点
        
        Args:
            model: PyTorch模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
        """
        if not self.initialized:
            return
            
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None
            }
            
            # 保存到本地
            checkpoint_path = f"checkpoints/epoch_{epoch}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            
            # 上传到W&B
            artifact = wandb.Artifact(
                name=f"model-checkpoint-{epoch}",
                type="model",
                description=f"Model checkpoint at epoch {epoch}"
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            
        except Exception as e:
            logging.error(f"模型保存失败: {e}")
            
    def log_gpu_stats(self):
        """记录GPU统计信息"""
        if not self.initialized:
            return
            
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_stats = {
                        f"gpu_{i}/memory_allocated": torch.cuda.memory_allocated(i) / 1e9,  # GB
                        f"gpu_{i}/memory_cached": torch.cuda.memory_reserved(i) / 1e9,  # GB
                        f"gpu_{i}/utilization": torch.cuda.utilization(i)
                    }
                    wandb.log(gpu_stats)
        except Exception as e:
            logging.error(f"GPU统计信息记录失败: {e}")
            
    def log_batch(self, batch_metrics: Dict[str, float], batch_idx: int, epoch: int):
        """
        记录批次级别的指标
        
        Args:
            batch_metrics: 批次指标
            batch_idx: 批次索引
            epoch: 当前epoch
        """
        if not self.initialized or batch_idx % self.config['log_interval']['batch'] != 0:
            return
            
        try:
            metrics = {f"batch/{k}": v for k, v in batch_metrics.items()}
            metrics.update({
                "batch/index": batch_idx,
                "epoch": epoch
            })
            self.log_metrics(metrics)
            self.log_gpu_stats()
        except Exception as e:
            logging.error(f"批次指标记录失败: {e}")
            
    def log_epoch(self, epoch_metrics: Dict[str, float], epoch: int):
        """
        记录epoch级别的指标
        
        Args:
            epoch_metrics: epoch指标
            epoch: 当前epoch
        """
        if not self.initialized:
            return
            
        try:
            metrics = {f"epoch/{k}": v for k, v in epoch_metrics.items()}
            metrics["epoch"] = epoch
            self.log_metrics(metrics)
        except Exception as e:
            logging.error(f"Epoch指标记录失败: {e}")
            
    def finish(self):
        """结束记录"""
        if self.initialized:
            wandb.finish() 