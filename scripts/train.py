#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SBS System 训练脚本
整合了自适应学习、分布式训练、缓存管理和标注系统
"""

import os
import sys
import argparse
import logging
import pytz
from datetime import datetime
from pathlib import Path
import torch
import ray
from typing import Dict, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.models.llava_model import LLaVAModel
from src.adaptive_learning.adaptive_trainer import AdaptiveTrainer
from src.distributed.ray_trainer import DistributedTrainingManager
from src.cache.redis_manager import RedisManager
from src.web.label_studio_integration import LabelStudioIntegration
from src.self_supervised.utils.reward_calculator import RewardCalculator
from src.self_supervised.utils.active_learner import ActiveLearner
from src.self_supervised.utils.visualization import LearningVisualizer
from src.utils.taichi_renderer import TaichiKLineRenderer
from src.utils.render_manager import RenderManager

class BeijingFormatter(logging.Formatter):
    """北京时间日志格式化器"""
    def formatTime(self, record, datefmt=None):
        beijing_tz = pytz.timezone('Asia/Shanghai')
        utc_dt = datetime.fromtimestamp(record.created, pytz.utc)
        beijing_dt = utc_dt.astimezone(beijing_tz)
        return beijing_dt.strftime(datefmt or '%Y-%m-%d %H:%M:%S')

def clean_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def setup_training_environment(config: Dict) -> Dict:
    """
    设置训练环境
    
    Args:
        config: 配置字典
        
    Returns:
        环境配置
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(
            num_gpus=torch.cuda.device_count(),
            ignore_reinit_error=True,
            logging_level=logging.INFO
        )
    
    # 初始化Redis缓存
    redis_manager = RedisManager(config['cache'])
    
    # 初始化Label Studio集成
    label_studio = LabelStudioIntegration(config['label_studio'])
    
    # 初始化渲染管理器
    render_manager = RenderManager(config['render'])
    
    return {
        'device': device,
        'redis_manager': redis_manager,
        'label_studio': label_studio,
        'render_manager': render_manager
    }

async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SBS System 训练脚本')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                      help='训练配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='模型检查点路径')
    parser.add_argument('--num-workers', type=int, default=None,
                      help='分布式训练工作节点数')
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 设置日志
        logger = setup_logger('train', level=config['log_level'])
        logger.info("开始训练过程...")
        
        # 设置训练环境
        env = setup_training_environment(config)
        logger.info(f"使用设备: {env['device']}")
        
        # 初始化模型
        model = LLaVAModel(config['model'])
        model.to(env['device'])
        
        if args.checkpoint:
            model.load(args.checkpoint)
            logger.info(f"加载检查点: {args.checkpoint}")
        
        # 初始化训练组件
        adaptive_trainer = AdaptiveTrainer(model, config['training'])
        reward_calculator = RewardCalculator(config['reward'])
        active_learner = ActiveLearner(config['active_learning'])
        visualizer = LearningVisualizer(save_dir='logs/visualization')
        
        # 设置分布式训练
        num_workers = args.num_workers or config['training'].get('num_workers', 1)
        dist_manager = DistributedTrainingManager(num_workers, config['model'])
        
        # 训练循环
        for epoch in range(config['training']['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
            
            # 清理GPU内存
            clean_gpu_memory()
            
            # 获取训练数据
            train_data = active_learner.get_training_data()
            
            # 分布式训练
            train_results = await dist_manager.train(train_data)
            
            # 计算奖励和更新
            rewards = reward_calculator.calculate_batch_rewards(train_results)
            adaptive_trainer.update_from_rewards(rewards)
            
            # 调整学习率和批大小
            new_lr = adaptive_trainer.adjust_learning_rate(train_results['metrics'])
            new_batch_size = adaptive_trainer.update_batch_size(train_results['memory_usage'])
            
            logger.info(f"新学习率: {new_lr:.6f}, 新批大小: {new_batch_size}")
            
            # 可视化训练进度
            visualizer.plot_training_progress(epoch_stats=train_results)
            
            # 保存检查点
            if (epoch + 1) % config['training']['save_interval'] == 0:
                checkpoint_path = f"models/checkpoints/epoch_{epoch + 1}.pt"
                model.save(checkpoint_path)
                logger.info(f"保存检查点: {checkpoint_path}")
            
            # 同步标注结果
            if (epoch + 1) % config['training']['label_sync_interval'] == 0:
                annotations = env['label_studio'].export_annotations(config['label_studio']['project_id'])
                active_learner.update_from_annotations(annotations)
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        raise
    finally:
        # 清理资源
        if ray.is_initialized():
            ray.shutdown()
        env['redis_manager'].clear_cache()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main()) 