#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SBS模型超参数优化脚本
用于自动化超参数搜索，并使用最佳参数训练最终模型
"""

import os
import sys
import yaml
import logging
from pathlib import Path
import argparse
from datetime import datetime
import random
import numpy as np
import torch

# 添加项目根目录到PATH以便导入src模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sbs_data import SBSDataModule
from src.self_supervised.utils.sbs_optimizer import SBSOptimizer

def set_random_seeds(seed):
    """
    设置随机种子以确保可重复性
    
    参数:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def setup_logging(config):
    """
    设置日志配置
    
    参数:
        config: 配置字典
    """
    log_level = config.get('logging', {}).get('level', 'INFO')
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 确保日志目录存在
    log_dir = Path(config.get('logging', {}).get('log_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置控制台和文件处理器
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            log_dir / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    ]
    
    # 配置根日志记录器
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )

def ensure_directories(path_config):
    """
    确保所有必要的目录都存在
    
    参数:
        path_config: 路径配置字典
    """
    for key, path in path_config.items():
        if isinstance(path, str) and ('dir' in key or 'path' in key):
            Path(path).mkdir(parents=True, exist_ok=True)
            logging.info(f"确保目录存在: {path}")

def main():
    """
    主函数，执行SBS模型的超参数优化
    """
    parser = argparse.ArgumentParser(description='SBS模型超参数优化器')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子，若未设置则使用配置中的种子')
    parser.add_argument('--trials', type=int, default=None,
                       help='Optuna试验次数，若未设置则使用配置中的值')
    parser.add_argument('--timeout', type=int, default=None,
                       help='优化超时时间(秒)，若未设置则使用配置中的值')
    args = parser.parse_args()
    
    try:
        # 加载配置
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 覆盖配置中的特定参数
        if args.seed is not None:
            config['seed'] = args.seed
        if args.trials is not None:
            config['optuna']['n_trials'] = args.trials
        if args.timeout is not None:
            config['optuna']['timeout_seconds'] = args.timeout
        
        # 设置日志
        setup_logging(config)
        logging.info(f"加载配置文件: {args.config}")
        
        # 设置随机种子
        seed = config.get('seed', 42)
        set_random_seeds(seed)
        logging.info(f"设置随机种子: {seed}")
        
        # 确保目录存在
        ensure_directories(config.get('paths', {}))
        
        # 创建数据模块
        logging.info("初始化数据模块...")
        data_module = SBSDataModule(
            data_dir=config['paths']['data_dir'],
            batch_size=config['training']['dataloader']['batch_size'],
            sequence_length=config['training']['dataloader']['sequence_length'],
            train_ratio=config['training']['dataloader']['train_ratio'],
            val_ratio=config['training']['dataloader']['val_ratio'],
            num_workers=config['training']['dataloader']['num_workers']
        )
        
        # 准备数据
        logging.info("准备训练数据...")
        data_module.prepare_data()
        data_module.setup()
        
        # 创建和配置优化器
        logging.info("初始化SBSOptimizer...")
        optimizer = SBSOptimizer(config)
        
        # 开始优化流程
        logging.info(f"开始优化流程，最大试验次数: {config['optuna']['n_trials']}, " +
                    f"超时时间: {config['optuna']['timeout_seconds']}秒")
        
        # 执行超参数搜索
        best_params = optimizer.optimize(
            data_module.train_dataloader(),
            data_module.val_dataloader()
        )
        
        # 使用最佳参数训练最终模型
        logging.info(f"使用最佳参数训练最终模型: {best_params}")
        final_model = optimizer.train_with_best_params(
            data_module.train_dataloader(),
            data_module.val_dataloader(),
            data_module.test_dataloader()
        )
        
        logging.info("优化和训练完成！")
        
    except Exception as e:
        logging.error(f"优化过程中发生错误: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 