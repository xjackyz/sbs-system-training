#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SBS 训练入口脚本
提供统一的训练入口，支持多种训练模式
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from src.self_supervised.utils.config_manager import ConfigManager
from src.self_supervised.trainer.sbs_trainer import SBSTrainer
from src.self_supervised.utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SBS 训练入口脚本')
    
    # 基本参数
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--mode', type=str, default='standard', 
                        choices=['standard', 'self_supervised', 'rl', 'active_learning'],
                        help='训练模式: standard, self_supervised, rl, active_learning')
    parser.add_argument('--output_dir', type=str, default='', help='输出目录')
    parser.add_argument('--resume', type=str, default='', help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='', help='数据路径')
    parser.add_argument('--labeled_path', type=str, default='', help='已标记数据路径')
    parser.add_argument('--unlabeled_path', type=str, default='', help='未标记数据路径')
    parser.add_argument('--val_path', type=str, default='', help='验证数据路径')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=0, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=0, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0, help='学习率')
    parser.add_argument('--device', type=str, default='', help='训练设备')
    
    # 日志参数
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')
    parser.add_argument('--log_dir', type=str, default='', help='日志目录')
    
    return parser.parse_args()

def setup_environment(seed):
    """设置环境和随机种子"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    """主函数"""
    args = parse_args()
    
    # 设置环境
    setup_environment(args.seed)
    
    # 创建目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('outputs', f"{args.mode}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_dir = args.log_dir if args.log_dir else os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_level = getattr(logging, args.log_level)
    logger = setup_logger('SBSTraining', log_file=os.path.join(log_dir, 'training.log'), level=log_level)
    
    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 使用命令行参数覆盖配置
    if args.batch_size > 0:
        config['batch_size'] = args.batch_size
    if args.epochs > 0:
        config['epochs'] = args.epochs
    if args.learning_rate > 0:
        config['optimizer']['learning_rate'] = args.learning_rate
    if args.device:
        config['device'] = args.device
    if args.resume:
        config['resume_from'] = args.resume
        
    # 设置数据路径
    if args.data_path:
        config['data_path'] = args.data_path
    if args.labeled_path:
        config['labeled_data_path'] = args.labeled_path
    if args.unlabeled_path:
        config['unlabeled_data_path'] = args.unlabeled_path
    if args.val_path:
        config['val_data_path'] = args.val_path
        
    # 设置输出目录
    config['output_dir'] = output_dir
    config['log_dir'] = log_dir
    config['checkpoint_dir'] = os.path.join(output_dir, 'checkpoints')
    
    # 保存最终配置
    config_manager.save_config(config, os.path.join(output_dir, 'config.yaml'))
    
    # 创建训练器
    logger.info(f"创建SBS训练器，训练模式: {args.mode}")
    trainer = SBSTrainer(config, logger=logger)
    
    # 开始训练
    logger.info("开始训练")
    result = trainer.start_training(mode=args.mode)
    
    # 记录结果
    logger.info(f"训练完成，最佳指标: {result.get('best_metric', 0)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 