#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行主动学习采样脚本
用于选择需要人工标注的样本
"""

import os
import argparse
from pathlib import Path
from src.self_supervised.model.llava_processor import LlavaProcessor
from src.self_supervised.utils.active_learner import ActiveLearner
from src.utils.logger import setup_logger

logger = setup_logger('active_learning')

def parse_args():
    parser = argparse.ArgumentParser(description='运行主动学习采样')
    parser.add_argument('--unlabeled-dir', type=str, required=True,
                      help='未标注数据目录路径')
    parser.add_argument('--output-dir', type=str, default='data/samples_for_review',
                      help='输出目录路径')
    parser.add_argument('--model-path', type=str, default='models/llava-sbs',
                      help='LLaVA模型路径')
    parser.add_argument('--uncertainty-ratio', type=float, default=0.05,
                      help='不确定性采样比例')
    parser.add_argument('--random-ratio', type=float, default=0.10,
                      help='随机采样比例')
    parser.add_argument('--n-clusters', type=int, default=5,
                      help='聚类数量')
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    try:
        # 初始化LLaVA处理器
        logger.info(f"初始化LLaVA处理器，使用模型: {args.model_path}")
        processor = LlavaProcessor(model_path=args.model_path)
        
        # 配置主动学习参数
        config = {
            'uncertainty_ratio': args.uncertainty_ratio,
            'random_ratio': args.random_ratio,
            'n_clusters': args.n_clusters
        }
        
        # 初始化主动学习采样器
        learner = ActiveLearner(
            llava_processor=processor,
            unlabeled_dir=args.unlabeled_dir,
            output_dir=args.output_dir,
            config=config
        )
        
        # 运行采样过程
        logger.info("开始采样过程...")
        selected_files, stats = learner.run_sampling()
        
        # 输出统计信息
        logger.info("采样完成！统计信息：")
        logger.info(f"- 总样本数：{stats['total_samples']}")
        logger.info(f"- 选中样本数：{stats['selected_samples']}")
        logger.info(f"- 不确定性样本数：{stats['uncertainty_samples']}")
        logger.info(f"- 随机样本数：{stats['random_samples']}")
        logger.info(f"- 平均不确定性：{stats['mean_uncertainty']:.4f}")
        logger.info(f"- 时间戳：{stats['timestamp']}")
        
        logger.info(f"选中的样本已保存到：{args.output_dir}/samples_for_review_{stats['timestamp']}")
        
    except Exception as e:
        logger.error(f"运行失败: {str(e)}")
        raise

if __name__ == '__main__':
    main() 