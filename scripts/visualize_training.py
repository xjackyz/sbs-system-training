#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练可视化脚本
用于生成学习曲线和训练进度图表
"""

import os
import argparse
from pathlib import Path
from src.self_supervised.utils.visualization import LearningVisualizer
from src.utils.logger import setup_logger

logger = setup_logger('visualization')

def parse_args():
    parser = argparse.ArgumentParser(description='生成训练可视化图表')
    parser.add_argument('--stats-dir', type=str, default='logs/rewards',
                      help='统计数据目录路径')
    parser.add_argument('--output-dir', type=str, default='logs/visualization',
                      help='输出目录路径')
    parser.add_argument('--window-size', type=int, default=7,
                      help='移动平均窗口大小（天）')
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    try:
        # 初始化可视化工具
        visualizer = LearningVisualizer(save_dir=args.output_dir)
        
        # 1. 绘制学习曲线
        logger.info("正在生成学习曲线...")
        learning_curves_path = visualizer.plot_learning_curves(args.stats_dir)
        if learning_curves_path:
            logger.info(f"学习曲线已保存至: {learning_curves_path}")
        
        # 2. 绘制奖励分布图
        logger.info("正在生成奖励分布图...")
        reward_dist_path = visualizer.plot_reward_distribution(args.stats_dir)
        if reward_dist_path:
            logger.info(f"奖励分布图已保存至: {reward_dist_path}")
        
        # 3. 绘制训练进度图表
        logger.info("正在生成训练进度图表...")
        progress_path = visualizer.plot_training_progress(
            args.stats_dir,
            window_size=args.window_size
        )
        if progress_path:
            logger.info(f"训练进度图表已保存至: {progress_path}")
        
        logger.info("可视化完成！")
        
    except Exception as e:
        logger.error(f"生成可视化图表失败: {str(e)}")
        raise

if __name__ == '__main__':
    main() 