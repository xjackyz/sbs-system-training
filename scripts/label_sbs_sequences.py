#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SBS序列标注工具
用于人工标注和修正SBS序列点位
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.self_supervised.utils.label_studio_adapter import LabelStudioAdapter
from src.self_supervised.utils.active_learner import ActiveLearner
from src.self_supervised.utils.reward_calculator import RewardCalculator
from src.utils.logger import setup_logger

logger = setup_logger('label_tool')

def parse_args():
    parser = argparse.ArgumentParser(description='SBS序列标注工具')
    parser.add_argument('--data', type=str, required=True,
                      help='K线数据CSV文件路径')
    parser.add_argument('--output-dir', type=str, default='data/labeled',
                      help='标注结果输出目录')
    parser.add_argument('--feedback-dir', type=str, default='data/feedback',
                      help='反馈数据保存目录')
    parser.add_argument('--window-size', type=int, default=100,
                      help='显示窗口大小')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 初始化Label Studio适配器
    label_studio_adapter = LabelStudioAdapter(config)
    
    # 创建项目
    project_id = label_studio_adapter.create_project(name="SBS序列标注", description="用于标注SBS序列的项目")
    
    # 初始化主动学习器和奖励计算器
    active_learner = ActiveLearner()
    reward_calculator = RewardCalculator()
    
    # 加载数据
    data = pd.read_csv(args.data)
    logger.info(f"已加载数据: {len(data)} 条记录")
    
    # 收集不确定性样本
    uncertain_samples = active_learner.uncertainty_sampling(predictions, n_samples=10)
    
    # 导入不确定性样本
    label_studio_adapter.import_uncertain_tasks(project_id, uncertain_samples, kline_images)
    
    # 进行人工标注
    # ...
    
    # 计算奖励
    f1_score = reward_calculator.calculate_f1_score(true_labels, predicted_labels)
    logger.info(f"当前F1分数: {f1_score:.4f}")

if __name__ == '__main__':
    main() 