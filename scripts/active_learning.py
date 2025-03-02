#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SBS交易系统主动学习脚本
用于选择最具信息价值的样本进行标注，提高模型效率
"""

import os
import sys
import yaml
import logging
import torch
import argparse
from pathlib import Path
import random
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM

# 添加项目根目录到PATH以便导入src模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目自定义模块
from src.data_utils.sbs_dataset import SBSActiveLearningDataset
from src.training.custom_trainer import SBSActiveLearner
from src.utils.logging_utils import setup_logging


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


def load_config(config_path):
    """
    加载配置文件
    
    参数:
        config_path: 配置文件路径
        
    返回:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {e}")


def ensure_directories(paths):
    """
    确保所有必要的目录都存在
    
    参数:
        paths: 路径配置字典
    """
    for name, path in paths.items():
        if isinstance(path, str) and ('dir' in name or 'path' in name):
            Path(path).mkdir(parents=True, exist_ok=True)
            logging.info(f"确保目录存在: {path}")


def load_model_and_processor(model_path, device='cuda'):
    """
    加载模型和处理器
    
    参数:
        model_path: 模型路径
        device: 设备类型
    
    返回:
        processor, model: 处理器和模型
    """
    logging.info(f"从 {model_path} 加载模型")
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto" if device == 'cuda' else None
    )
    
    if device == 'cuda' and torch.cuda.is_available():
        logging.info(f"模型已加载到CUDA设备")
    else:
        model = model.to('cpu')
        logging.info(f"模型已加载到CPU设备")
    
    model.eval()
    
    return processor, model


def prepare_dataset(config, processor):
    """
    准备数据集
    
    参数:
        config: 配置字典
        processor: 处理器
        
    返回:
        dataset: 主动学习数据集
    """
    logging.info("准备主动学习数据集")
    
    # 创建数据集
    data_dir = Path(config['paths']['data_dir']) / 'active_learning'
    dataset = SBSActiveLearningDataset(
        data_dir=data_dir,
        processor=processor,
        image_size=config['model']['image_size'],
        max_length=config['model']['max_length']
    )
    
    logging.info(f"加载了 {len(dataset)} 个样本")
    
    return dataset


def run_active_learning(config, model, processor, dataset, args):
    """
    运行主动学习过程
    
    参数:
        config: 配置字典
        model: 模型
        processor: 处理器
        dataset: 数据集
        args: 命令行参数
    """
    logging.info("开始主动学习过程")
    
    # 创建主动学习器
    active_learner = SBSActiveLearner(
        model=model,
        dataset=dataset,
        processor=processor,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # 选择样本
    selected_indices = active_learner.select_samples(
        num_samples=args.num_samples,
        strategy=args.strategy
    )
    
    # 输出选择的样本
    logging.info(f"选择了 {len(selected_indices)} 个样本进行标注")
    
    # 获取并保存选择的样本信息
    selected_samples = []
    for idx in selected_indices:
        metadata = dataset.get_sample_metadata(idx)
        selected_samples.append(metadata)
        logging.info(f"样本 {idx}: {metadata['image_path']}")
    
    # 保存选择的样本列表到文件
    output_file = Path(config['paths']['data_dir']) / 'active_learning' / 'selected_samples.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in selected_indices:
            f.write(f"{idx}\n")
    
    logging.info(f"选择的样本索引已保存到 {output_file}")
    
    # 如果指定了标记标志，则将选择的样本标记为已标记
    if args.mark_as_labeled:
        dataset.mark_as_labeled(selected_indices)
        logging.info(f"已将选择的样本标记为已标记")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='SBS主动学习脚本')
    parser.add_argument('--config', type=str, default='config/active_learning_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model_path', type=str,
                       help='模型路径，覆盖配置文件中的设置')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='要选择的样本数量')
    parser.add_argument('--strategy', type=str, default='uncertainty', choices=['uncertainty', 'random'],
                       help='样本选择策略')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批处理大小')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='设备类型')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    parser.add_argument('--mark_as_labeled', action='store_true',
                       help='将选择的样本标记为已标记')
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 设置日志
        setup_logging(config)
        logging.info(f"加载配置文件: {args.config}")
        
        # 设置随机种子
        seed = args.seed if args.seed is not None else config.get('seed', 42)
        set_random_seeds(seed)
        logging.info(f"设置随机种子: {seed}")
        
        # 确保目录存在
        ensure_directories(config.get('paths', {}))
        
        # 如果命令行指定了模型路径，则覆盖配置中的设置
        if args.model_path:
            config['model']['path'] = args.model_path
        
        # 加载模型和处理器
        processor, model = load_model_and_processor(
            config['model']['path'], 
            device=args.device
        )
        
        # 准备数据集
        dataset = prepare_dataset(config, processor)
        
        # 运行主动学习
        run_active_learning(config, model, processor, dataset, args)
        
        logging.info("主动学习过程完成")
        
    except Exception as e:
        logging.error(f"主动学习过程中发生错误: {str(e)}", exc_info=True)
        raise
        
        
if __name__ == "__main__":
    main() 