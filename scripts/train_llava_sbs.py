#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaVA-SBS模型微调训练脚本
用于对LLaVA模型进行交易知识微调，学习识别SBS交易序列和形态
"""

import os
import sys
import yaml
import logging
import torch
from pathlib import Path
import argparse
from datetime import datetime
import random
import numpy as np
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import datasets
from torch.utils.data import DataLoader

# 添加项目根目录到PATH以便导入src模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目自定义模块
from src.data_utils.sbs_dataset import SBSImageTextDataset
from src.training.custom_trainer import SBSTrainer
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


def prepare_llava_model(config):
    """
    准备LLaVA模型，加载预训练权重
    
    参数:
        config: 配置字典
        
    返回:
        处理器、模型
    """
    logging.info("加载LLaVA模型和处理器...")
    
    # 量化配置（可选）
    quantization_config = None
    if config.get('training', {}).get('quantization', {}).get('enabled', False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        config['model']['path'],
        trust_remote_code=True
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['path'],
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # 配置LoRA
    if config.get('training', {}).get('lora', {}).get('enabled', True):
        logging.info("应用LoRA微调配置...")
        lora_config = LoraConfig(
            r=config['training']['lora'].get('r', 16),
            lora_alpha=config['training']['lora'].get('alpha', 32),
            lora_dropout=config['training']['lora'].get('dropout', 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config['training']['lora'].get('target_modules', ["q_proj", "v_proj"])
        )
        model = get_peft_model(model, lora_config)
        
    return processor, model


def prepare_datasets(config, processor):
    """
    准备训练和验证数据集
    
    参数:
        config: 配置字典
        processor: 模型处理器
        
    返回:
        训练集、验证集
    """
    logging.info("准备训练和验证数据集...")
    
    # 创建数据集
    train_dataset = SBSImageTextDataset(
        data_dir=config['paths']['data_dir'] + '/processed/train',
        processor=processor,
        image_size=config['model']['image_size'],
        max_length=config['model']['max_length']
    )
    
    val_dataset = SBSImageTextDataset(
        data_dir=config['paths']['data_dir'] + '/processed/val',
        processor=processor,
        image_size=config['model']['image_size'],
        max_length=config['model']['max_length']
    )
    
    logging.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def train_model(config, model, processor, train_dataset, val_dataset):
    """
    训练模型
    
    参数:
        config: 配置字典
        model: 模型
        processor: 处理器
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        
    返回:
        训练后的模型
    """
    logging.info("开始模型训练...")
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=config['paths']['model_dir'] + '/checkpoints',
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        num_train_epochs=config['training']['epochs'],
        lr_scheduler_type=config['training']['scheduler']['type'],
        warmup_steps=config['training']['scheduler']['warmup_steps'],
        logging_dir=config['paths']['log_dir'],
        logging_steps=config['logging']['log_every_n_steps'],
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        report_to=config['tracking'].get('report_to', ["tensorboard"]),
    )
    
    # 创建训练器
    trainer = SBSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
    )
    
    # 开始训练
    logging.info("开始微调过程...")
    trainer.train()
    
    # 保存最终模型
    final_model_path = config['paths']['model_dir'] + '/final'
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    logging.info(f"模型已保存到 {final_model_path}")
    
    return model


def main():
    """
    主函数，执行LLaVA-SBS模型的微调
    """
    parser = argparse.ArgumentParser(description='LLaVA-SBS模型微调脚本')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子，若未设置则使用配置中的种子')
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 覆盖配置中的特定参数
        if args.seed is not None:
            config['seed'] = args.seed
        
        # 设置日志
        setup_logging(config)
        logging.info(f"加载配置文件: {args.config}")
        
        # 设置随机种子
        seed = config.get('seed', 42)
        set_random_seeds(seed)
        logging.info(f"设置随机种子: {seed}")
        
        # 确保目录存在
        ensure_directories(config.get('paths', {}))
        
        # 准备模型
        processor, model = prepare_llava_model(config)
        
        # 准备数据集
        train_dataset, val_dataset = prepare_datasets(config, processor)
        
        # 训练模型
        trained_model = train_model(config, model, processor, train_dataset, val_dataset)
        
        logging.info("LLaVA-SBS模型微调完成！")
        
    except Exception as e:
        logging.error(f"微调过程中发生错误: {str(e)}", exc_info=True)
        raise
        
        
if __name__ == "__main__":
    main() 