#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易图表图像处理脚本
用于准备LLaVA-SBS模型的训练数据
"""

import os
import sys
import yaml
import logging
import argparse
import json
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到PATH以便导入src模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目自定义模块
from src.utils.logging_utils import setup_logging


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


def resize_image(image_path, output_path, target_size=(336, 336)):
    """
    调整图像大小
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径
        target_size: 目标大小，默认为(336, 336)
    """
    try:
        with Image.open(image_path) as img:
            # 转换为RGB格式（处理可能的RGBA或其他格式）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 调整大小
            resized_img = img.resize(target_size, Image.LANCZOS)
            
            # 保存图像
            resized_img.save(output_path)
            
    except Exception as e:
        logging.error(f"处理图像 {image_path} 时出错: {e}")
        raise


def create_prompt_file(prompt_template, metadata, output_path):
    """
    创建提示文件
    
    参数:
        prompt_template: 提示模板
        metadata: 图像元数据
        output_path: 输出文件路径
    """
    try:
        # 替换模板中的占位符
        prompt = prompt_template.format(**metadata)
        
        # 写入提示文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
            
    except Exception as e:
        logging.error(f"创建提示文件 {output_path} 时出错: {e}")
        raise


def process_images(src_dir, output_dir, prompt_templates, image_size=(336, 336), train_ratio=0.8):
    """
    处理图像并创建训练数据
    
    参数:
        src_dir: 源目录
        output_dir: 输出目录
        prompt_templates: 提示模板字典
        image_size: 目标图像大小
        train_ratio: 训练集比例
    """
    # 确保输出目录存在
    train_img_dir = Path(output_dir) / 'processed' / 'train' / 'images'
    train_txt_dir = Path(output_dir) / 'processed' / 'train' / 'texts'
    val_img_dir = Path(output_dir) / 'processed' / 'val' / 'images'
    val_txt_dir = Path(output_dir) / 'processed' / 'val' / 'texts'
    
    for dir_path in [train_img_dir, train_txt_dir, val_img_dir, val_txt_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 加载元数据文件
    metadata_path = Path(src_dir) / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"元数据文件 {metadata_path} 不存在")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)
    
    # 随机打乱并分割
    random.shuffle(all_metadata)
    split_idx = int(len(all_metadata) * train_ratio)
    train_metadata = all_metadata[:split_idx]
    val_metadata = all_metadata[split_idx:]
    
    logging.info(f"总样本数: {len(all_metadata)}, 训练集: {len(train_metadata)}, 验证集: {len(val_metadata)}")
    
    # 处理训练集
    logging.info("处理训练集图像...")
    for item in tqdm(train_metadata):
        # 处理图像
        src_img_path = Path(src_dir) / item['image_path']
        pattern_type = item.get('pattern_type', '未知')
        
        # 确定使用哪个提示模板
        if pattern_type in prompt_templates:
            prompt_template = prompt_templates[pattern_type]
        else:
            prompt_template = prompt_templates['default']
        
        # 输出文件路径
        img_name = Path(item['image_path']).name
        dst_img_path = train_img_dir / img_name
        txt_path = train_txt_dir / f"{Path(img_name).stem}.txt"
        
        # 处理图像和创建提示文件
        resize_image(src_img_path, dst_img_path, image_size)
        create_prompt_file(prompt_template, item, txt_path)
    
    # 处理验证集
    logging.info("处理验证集图像...")
    for item in tqdm(val_metadata):
        # 处理图像
        src_img_path = Path(src_dir) / item['image_path']
        pattern_type = item.get('pattern_type', '未知')
        
        # 确定使用哪个提示模板
        if pattern_type in prompt_templates:
            prompt_template = prompt_templates[pattern_type]
        else:
            prompt_template = prompt_templates['default']
        
        # 输出文件路径
        img_name = Path(item['image_path']).name
        dst_img_path = val_img_dir / img_name
        txt_path = val_txt_dir / f"{Path(img_name).stem}.txt"
        
        # 处理图像和创建提示文件
        resize_image(src_img_path, dst_img_path, image_size)
        create_prompt_file(prompt_template, item, txt_path)
    
    # 创建训练集和验证集的元数据文件
    train_meta_path = Path(output_dir) / 'processed' / 'train' / 'metadata.json'
    val_meta_path = Path(output_dir) / 'processed' / 'val' / 'metadata.json'
    
    with open(train_meta_path, 'w', encoding='utf-8') as f:
        json.dump([{'image_path': f"images/{Path(item['image_path']).name}", 'prompt': prompt_templates.get(item.get('pattern_type', '未知'), prompt_templates['default']).format(**item)} for item in train_metadata], f, ensure_ascii=False, indent=2)
    
    with open(val_meta_path, 'w', encoding='utf-8') as f:
        json.dump([{'image_path': f"images/{Path(item['image_path']).name}", 'prompt': prompt_templates.get(item.get('pattern_type', '未知'), prompt_templates['default']).format(**item)} for item in val_metadata], f, ensure_ascii=False, indent=2)
    
    logging.info(f"数据处理完成，训练集和验证集已保存到 {output_dir}/processed/")


def setup_active_learning(output_dir, image_size=(336, 336)):
    """
    设置主动学习目录
    
    参数:
        output_dir: 输出目录
        image_size: 目标图像大小
    """
    # 创建主动学习目录
    al_img_dir = Path(output_dir) / 'active_learning' / 'images'
    al_txt_dir = Path(output_dir) / 'active_learning' / 'texts'
    
    for dir_path in [al_img_dir, al_txt_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 创建空元数据文件
    metadata_path = Path(output_dir) / 'active_learning' / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump([], f)
    
    logging.info(f"主动学习目录已设置: {Path(output_dir)/'active_learning'}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='交易图表图像处理脚本')
    parser.add_argument('--config', type=str, default='config/data_processing_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--src_dir', type=str,
                       help='源图像目录，覆盖配置文件中的设置')
    parser.add_argument('--output_dir', type=str,
                       help='输出目录，覆盖配置文件中的设置')
    parser.add_argument('--image_size', type=int, nargs=2, default=[336, 336],
                       help='目标图像大小，格式为"宽 高"')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--setup_active_learning', action='store_true',
                       help='是否设置主动学习目录')
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 设置日志
        setup_logging(config)
        logging.info(f"加载配置文件: {args.config}")
        
        # 获取源目录和输出目录
        src_dir = args.src_dir if args.src_dir else config['paths']['source_data_dir']
        output_dir = args.output_dir if args.output_dir else config['paths']['data_dir']
        
        # 确保目录存在
        ensure_directories({'src_dir': src_dir, 'output_dir': output_dir})
        
        # 获取提示模板
        prompt_templates = config.get('prompt_templates', {
            'default': "图片中是否显示了市场结构的突破？前一个高点或低点是否被实体K线突破？请指出具体位置。",
            'structure_break': "图片中是否显示了市场结构的突破？前一个高点或低点是否被实体K线突破？请指出具体位置。",
            'double_top_bottom': "在这张图中，双顶或双底形态是否已确认？如果确认，何时入场？在12345（SBS）序列中双顶或双底形态通常发生在哪个位置？",
            'liquidation': "图片中是一个有效的liquidation吗？在sbs序列中liquidate通常发生在什么地方？",
            'sbs_sequence': "这张图显示的是否是一个有效的SBS序列？请标出12345五个点的位置。",
            'sce': "图片里是一个有效的SCE吗？SCE是什么？",
            'trend': "图片中的sma20和sma200分别是哪条线？如何用这两条线判断市场趋势？",
            'swing': "图片中的swing有什么特点？它代表了什么市场含义？"
        })
        
        # 处理图像
        process_images(
            src_dir, 
            output_dir, 
            prompt_templates, 
            image_size=tuple(args.image_size),
            train_ratio=args.train_ratio
        )
        
        # 如果指定了设置主动学习目录，则设置
        if args.setup_active_learning:
            setup_active_learning(output_dir, image_size=tuple(args.image_size))
        
        logging.info("图像处理完成")
        
    except Exception as e:
        logging.error(f"图像处理过程中发生错误: {str(e)}", exc_info=True)
        raise
        
        
if __name__ == "__main__":
    main() 