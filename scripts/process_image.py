#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像处理命令行工具
"""

import os
import sys
import argparse
from pathlib import Path
import contextlib

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image.processor import ImageProcessor
from src.utils.logger import setup_logger

def main():
    """主函数"""
    # 设置日志
    logger = setup_logger('process_image')
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='图像处理工具')
    parser.add_argument('input', help='输入图像路径或包含图像的目录')
    parser.add_argument('--output', '-o', help='输出目录，默认与输入相同')
    parser.add_argument('--preprocess', '-p', action='store_true', help='预处理图像')
    parser.add_argument('--crop', '-c', action='store_true', help='裁剪图表区域')
    parser.add_argument('--extract', '-e', action='store_true', help='提取图表特征')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归处理目录中的所有图像')
    parser.add_argument('--quality-check', '-q', action='store_true', help='检查图像质量')
    parser.add_argument('--no-enhance', '-n', action='store_true', help='不进行图像增强')
    
    args = parser.parse_args()
    
    # 初始化图像处理器
    processor = ImageProcessor()
    
    # 解析输入路径
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        logger.error(f"输入路径不存在: {input_path}")
        return 1
    
    # 设置输出目录
    output_dir = os.path.abspath(args.output) if args.output else None
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 收集要处理的图像文件
    image_files = []
    if os.path.isfile(input_path):
        image_files.append(input_path)
    else:  # 目录
        for root, _, files in os.walk(input_path) if args.recursive else [(input_path, None, os.listdir(input_path))]:
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    image_files.append(os.path.join(root, file))
    
    if not image_files:
        logger.warning(f"未找到图像文件: {input_path}")
        return 0
    
    # 处理图像
    for image_file in image_files:
        logger.info(f"处理图像: {image_file}")
        
        # 如果指定了输出目录，生成输出路径
        if output_dir:
            rel_path = os.path.relpath(image_file, input_path)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = image_file
        
        current_file = image_file
        
        # 检查图像质量
        if args.quality_check:
            quality_ok = processor.check_image_quality(current_file)
            if not quality_ok:
                logger.warning(f"图像质量不合格，跳过处理: {current_file}")
                continue
            logger.info(f"图像质量检查通过: {current_file}")
        
        # 预处理图像
        if args.preprocess:
            processed_file = processor.preprocess_image(current_file, enhance=not args.no_enhance)
            if not processed_file:
                logger.error(f"预处理图像失败: {current_file}")
                continue
            current_file = processed_file
            logger.info(f"预处理完成: {current_file}")
        
        # 裁剪图表区域
        if args.crop:
            cropped_file = processor.crop_chart_area(current_file)
            if not cropped_file:
                logger.error(f"裁剪图表区域失败: {current_file}")
                continue
            current_file = cropped_file
            logger.info(f"裁剪完成: {current_file}")
        
        # 提取图表特征
        if args.extract:
            features = processor.extract_chart_features(current_file)
            if not features:
                logger.error(f"提取图表特征失败: {current_file}")
                continue
            
            # 将特征保存到文件
            feature_file = os.path.splitext(current_file)[0] + '_features.txt'
            with open(feature_file, 'w', encoding='utf-8') as f:
                f.write("图表特征:\n")
                f.write(f"图像尺寸: {features['image_shape']}\n")
                f.write(f"水平线数量: {features['horizontal_lines']}\n")
                f.write(f"垂直线数量: {features['vertical_lines']}\n")
                f.write(f"边缘密度: {features['edge_density']:.4f}\n")
                f.write("纹理特征:\n")
                for key, value in features['texture_features'].items():
                    f.write(f"  {key}: {value:.4f}\n")
            
            logger.info(f"特征提取完成并保存到: {feature_file}")
    
    logger.info(f"处理完成，共处理 {len(image_files)} 个图像文件")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 