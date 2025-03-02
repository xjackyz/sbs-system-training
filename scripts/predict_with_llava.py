#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用微调后的LLaVA-SBS模型分析交易图表
该脚本加载微调后的模型，处理输入的交易图表图像，并生成分析结果
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt

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
        logging.info("模型已加载到CUDA设备")
    else:
        model = model.to('cpu')
        logging.info("模型已加载到CPU设备")
    
    model.eval()
    
    return processor, model


def process_image(image_path, processor, image_size=(336, 336)):
    """
    处理图像
    
    参数:
        image_path: 图像路径
        processor: 处理器
        image_size: 图像大小
        
    返回:
        处理后的图像
    """
    try:
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # 调整图像大小
        if image_size:
            image = image.resize(image_size, Image.LANCZOS)
            
        return image
        
    except Exception as e:
        logging.error(f"处理图像失败: {e}")
        raise


def analyze_chart(image, processor, model, prompt_template=None):
    """
    分析图表
    
    参数:
        image: 图像
        processor: 处理器
        model: 模型
        prompt_template: 提示模板
        
    返回:
        分析结果
    """
    # 设置默认提示模板
    if prompt_template is None:
        prompt_template = "图片中是否显示了市场结构的突破？前一个高点或低点是否被实体K线突破？请指出具体位置。"
    
    try:
        # 使用处理器处理图像和文本
        inputs = processor(
            text=prompt_template,
            images=image,
            return_tensors="pt"
        ).to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # 解码输出
        generated_text = processor.batch_decode(
            outputs, 
            skip_special_tokens=True
        )[0]
        
        # 提取回答 (去除提示部分)
        if prompt_template in generated_text:
            answer = generated_text.split(prompt_template)[1].strip()
        else:
            answer = generated_text
            
        return answer
        
    except Exception as e:
        logging.error(f"分析图表失败: {e}")
        raise


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='使用LLaVA-SBS模型分析交易图表')
    parser.add_argument('--config', type=str, default='config/predict_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model_path', type=str,
                       help='模型路径，覆盖配置文件中的设置')
    parser.add_argument('--image_path', type=str, required=True,
                       help='图像路径')
    parser.add_argument('--prompt', type=str,
                       help='分析提示')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='设备类型')
    parser.add_argument('--output_file', type=str,
                       help='输出文件路径')
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 设置日志
        setup_logging(config)
        logging.info(f"加载配置文件: {args.config}")
        
        # 如果命令行指定了模型路径，则覆盖配置中的设置
        model_path = args.model_path if args.model_path else config['model']['path']
        
        # 加载模型和处理器
        processor, model = load_model_and_processor(model_path, device=args.device)
        
        # 处理图像
        image = process_image(
            args.image_path, 
            processor,
            image_size=tuple(config['model']['image_size'])
        )
        
        # 获取提示
        prompt = args.prompt if args.prompt else config.get('prompt_template', {}).get('default', None)
        
        # 分析图表
        result = analyze_chart(image, processor, model, prompt)
        
        # 打印结果
        print("分析结果:")
        print(result)
        
        # 如果指定了输出文件，则保存结果
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            logging.info(f"结果已保存到 {args.output_file}")
        
    except Exception as e:
        logging.error(f"分析过程中发生错误: {str(e)}", exc_info=True)
        raise
        
        
if __name__ == "__main__":
    main() 