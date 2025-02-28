#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
K线图渲染脚本
使用Taichi加速的K线图渲染器生成图像
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.taichi_renderer import TaichiKLineRenderer
from src.utils.logger import setup_logger
import talib

logger = setup_logger('kline_renderer')

def parse_args():
    parser = argparse.ArgumentParser(description='渲染K线图')
    parser.add_argument('--input', type=str, required=True,
                      help='输入CSV文件路径')
    parser.add_argument('--output-dir', type=str, default='data/images',
                      help='输出目录路径')
    parser.add_argument('--width', type=int, default=1920,
                      help='图像宽度')
    parser.add_argument('--height', type=int, default=1080,
                      help='图像高度')
    parser.add_argument('--window-size', type=int, default=100,
                      help='每张图片包含的K线数量')
    parser.add_argument('--step-size', type=int, default=20,
                      help='滑动窗口步长')
    parser.add_argument('--add-indicators', action='store_true',
                      help='是否添加技术指标')
    return parser.parse_args()

def calculate_indicators(df: pd.DataFrame) -> dict:
    """
    计算技术指标
    
    Args:
        df: K线数据DataFrame
        
    Returns:
        技术指标字典
    """
    indicators = {}
    
    # 计算移动平均线
    indicators['MA5'] = talib.MA(df['close'], timeperiod=5)
    indicators['MA10'] = talib.MA(df['close'], timeperiod=10)
    indicators['MA20'] = talib.MA(df['close'], timeperiod=20)
    
    # 计算布林带
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    indicators['BOLL_UPPER'] = upper
    indicators['BOLL_LOWER'] = lower
    
    return indicators

def main():
    args = parse_args()
    
    try:
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取CSV数据
        logger.info(f"正在读取数据: {args.input}")
        df = pd.read_csv(args.input)
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV文件必须包含以下列: {required_columns}")
            
        # 计算技术指标
        indicators = calculate_indicators(df) if args.add_indicators else {}
        
        # 初始化渲染器
        renderer = TaichiKLineRenderer(width=args.width, height=args.height)
        
        # 使用滑动窗口渲染K线图
        total_windows = (len(df) - args.window_size) // args.step_size + 1
        logger.info(f"开始渲染 {total_windows} 张K线图...")
        
        for i in range(total_windows):
            start_idx = i * args.step_size
            end_idx = start_idx + args.window_size
            
            if end_idx > len(df):
                break
                
            # 获取当前窗口的数据
            window_df = df.iloc[start_idx:end_idx].copy()
            window_df = window_df.reset_index(drop=True)
            
            # 渲染K线图
            renderer.render_klines(window_df)
            
            # 添加技术指标
            if args.add_indicators:
                window_indicators = {
                    name: data[start_idx:end_idx]
                    for name, data in indicators.items()
                }
                renderer.add_technical_indicators(window_df, window_indicators)
            
            # 保存图像
            output_path = output_dir / f"kline_{i:04d}.png"
            renderer.save_image(str(output_path))
            
            if (i + 1) % 10 == 0:
                logger.info(f"已完成 {i + 1}/{total_windows} 张图片的渲染")
                
        logger.info("渲染完成！")
        logger.info(f"图片已保存到: {output_dir}")
        
    except Exception as e:
        logger.error(f"渲染失败: {str(e)}")
        raise

if __name__ == '__main__':
    main() 