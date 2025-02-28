import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
import time
import torch
from pathlib import Path
import logging
import random
import shutil
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

from src.data.chart_generator import ChartGenerator


class CSVToChartProcessor:
    """
    从CSV文件生成1分钟K线图并处理成LLaVA模型可用的格式
    用于自监督学习过程
    """
    
    def __init__(self, output_dir: str = 'data/charts', window_size: int = 100):
        """
        初始化处理器
        
        Args:
            output_dir: 输出目录
            window_size: 每个图表包含的K线数量
        """
        self.output_dir = output_dir
        self.window_size = window_size
        
        # 默认配置
        self.config = {
            'output_dir': output_dir,
            'sbs_collection_dir': '/home/easyai/桌面/sbs_system/sbs_sequences',
            'window_size': window_size,  # 每个图表包含的K线数量
            'stride': 1,  # 滑动窗口步长(1表示每分钟生成一张图)
            'save_ratio': 0.01,  # 保存1/100的SBS序列图片
            'indicators': ['sma20', 'sma200'],  # 要添加的技术指标
            'chart_config': {
                'width': 10,
                'height': 6,
                'dpi': 100,
                'style': 'yahoo',
                'volume': False,  # 不显示成交量
                'custom_style': {
                    'base_mpf_style': 'yahoo',
                    'facecolor': 'white',
                    'edgecolor': 'black',
                    'up_color': 'darkgray',  # 更深的灰色作为阳线颜色
                    'down_color': 'black',
                    'sma20_color': 'lightgray',
                    'sma200_color': 'darkgray',
                    'volume_up_color': 'darkgray',
                    'volume_down_color': 'black'
                }
            }
        }
        
        self.logger = logging.getLogger('csv_to_chart')
        
        # 创建输出目录
        if os.path.exists(self.config['output_dir']):
            if os.path.isfile(self.config['output_dir']):
                os.remove(self.config['output_dir'])  # 删除文件
            else:
                self.logger.info(f"输出目录已存在: {self.config['output_dir']}")
        else:
            os.makedirs(self.config['output_dir'], exist_ok=True)

        # 检查output_dir是否为文件
        if os.path.isfile(self.config['output_dir']):
            raise ValueError(f"路径已存在且为文件: {self.config['output_dir']}")

        # 检查sbs_collection_dir是否为文件
        if os.path.isfile(self.config['sbs_collection_dir']):
            raise ValueError(f"路径已存在且为文件: {self.config['sbs_collection_dir']}")

        # 创建sbs_collection_dir
        if os.path.exists(self.config['sbs_collection_dir']):
            if os.path.isfile(self.config['sbs_collection_dir']):
                os.remove(self.config['sbs_collection_dir'])  # 删除文件
            else:
                self.logger.info(f"SBS收集目录已存在: {self.config['sbs_collection_dir']}")
        else:
            os.makedirs(self.config['sbs_collection_dir'], exist_ok=True)
        
        # 初始化图表生成器
        self.chart_generator = ChartGenerator({
            'chart': self.config['chart_config'],
            'output': {
                'dir': self.config['output_dir'],
                'format': 'png'
            }
        })
    
    def create_charts_from_csv(self, csv_path: str, stride: int = 1, callback=None) -> List[str]:
        """
        从CSV文件生成K线图
        
        Args:
            csv_path: CSV文件路径
            stride: 滑动窗口步长
            callback: 进度回调函数
            
        Returns:
            生成的图表路径列表
        """
        self.logger.info(f"从CSV生成图表：{csv_path}")
        
        # 确保输出目录存在
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 生成所有图表
        self.logger.info(f"开始生成所有图表，窗口大小: {self.config['window_size']}, 步长: {stride}")
        start_time = time.time()
        
        chart_paths = self.chart_generator.generate_sequence_charts(
            csv_path,
            window_size=self.config['window_size'],
            stride=stride,
            output_dir=self.config['output_dir'],
            indicators=self.config['indicators']  # 使用配置中的指标（SMA20和SMA200）
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info(f"完成图表生成，共 {len(chart_paths)} 个图表，耗时 {total_time:.2f} 秒")
        
        return chart_paths
    
    def create_chart_from_dataframe(self, df: pd.DataFrame) -> str:
        """
        从DataFrame生成单个K线图
        
        Args:
            df: DataFrame格式的K线数据
            
        Returns:
            生成的图表路径
        """
        if df.empty:
            self.logger.warning("传入的DataFrame为空，无法生成图表")
            return None
        
        try:
            # 确保日期列为索引
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'datetime' in df.columns:
                    df['date'] = pd.to_datetime(df['datetime'])
                    df.set_index('date', inplace=True)
                else:
                    self.logger.error("DataFrame缺少日期列")
                    return None
            
            # 生成唯一文件名
            timestamp = int(time.time() * 1000)
            random_suffix = random.randint(1000, 9999)
            filename = f"chart_{timestamp}_{random_suffix}.png"
            output_path = os.path.join(self.config['output_dir'], filename)
            
            # 生成图表
            chart_path = self.chart_generator.generate_chart(
                df,
                output_path=output_path,
                title=f"NQ K线图 {df.index[0].strftime('%Y-%m-%d %H:%M')} - {df.index[-1].strftime('%Y-%m-%d %H:%M')}",
                indicators=self.config['indicators']  # 使用配置中的指标（SMA20和SMA200）
            )
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"从DataFrame生成图表失败: {str(e)}")
            return None
    
    def process_charts(self, chart_paths: List[str], processor=None) -> List[Dict]:
        """
        处理图表，识别特征
        
        Args:
            chart_paths: 图表路径列表
            processor: 可选的处理器（如LLaVA模型）
            
        Returns:
            处理结果列表
        """
        results = []
        
        for chart_path in chart_paths:
            try:
                # 加载图像
                image = Image.open(chart_path)
                
                # 如果提供了处理器，使用处理器处理图像
                if processor:
                    result = processor.process_image(image, chart_path)
                    if result:
                        results.append(result)
                else:
                    # 如果没有处理器，只返回图表路径
                    results.append({"path": chart_path})
                    
            except Exception as e:
                self.logger.error(f"处理图表失败 {chart_path}: {str(e)}")
        
        return results
        
    def detect_sbs_sequence(self, chart_path: str, model: Any) -> Tuple[bool, Dict]:
        """
        使用LLaVA模型检测图表中的SBS序列
        
        Args:
            chart_path: 图表路径
            model: LLaVA模型实例
            
        Returns:
            has_sequence: 是否包含完整SBS序列
            prediction: 模型预测结果
        """
        # 加载图像
        image = Image.open(chart_path)
        
        # 使用LLaVA模型进行预测
        prediction = model.predict_chart(image)
        
        # 判断是否包含完整SBS序列
        has_complete_sequence = self._has_complete_sbs_sequence(prediction)
        
        return has_complete_sequence, prediction
    
    def _has_complete_sbs_sequence(self, prediction: Dict) -> bool:
        """
        判断预测结果是否包含完整的SBS序列
        
        Args:
            prediction: 模型预测结果
            
        Returns:
            是否包含完整SBS序列
        """
        # 检查是否包含点1到点4
        required_points = ['point1', 'point2', 'point3', 'point4']
        
        # 所有必需的点都存在且有位置信息
        if all(point in prediction and prediction[point] is not None for point in required_points):
            return True
        
        return False
    
    def save_sbs_sequence_chart(self, chart_path: str, prediction: Dict):
        """
        保存包含完整SBS序列的图表，用于后续微调
        
        Args:
            chart_path: 图表路径
            prediction: 模型预测结果
        """
        # 提取文件名
        file_name = os.path.basename(chart_path)
        
        # 创建目标路径
        target_path = os.path.join(self.config['sbs_collection_dir'], file_name)
        
        # 复制文件
        shutil.copy(chart_path, target_path)
        
        # 保存预测结果为JSON
        json_path = os.path.join(self.config['sbs_collection_dir'], 
                                os.path.splitext(file_name)[0] + '_prediction.json')
        
        with open(json_path, 'w') as f:
            import json
            json.dump(prediction, f, indent=2)
        
        self.logger.info(f"已保存SBS序列图表: {target_path}")
    
    def process_charts_with_model(self, chart_paths: List[str], model: Any, signal_tracker: Any = None):
        """
        使用LLaVA模型处理所有图表，检测SBS序列并进行跟踪
        
        Args:
            chart_paths: 图表路径列表
            model: LLaVA模型实例
            signal_tracker: 信号跟踪器实例
            
        Returns:
            sbs_charts: 包含SBS序列的图表路径列表
        """
        sbs_charts = []
        total_charts = len(chart_paths)
        
        self.logger.info(f"开始处理 {total_charts} 个图表")
        
        for i, chart_path in enumerate(chart_paths):
            # 每100张图表打印一次进度
            if i % 100 == 0:
                self.logger.info(f"处理进度: {i}/{total_charts} ({i/total_charts*100:.2f}%)")
            
            # 检测SBS序列
            has_sequence, prediction = self.detect_sbs_sequence(chart_path, model)
            
            # 如果包含SBS序列
            if has_sequence:
                sbs_charts.append(chart_path)
                
                # 根据比例决定是否保存
                if random.random() < self.config['save_ratio']:
                    self.save_sbs_sequence_chart(chart_path, prediction)
                
                # 如果提供了信号跟踪器，记录信号
                if signal_tracker:
                    # 提取图表数据
                    chart_data = {
                        'path': chart_path,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # 记录信号
                    confidence = prediction.get('confidence', 0.8)
                    signal_id = signal_tracker.record_signal(chart_data, prediction, confidence)
                    
                    self.logger.info(f"记录信号: {signal_id}")
        
        self.logger.info(f"处理完成，共找到 {len(sbs_charts)} 个包含SBS序列的图表")
        return sbs_charts 