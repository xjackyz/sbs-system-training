"""
使用Taichi实现的高性能K线图渲染模块
"""

import taichi as ti
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd

ti.init(arch=ti.gpu)  # 使用GPU加速

@ti.data_oriented
class TaichiKLineRenderer:
    def __init__(self, width: int = 1920, height: int = 1080):
        """
        初始化K线渲染器
        
        Args:
            width: 图像宽度
            height: 图像高度
        """
        self.width = width
        self.height = height
        
        # 定义图像缓冲区
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.depth_buffer = ti.field(dtype=ti.f32, shape=(width, height))
        
        # 颜色定义
        self.up_color = ti.Vector([1.0, 0.2, 0.2])  # 红色
        self.down_color = ti.Vector([0.2, 1.0, 0.2])  # 绿色
        self.grid_color = ti.Vector([0.2, 0.2, 0.2])  # 网格颜色
        self.bg_color = ti.Vector([0.1, 0.1, 0.1])    # 背景颜色
        
        # 渲染参数
        self.padding = 0.1  # 边距比例
        self.grid_size = 50  # 网格大小
        
    @ti.kernel
    def clear_buffers(self):
        """清除缓冲区"""
        for i, j in self.pixels:
            self.pixels[i, j] = self.bg_color
            self.depth_buffer[i, j] = 1.0
            
    @ti.func
    def draw_line(self, start: ti.Vector, end: ti.Vector, color: ti.Vector):
        """
        绘制线段
        
        Args:
            start: 起点坐标
            end: 终点坐标
            color: 线段颜色
        """
        x1, y1 = start
        x2, y2 = end
        steep = abs(y2 - y1) > abs(x2 - x1)
        
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
            
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            
        dx = x2 - x1
        dy = abs(y2 - y1)
        error = dx / 2.0
        y = y1
        ystep = 1 if y1 < y2 else -1
        
        for x in range(int(x1), int(x2) + 1):
            if steep:
                if 0 <= y < self.width and 0 <= x < self.height:
                    self.pixels[y, x] = color
            else:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.pixels[x, y] = color
            error -= dy
            if error < 0:
                y += ystep
                error += dx
                
    @ti.kernel
    def draw_candlestick(self, 
                        x: ti.f32, 
                        open_price: ti.f32, 
                        high_price: ti.f32, 
                        low_price: ti.f32, 
                        close_price: ti.f32,
                        price_scale: ti.f32,
                        x_scale: ti.f32):
        """
        绘制单个K线
        
        Args:
            x: x坐标
            open_price: 开盘价
            high_price: 最高价
            low_price: 最低价
            close_price: 收盘价
            price_scale: 价格缩放因子
            x_scale: x轴缩放因子
        """
        # 计算坐标
        x_pos = int(x * x_scale)
        open_y = int(open_price * price_scale)
        high_y = int(high_price * price_scale)
        low_y = int(low_price * price_scale)
        close_y = int(close_price * price_scale)
        
        # 选择颜色
        color = self.up_color if close_price >= open_price else self.down_color
        
        # 绘制影线
        self.draw_line(ti.Vector([x_pos, low_y]), ti.Vector([x_pos, high_y]), color)
        
        # 绘制实体
        body_start = min(open_y, close_y)
        body_end = max(open_y, close_y)
        body_width = max(1, int(x_scale * 0.8))  # K线宽度
        
        for i in range(-body_width//2, body_width//2 + 1):
            if 0 <= x_pos + i < self.width:
                self.draw_line(
                    ti.Vector([x_pos + i, body_start]),
                    ti.Vector([x_pos + i, body_end]),
                    color
                )
                
    def render_klines(self, df: pd.DataFrame) -> np.ndarray:
        """
        渲染K线图
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            渲染后的图像数组
        """
        # 清除缓冲区
        self.clear_buffers()
        
        # 计算缩放因子
        price_min = df[['open', 'high', 'low', 'close']].min().min()
        price_max = df[['open', 'high', 'low', 'close']].max().max()
        price_range = price_max - price_min
        
        effective_height = self.height * (1 - 2 * self.padding)
        effective_width = self.width * (1 - 2 * self.padding)
        
        price_scale = effective_height / price_range
        x_scale = effective_width / len(df)
        
        # 绘制网格
        self.draw_grid(price_min, price_max, price_scale)
        
        # 绘制K线
        for i, row in df.iterrows():
            x = i
            self.draw_candlestick(
                x,
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                price_scale,
                x_scale
            )
            
        # 返回渲染结果
        return self.pixels.to_numpy()
        
    @ti.kernel
    def draw_grid(self, price_min: ti.f32, price_max: ti.f32, price_scale: ti.f32):
        """
        绘制网格线
        
        Args:
            price_min: 最小价格
            price_max: 最大价格
            price_scale: 价格缩放因子
        """
        # 绘制水平网格线
        price_step = (price_max - price_min) / 10
        for i in range(11):
            y = int((price_min + i * price_step) * price_scale)
            if 0 <= y < self.height:
                self.draw_line(
                    ti.Vector([0, y]),
                    ti.Vector([self.width-1, y]),
                    self.grid_color
                )
                
        # 绘制垂直网格线
        x_step = self.width / 10
        for i in range(11):
            x = int(i * x_step)
            self.draw_line(
                ti.Vector([x, 0]),
                ti.Vector([x, self.height-1]),
                self.grid_color
            )
            
    def add_technical_indicators(self, 
                               df: pd.DataFrame,
                               indicators: Dict[str, np.ndarray]):
        """
        添加技术指标
        
        Args:
            df: K线数据
            indicators: 技术指标数据字典
        """
        # 为每个指标分配不同颜色
        indicator_colors = {
            'MA5': ti.Vector([1.0, 0.5, 0.0]),  # 橙色
            'MA10': ti.Vector([0.0, 0.5, 1.0]), # 蓝色
            'MA20': ti.Vector([1.0, 1.0, 0.0]), # 黄色
            'BOLL_UPPER': ti.Vector([0.8, 0.4, 0.8]), # 紫色
            'BOLL_LOWER': ti.Vector([0.8, 0.4, 0.8])  # 紫色
        }
        
        # 绘制每个指标
        for name, data in indicators.items():
            if name in indicator_colors:
                self._draw_indicator_line(df, data, indicator_colors[name])
                
    @ti.func
    def _draw_indicator_line(self, df: pd.DataFrame, data: np.ndarray, color: ti.Vector):
        """
        绘制指标线
        
        Args:
            df: K线数据
            data: 指标数据
            color: 线条颜色
        """
        price_min = df[['open', 'high', 'low', 'close']].min().min()
        price_range = df[['open', 'high', 'low', 'close']].max().max() - price_min
        price_scale = self.height / price_range
        x_scale = self.width / len(df)
        
        for i in range(len(data)-1):
            if not ti.math.isnan(data[i]) and not ti.math.isnan(data[i+1]):
                start = ti.Vector([
                    int(i * x_scale),
                    int((data[i] - price_min) * price_scale)
                ])
                end = ti.Vector([
                    int((i+1) * x_scale),
                    int((data[i+1] - price_min) * price_scale)
                ])
                self.draw_line(start, end, color)
                
    def save_image(self, filepath: str):
        """
        保存渲染结果
        
        Args:
            filepath: 保存路径
        """
        import cv2
        img = (self.pixels.to_numpy() * 255).astype(np.uint8)
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 