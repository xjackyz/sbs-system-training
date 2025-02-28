"""
渲染管理器模块
用于管理训练过程中的图像渲染
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
from ..utils.taichi_renderer import TaichiKLineRenderer
from ...utils.logger import setup_logger

logger = setup_logger('render_manager')

class RenderManager:
    """渲染管理器类"""
    
    def __init__(self, config: Dict):
        """
        初始化渲染管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.render_config = config.get('render_settings', {})
        
        # 初始化渲染器
        self.renderer = TaichiKLineRenderer(
            width=self.render_config.get('width', 1920),
            height=self.render_config.get('height', 1080)
        )
        
        # 缓存设置
        self.cache_dir = Path(self.render_config.get('cache_dir', 'data/cache/images'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def render_batch(self, 
                    data: pd.DataFrame,
                    batch_indices: np.ndarray,
                    window_size: int) -> Dict[int, str]:
        """
        渲染一批K线图
        
        Args:
            data: K线数据
            batch_indices: 批次索引
            window_size: 窗口大小
            
        Returns:
            索引到图片路径的映射
        """
        image_paths = {}
        
        try:
            for idx in batch_indices:
                # 检查缓存
                cache_path = self.cache_dir / f"kline_{idx:08d}.png"
                if cache_path.exists():
                    image_paths[idx] = str(cache_path)
                    continue
                
                # 获取窗口数据
                start_idx = idx
                end_idx = idx + window_size
                if end_idx > len(data):
                    continue
                    
                window_data = data.iloc[start_idx:end_idx].copy()
                window_data = window_data.reset_index(drop=True)
                
                # 渲染K线图
                self.renderer.render_klines(window_data)
                
                # 添加技术指标
                if self.render_config.get('add_indicators', True):
                    indicators = self._calculate_indicators(window_data)
                    self.renderer.add_technical_indicators(window_data, indicators)
                
                # 保存图像
                self.renderer.save_image(str(cache_path))
                image_paths[idx] = str(cache_path)
                
        except Exception as e:
            logger.error(f"渲染批次时发生错误: {str(e)}")
            raise
            
        return image_paths
        
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        计算技术指标
        
        Args:
            df: K线数据
            
        Returns:
            技术指标字典
        """
        import talib
        indicators = {}
        
        try:
            # 计算移动平均线
            indicators['MA5'] = talib.MA(df['close'], timeperiod=5)
            indicators['MA10'] = talib.MA(df['close'], timeperiod=10)
            indicators['MA20'] = talib.MA(df['close'], timeperiod=20)
            
            # 计算布林带
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
            indicators['BOLL_UPPER'] = upper
            indicators['BOLL_LOWER'] = lower
            
        except Exception as e:
            logger.warning(f"计算技术指标时发生错误: {str(e)}")
            
        return indicators
        
    def clear_cache(self):
        """清理渲染缓存"""
        try:
            for file in self.cache_dir.glob("*.png"):
                file.unlink()
            logger.info("已清理渲染缓存")
        except Exception as e:
            logger.error(f"清理缓存时发生错误: {str(e)}")
            
    def get_cache_size(self) -> int:
        """
        获取缓存大小
        
        Returns:
            缓存中的图片数量
        """
        return len(list(self.cache_dir.glob("*.png")))
        
    def prerender_dataset(self, 
                         data: pd.DataFrame,
                         window_size: int,
                         step_size: int = 1) -> None:
        """
        预渲染整个数据集
        
        Args:
            data: K线数据
            window_size: 窗口大小
            step_size: 步长
        """
        try:
            total_windows = (len(data) - window_size) // step_size + 1
            logger.info(f"开始预渲染 {total_windows} 张图片...")
            
            for i in range(0, len(data) - window_size + 1, step_size):
                self.render_batch(data, np.array([i]), window_size)
                
                if (i // step_size + 1) % 100 == 0:
                    logger.info(f"已完成 {i // step_size + 1}/{total_windows} 张图片的渲染")
                    
            logger.info("预渲染完成")
            
        except Exception as e:
            logger.error(f"预渲染数据集时发生错误: {str(e)}")
            raise 