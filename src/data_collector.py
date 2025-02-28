"""
数据收集器组件
负责从 TradingView API 收集市场数据
"""

import os
import logging
from typing import Dict, List, Optional
import requests
from datetime import datetime, timedelta

from .utils.logger import setup_logger
from .utils.config import load_config

logger = setup_logger('data_collector')

class DataCollector:
    """数据收集器类"""
    
    def __init__(self, api_key: str = None, config: Dict = None):
        """
        初始化数据收集器
        
        Args:
            api_key: TradingView API密钥
            config: 配置参数
        """
        self.api_key = api_key or os.getenv('TRADINGVIEW_API_KEY')
        if not self.api_key:
            raise ValueError('TradingView API密钥未设置')
            
        self.config = config or load_config()
        self.base_url = self.config.get('MIRROR_URL', 'https://api.tradingview.com')
        self.verify_ssl = self.config.get('VERIFY_SSL', True)
        self.timeout = self.config.get('API_TIMEOUT', 30)
        
    def collect_market_data(self, 
                          symbol: str,
                          interval: str = '1d',
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Dict]:
        """
        收集市场数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            收集到的市场数据列表
        """
        try:
            # 设置默认日期范围
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
                
            # 构建请求参数
            params = {
                'symbol': symbol,
                'interval': interval,
                'start_time': int(start_date.timestamp()),
                'end_time': int(end_date.timestamp())
            }
            
            # 发送请求
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.get(
                f'{self.base_url}/market_data',
                params=params,
                headers=headers,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            
            # 检查响应
            response.raise_for_status()
            data = response.json()
            
            logger.info(f'成功收集 {symbol} 的市场数据，共 {len(data)} 条记录')
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f'收集市场数据失败: {str(e)}')
            raise
            
    def validate_data(self, data: List[Dict]) -> bool:
        """
        验证收集到的数据
        
        Args:
            data: 收集到的数据
            
        Returns:
            数据是否有效
        """
        if not data:
            logger.warning('收集到的数据为空')
            return False
            
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for item in data:
            if not all(field in item for field in required_fields):
                logger.warning(f'数据缺少必要字段: {required_fields}')
                return False
                
        return True
        
    def save_data(self, data: List[Dict], filepath: str) -> None:
        """
        保存收集到的数据
        
        Args:
            data: 收集到的数据
            filepath: 保存路径
        """
        try:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f'数据已保存到: {filepath}')
        except Exception as e:
            logger.error(f'保存数据失败: {str(e)}')
            raise 