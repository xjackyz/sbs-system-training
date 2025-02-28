import os
import json
import requests
from typing import Dict, Any

class CollectorConfig:
    def __init__(self, screenshot_interval: int, tradingview_api_url: str, tradingview_api_token: str,
                 backup_enabled: bool, backup_interval: int, compression_enabled: bool,
                 max_retries: int, retry_delay: int, symbols: list, timeframes: list, indicators: list):
        self.screenshot_interval = screenshot_interval
        self.tradingview_api_url = tradingview_api_url
        self.tradingview_api_token = tradingview_api_token
        self.backup_enabled = backup_enabled
        self.backup_interval = backup_interval
        self.compression_enabled = compression_enabled
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.symbols = symbols
        self.timeframes = timeframes
        self.indicators = indicators

class DataCollector:
    def __init__(self, config: CollectorConfig):
        self.config = config

    def fetch_chart_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        url = f'{self.config.tradingview_api_url}/chart_data?symbol={symbol}&timeframe={timeframe}'
        headers = {'Authorization': f'Bearer {self.config.tradingview_api_token}'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def validate_data(self, data: Dict[str, Any]) -> bool:
        # 简单的数据验证逻辑
        return 'symbol' in data and 'data' in data

    def save_chart_data(self, data: Dict[str, Any], directory: str) -> str:
        filename = os.path.join(directory, f'{data["symbol"]}_{data["timeframe"]}.json')
        with open(filename, 'w') as f:
            json.dump(data, f)
        return filename 