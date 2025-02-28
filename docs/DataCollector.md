# DataCollector

## 功能描述
`DataCollector` 类负责从 TradingView API 收集市场数据。它提供了数据验证、保存和备份的功能，确保数据的完整性和可用性。

## 主要方法

### `fetch_chart_data(symbol: str, timeframe: str) -> Dict[str, Any]`
- **参数**:
  - `symbol`: 交易对的符号，例如 'BTCUSDT'。
  - `timeframe`: 数据的时间框架，例如 '1h'。
- **返回**: 返回包含市场数据的字典。

### `validate_data(data: Dict[str, Any]) -> bool`
- **参数**:
  - `data`: 收集到的数据字典。
- **返回**: 如果数据有效，返回 `True`；否则返回 `False`。

### `save_chart_data(data: Dict[str, Any], directory: str) -> str`
- **参数**:
  - `data`: 要保存的数据字典。
  - `directory`: 保存数据的目录。
- **返回**: 返回保存的文件名。

## 示例代码
```python
config = CollectorConfig(
    screenshot_interval=300,
    tradingview_api_url="https://api.tradingview.com",
    tradingview_api_token="test_token",
    backup_enabled=True,
    backup_interval=3600,
    compression_enabled=True,
    max_retries=3,
    retry_delay=5,
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframes=['1h', '4h', '1d'],
    indicators=['RSI', 'MACD', 'BB']
)

collector = DataCollector(config)
chart_data = collector.fetch_chart_data('BTCUSDT', '1h')
if collector.validate_data(chart_data):
    collector.save_chart_data(chart_data, 'data/')
```

## 注意事项
- 确保在使用前配置正确的 API URL 和 Token。
- 处理 API 请求失败的情况，建议使用重试机制。 