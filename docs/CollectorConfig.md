# CollectorConfig

## 功能描述
`CollectorConfig` 类用于配置数据收集器的参数，包括 TradingView API 的 URL、Token 以及数据收集的相关设置。

## 主要属性

### `screenshot_interval`
- **类型**: `int`
- **描述**: 截图的时间间隔（秒）。

### `tradingview_api_url`
- **类型**: `str`
- **描述**: TradingView API 的基础 URL。

### `tradingview_api_token`
- **类型**: `str`
- **描述**: 用于访问 TradingView API 的 Token。

### `backup_enabled`
- **类型**: `bool`
- **描述**: 是否启用数据备份。

### `backup_interval`
- **类型**: `int`
- **描述**: 数据备份的时间间隔（秒）。

### `compression_enabled`
- **类型**: `bool`
- **描述**: 是否启用数据压缩。

### `max_retries`
- **类型**: `int`
- **描述**: 最大重试次数。

### `retry_delay`
- **类型**: `int`
- **描述**: 重试之间的延迟（秒）。

### `symbols`
- **类型**: `list`
- **描述**: 需要收集数据的交易对符号列表。

### `timeframes`
- **类型**: `list`
- **描述**: 数据的时间框架列表。

### `indicators`
- **类型**: `list`
- **描述**: 需要计算的技术指标列表。

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
```

## 注意事项
- 确保在使用前配置正确的 API URL 和 Token。 