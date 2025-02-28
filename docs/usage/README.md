# SBS系统使用指南

本文档提供SBS交易分析系统的使用说明，包括基本操作和常见使用场景。

## 系统功能概述

SBS交易分析系统提供以下核心功能：

1. 自动化图表分析
2. 交易信号识别和生成
3. 实时通知机制
4. 历史数据分析和回测
5. 自定义规则和策略

## 快速开始

### 基本分析流程

1. **准备图表图像**
   - 截取或保存需要分析的交易图表
   - 支持的格式：PNG, JPG, JPEG
   - 建议分辨率：至少1280x720

2. **上传图表**
   ```bash
   # 使用命令行工具
   python scripts/process_image.py --image_path "/path/to/chart.png" --output "/path/to/results"
   
   # 或者通过Web界面上传
   # 访问 http://localhost:5000/upload
   ```

3. **查看分析结果**
   - 命令行工具会输出结果到指定路径
   - Web界面会直接显示分析结果

## 使用场景示例

### 场景1：单一图表分析

```bash
# 分析单个图表
python scripts/process_image.py --image_path "examples/btc_daily.png" --verbose
```

输出将包含：
- 市场趋势分析
- 主要支撑/阻力位
- 交易信号建议
- 置信度分数

### 场景2：设置定时监控

1. 编辑配置文件 `config/system/system_config.yaml`：
```yaml
monitoring:
  enabled: true
  interval: 60  # 分钟
  sources:
    - name: "BTC/USD"
      url: "https://www.tradingview.com/chart/..."
      screenshot: true
    - name: "ETH/USD"
      url: "https://www.tradingview.com/chart/..."
      screenshot: true
```

2. 启动监控服务：
```bash
python -m src.monitoring.start
```

3. 检查任务状态：
```bash
python -m src.monitoring.status
```

### 场景3：使用Discord机器人

1. 设置Discord机器人（参见[机器人设置](bot-setup.md)）

2. 在Discord频道中使用：
```
/analyze chart.png
```

3. 机器人将回复分析结果

## 自定义系统

### 自定义提示词模板

编辑 `config/sbs_prompt.py` 文件：
```python
CHART_ANALYSIS_TEMPLATE = """
分析以下交易图表:
1. 识别当前市场趋势
2. 找出关键支撑和阻力位
3. 识别可能的交易信号
4. 提供交易建议

{additional_instructions}
"""
```

### 自定义信号过滤规则

编辑 `config/system/system_config.yaml`：
```yaml
signal_filters:
  min_confidence: 0.75
  require_confirmation: true
  allowed_signals:
    - "BUY"
    - "SELL"
    - "HOLD"
  blacklist_patterns:
    - "unclear"
    - "uncertain"
```

## 高级功能

### 运行回测

```bash
python scripts/run_backtest.py --strategy "ma_crossover" --period "2023-01-01,2023-12-31"
```

### 生成性能报告

```bash
python -m src.analysis.generate_report --period "last_month" --format "pdf"
```

## 故障排除

### 常见问题

**问题**: 分析结果不准确或质量差

**解决方案**:
- 确保图表图像清晰可读
- 尝试使用更高分辨率的图像
- 检查是否包含了足够的上下文信息（如时间框架、指标）

**问题**: 服务响应缓慢

**解决方案**:
- 检查GPU使用情况
- 调整系统配置中的批处理大小
- 考虑使用更小的模型版本

## 相关资源

- [常见问题解答](faq.md)
- [API使用指南](../api/README.md)
- [配置参考](configuration.md)
- [故障排除指南](troubleshooting.md) 