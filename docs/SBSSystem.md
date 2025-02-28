# SBSSystem

## 功能描述
`SBSSystem` 类是整个系统的管理类，负责协调各个组件的工作，包括数据收集、分析、信号生成和监控。该类提供了系统的主要功能和接口。

## 主要方法

### `__init__(config: SystemConfig, collector: DataCollector, analyzer: LLaVAAnalyzer, generator: SignalGenerator, monitor: SystemMonitor)`
- **参数**:
  - `config`: 系统配置对象。
  - `collector`: 数据收集器实例。
  - `analyzer`: 分析器实例。
  - `generator`: 信号生成器实例。
  - `monitor`: 监控系统实例。

### `startup() -> None`
- **描述**: 启动系统，初始化各个组件并开始数据收集。

### `shutdown() -> None`
- **描述**: 关闭系统，清理资源并停止所有组件。

### `run_once() -> None`
- **描述**: 执行一次主循环，收集数据、分析数据并生成信号。

### `get_system_state() -> Dict[str, Any]`
- **返回**: 返回系统的当前状态，包括各个组件的状态信息。

## 示例代码
```python
config = SystemConfig(
    run_mode='test',
    debug=True,
    log_level='INFO',
    use_gpu=False,
    num_workers=2,
    data_dir='test_data',
    model_dir='test_models',
    log_dir='test_logs'
)

system = SBSSystem(
    config=config,
    collector=DataCollector(config),
    analyzer=LLaVAAnalyzer(),
    generator=SignalGenerator(),
    monitor=SystemMonitor()
)

# 启动系统
system.startup()

# 执行一次主循环
system.run_once()
```

## 注意事项
- 确保在使用前正确初始化所有组件。
- 定期检查系统状态以确保正常运行。 