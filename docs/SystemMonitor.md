# SystemMonitor

## 功能描述
`SystemMonitor` 类用于监控系统的性能和状态，包括资源使用情况、错误跟踪和系统健康检查。该类通常与其他组件配合使用，以确保系统的稳定性和性能。

## 主要方法

### `start_monitoring() -> None`
- **描述**: 启动监控系统，开始收集性能数据。

### `track_metrics() -> None`
- **描述**: 跟踪系统的性能指标，如 CPU、内存和 GPU 使用情况。

### `track_error(error: str) -> None`
- **参数**:
  - `error`: 错误信息字符串。
- **描述**: 记录系统中的错误信息。

### `check_components_health() -> Dict[str, Any]`
- **返回**: 返回各个组件的健康状态。

## 示例代码
```python
monitor = SystemMonitor()

# 启动监控
monitor.start_monitoring()

# 跟踪性能指标
monitor.track_metrics()

# 检查组件健康
health_status = monitor.check_components_health()
print(health_status)
```

## 注意事项
- 确保在使用前正确初始化监控系统。
- 定期检查系统健康状态以防止潜在问题。 