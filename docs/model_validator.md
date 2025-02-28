# ModelValidator 类实现步骤

## 1. 初始化
- **`__init__` 方法**：接收 `config` 参数，初始化模型验证器。

## 2. 序列验证
- **`validate_sequence` 方法**：验证模型在数据集上的性能，支持交叉验证。
- **`_validate_fold` 方法**：验证单个折叠的性能。

## 3. 指标计算
- **`_calculate_metrics` 方法**：计算模型输出的性能指标。
- **`_calculate_average_metrics` 方法**：计算多个折叠的平均指标。

## 4. 指标历史
- **`_update_metrics_history` 方法**：更新指标历史记录。
- **`get_metrics_history` 方法**：获取指标历史记录。 