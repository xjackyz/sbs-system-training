# SBSPredictor 类实现步骤

## 1. 初始化
- **`__init__` 方法**：接收 `config` 参数，初始化 SBS 预测器。
- **`_init_model_components` 方法**：初始化模型组件。

## 2. 前向传播
- **`forward` 方法**：执行模型的前向传播，返回预测结果。

## 3. 点位预测
- **`predict_points` 方法**：预测 K 线数据中的 SBS 点位。
- **`_prepare_input` 方法**：准备模型输入数据。
- **`_normalize_data` 方法**：归一化数据。

## 4. 预测验证
- **`_validate_predictions` 方法**：验证预测结果的有效性。 