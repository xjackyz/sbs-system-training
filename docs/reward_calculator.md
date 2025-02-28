# RewardCalculator 类实现步骤

## 1. 初始化
- **`__init__` 方法**：接收 `config`、`alpha` 和 `beta` 参数，初始化奖励计算器。

## 2. 奖励计算
- **`calculate_reward` 方法**：计算预测和人工标注的奖励值。
- **`_calculate_dynamic_weight` 方法**：计算动态权重。
- **`_calculate_accuracy_reward` 方法**：计算准确率奖励。
- **`_calculate_profit_reward` 方法**：计算利润奖励。
- **`_apply_penalties` 方法**：应用惩罚项。

## 3. 数据管理
- **`add_human_label` 方法**：添加人工标注数据。
- **`add_model_prediction` 方法**：添加模型预测数据。
- **`get_training_stats` 方法**：获取训练统计信息。

## 4. 趋势分析
- **`_calculate_trend` 方法**：计算数值趋势。
- **`_analyze_penalties` 方法**：分析惩罚情况。

## 5. 历史记录
- **`_update_history` 方法**：更新历史记录。
- **`save_state` 方法**：保存状态。
- **`load_state` 方法**：加载状态。
- **`save_stats` 方法**：保存统计信息。

## 6. 统计指标
- **`get_current_stats` 方法**：获取当前统计信息。
- **`calculate_f1_score` 方法**：计算F1分数。
- **`calculate_compound_reward` 方法**：计算复合奖励。 