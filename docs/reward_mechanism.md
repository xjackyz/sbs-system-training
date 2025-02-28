# RewardMechanism 类实现步骤

## 1. 初始化
- **`__init__` 方法**：接收 `signal_tracker` 和 `config` 参数，初始化奖励机制。

## 2. 奖励计算
- **`calculate_reward` 方法**：根据LLaVA输出和交易结果计算奖励值。
- **`_calculate_profit_score` 方法**：计算利润得分。
- **`_evaluate_sbs_sequence` 方法**：评估SBS序列的质量。
- **`_evaluate_risk_control` 方法**：评估风险控制情况。

## 3. 样本管理
- **`add_benchmark_sample` 方法**：添加基准样本。
- **`remove_benchmark_sample` 方法**：移除基准样本。
- **`reset_daily_stats` 方法**：重置每日统计数据。

## 4. 数据统计
- **`get_stats` 方法**：获取统计信息。
- **`get_trade_stats` 方法**：获取交易统计信息。
- **`calculate_sample_weights` 方法**：计算样本权重。

## 5. 训练数据处理
- **`get_training_data` 方法**：获取训练数据。
- **`get_validation_data` 方法**：获取验证数据。
- **`get_weighted_batch` 方法**：获取加权批次数据。

## 6. 伪标签生成
- **`_generate_pseudo_labels` 方法**：生成伪标签。

## 7. 可视化和优化
- **`visualize_rewards` 方法**：可视化奖励分布。
- **`weighted_loss_function` 方法**：加权损失函数。
- **`apply_rewards` 方法**：将奖励应用到优化器。 