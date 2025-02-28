# ActiveLearner 类实现步骤

## 1. 初始化
- **`__init__` 方法**：接收 `llava_processor`、`unlabeled_dir`、`output_dir` 和 `config` 参数，初始化主动学习器。

## 2. 采样流程
- **`run_sampling` 方法**：运行采样过程，返回选中的文件和统计信息。
- **`_analyze_market_states` 方法**：分析市场状态。
- **`_diversity_sampling` 方法**：多样性采样。
- **`_get_market_states_stats` 方法**：获取市场状态统计信息。

## 3. 不确定性评估
- **`_calculate_uncertainty` 方法**：计算预测的不确定性。
- **`_extract_features` 方法**：提取特征。
- **`uncertainty_sampling` 方法**：基于不确定性进行采样。

## 4. 人工反馈处理
- **`collect_human_feedback` 方法**：收集人工反馈。
- **`adjust_human_labeling_ratio` 方法**：根据性能指标调整人工标注比例。 