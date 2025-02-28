# LearningVisualizer 类实现步骤

## 1. 初始化
- **`__init__` 方法**：接收 `save_dir` 参数，初始化可视化器。

## 2. 学习曲线
- **`plot_learning_curves` 方法**：绘制学习曲线。
- **`_load_stats_data` 方法**：加载统计数据。
- **`_plot_accuracy_curves` 方法**：绘制准确率曲线。
- **`_plot_profit_curves` 方法**：绘制利润曲线。

## 3. 奖励分布
- **`plot_reward_distribution` 方法**：绘制奖励分布图。

## 4. 训练进度
- **`plot_training_progress` 方法**：绘制训练进度图表。
- **`_plot_moving_averages` 方法**：绘制移动平均线。
- **`_plot_performance_heatmap` 方法**：绘制性能热图。 