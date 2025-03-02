# 交易结果跟踪器（TradeResultTracker）示例脚本

本目录包含了演示 `TradeResultTracker` 类各项功能的示例脚本。这些脚本展示了如何使用交易结果跟踪器进行交易数据分析、内存管理、可视化和参数优化。

## 示例脚本列表

### 1. 内存管理和分批处理示例 (memory_management_example.py)

演示如何使用 `TradeResultTracker` 的内存管理和分批处理功能处理大量交易数据。

**功能特点：**
- 生成大量模拟交易数据
- 演示内存自动清理和归档功能
- 展示分批处理大数据集的高效方法
- 对比批处理与非批处理的性能差异

**使用方法：**
```bash
# 基本使用
python memory_management_example.py

# 自定义参数
python memory_management_example.py --num_trades 20000 --batch_size 1000 --save_data
```

**参数说明：**
- `--num_trades`: 生成的交易数量，默认为10000
- `--batch_size`: 处理批次大小，默认为500
- `--save_data`: 保存生成的交易数据，默认不保存

### 2. 交易可视化示例 (visualization_example.py)

演示 `TradeResultTracker` 的各种可视化功能，用于分析交易结果和探索效果。

**功能特点：**
- 生成具有随机但有规律性的交易数据
- 展示交易业绩、权益曲线等基础可视化
- 生成SBS模式分析和确认信号效果分析
- 创建自定义分析图表
- 生成完整的HTML交易报告

**使用方法：**
```bash
# 基本使用
python visualization_example.py

# 自定义参数
python visualization_example.py --num_trades 500
```

**参数说明：**
- `--num_trades`: 生成的交易数量，默认为200

### 3. Optuna参数优化示例 (optuna_optimization_example.py)

演示如何使用 `TradeResultTracker` 的优化功能寻找最佳的探索策略参数。

**功能特点：**
- 生成三种不同市场环境的交易数据
- 使用Optuna进行探索参数的超参数优化
- 可视化优化过程和结果
- 生成详细的优化报告
- 使用最佳参数配置生成交易业绩报告

**使用方法：**
```bash
# 基本使用
python optuna_optimization_example.py

# 自定义参数
python optuna_optimization_example.py --num_trades 1000 --num_trials 100 --random_seed 123
```

**参数说明：**
- `--num_trades`: 生成的交易数量，默认为500
- `--num_trials`: Optuna优化的试验次数，默认为50
- `--random_seed`: 随机数种子，默认为42
- `--load_data`: 从文件加载现有交易数据，默认为False
- `--output_dir`: 输出目录，默认为data/optimization

## 依赖库

运行这些示例脚本需要安装以下Python库：

```bash
# 基础依赖
pip install numpy pandas matplotlib seaborn tqdm

# 内存和性能监控
pip install psutil

# 优化依赖
pip install optuna
```

## 目录结构建议

为了使示例脚本正常运行，建议创建以下目录结构：

```
data/
  ├── memory_test/     # 内存管理示例输出目录
  ├── visualization/   # 可视化示例输出目录
  └── optimization/    # 参数优化示例输出目录
```

可以使用以下命令创建：

```bash
mkdir -p data/memory_test data/visualization data/optimization
```

## 注意事项

1. 这些示例脚本生成的是模拟交易数据，仅用于演示功能
2. 实际使用时，您应该替换为自己的真实交易数据
3. 优化和可视化过程可能需要较长时间，特别是数据量大时
4. 系统内存有限时，请适当调整 `num_trades` 和 `batch_size` 参数 