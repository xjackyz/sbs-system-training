# SBS Trading Analysis System - 自监督学习模块

本模块实现了基于LLaVA模型的自监督学习功能，用于从交易图表中识别SBS（买卖序列）信号，并通过自监督学习方式不断优化模型性能。

## 功能特点

- 从CSV数据生成交易图表并进行处理
- 使用微调后的LLaVA模型识别SBS信号
- 追踪交易信号并计算收益/亏损
- 基于交易结果实现自监督学习
- 保存完整SBS序列的图表用于后续微调
- 提供A100 GPU处理时间估算

## 目录结构

```
src/self_supervised/
├── config/                     # 配置文件
│   └── default_config.json     # 默认配置
├── data/                       # 数据处理模块
│   └── csv_to_chart_processor.py  # CSV数据转图表处理器
├── model/                      # 模型相关
│   └── llava_model_wrapper.py  # LLaVA模型封装
├── signals/                    # 信号处理
│   └── signal_tracker.py       # 信号追踪器
├── trainer/                    # 训练相关
│   └── self_supervised_manager.py  # 自监督学习管理器
├── run_self_supervised_learning.py  # 主运行脚本
└── README.md                   # 本文档
```

## 快速开始

1. 准备数据：确保您的CSV数据符合要求，包含必要的OHLCV数据列

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行自监督学习：
```bash
python src/self_supervised/run_self_supervised_learning.py --csv_path /path/to/your/data.csv --model_path /path/to/llava/model
```

4. 仅估算计算时间：
```bash
python src/self_supervised/run_self_supervised_learning.py --csv_path /path/to/your/data.csv --estimate_only
```

## 参数配置

可以通过命令行参数或配置文件自定义运行参数：

### 命令行参数

- `--csv_path`: CSV数据文件路径
- `--model_path`: LLaVA模型路径
- `--output_dir`: 图表输出目录
- `--window_size`: 图表窗口大小（K线数量）
- `--stride`: 滑动窗口步长
- `--batch_size`: 批处理大小
- `--save_ratio`: 保存SBS序列图表的比例
- `--device`: 使用的设备，如 cuda:0, cpu
- `--estimate_only`: 仅估算处理时间，不实际运行
- `--config`: 配置文件路径

### 配置文件

您可以通过JSON配置文件设置所有参数，示例见 `config/default_config.json`：

```json
{
  "csv_path": "NQ_full_1min_continuous/NQ_full_1min_continuous.csv",
  "model_path": "models/llava-sbs/",
  "output_dir": "data/charts",
  "sbs_dir": "data/sbs_sequences",
  "signals_dir": "data/signals",
  "checkpoint_dir": "models/checkpoints",
  "device": "cuda",
  "batch_size": 4,

  "chart": {
    "window_size": 100,
    "stride": 1,
    "figsize": [10, 6],
    "dpi": 100,
    "save_ratio": 0.01,
    "format": "png"
  },

  "signal_tracker": {
    "profit_target": 0.01,
    "stop_loss": 0.005,
    "max_holding_periods": 20,
    "reward_profit_factor": 2.0,
    "penalty_loss_factor": 1.0
  },

  "training": {
    "epochs": 3,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "evaluation_steps": 200,
    "save_steps": 500,
    "max_samples": 10000
  },

  "a100_estimation": {
    "time_per_chart": 0.05,
    "memory_utilization": 0.8
  }
}
```

## 处理流程

1. **数据加载**：从CSV加载历史K线数据
2. **图表生成**：根据窗口大小和步长生成交易图表
3. **信号识别**：使用LLaVA模型检测SBS序列
4. **信号追踪**：追踪交易信号并计算收益/亏损
5. **奖励计算**：根据交易结果计算奖励/惩罚
6. **样本筛选**：选择有效样本用于模型训练
7. **模型训练**：基于筛选后的样本进行微调
8. **指标评估**：计算模型性能指标

## A100 GPU 时间估算

默认配置下，每张图表的处理时间约为0.05秒。系统会根据您的数据大小和配置自动计算总预计时间。

您可以通过 `--estimate_only` 参数或调整 `a100_estimation` 配置来优化估算。

## 输出文件

- 生成的图表保存在 `output_dir` 目录
- 识别到的SBS序列保存在 `sbs_dir` 目录
- 信号追踪信息保存在 `signals_dir` 目录
- 模型检查点保存在 `checkpoint_dir` 目录
- 运行日志保存在 `logs/self_supervised` 目录

## 注意事项

- 确保您的GPU有足够内存运行LLaVA模型
- 根据您的数据量调整批处理大小
- 对于大型数据集，建议增加步长以减少总图表数量
- 对于生产环境，可能需要增加窗口大小以获取更多上下文信息

## 自定义扩展

您可以通过以下方式扩展功能：

1. 修改 `csv_to_chart_processor.py` 以自定义图表生成
2. 调整 `signal_tracker.py` 中的信号逻辑和奖励机制
3. 更新 `llava_model_wrapper.py` 中的系统提示以改变模型解释方式
4. 在 `self_supervised_manager.py` 添加新的训练策略 