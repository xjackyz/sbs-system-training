# SBS训练系统使用指南

本文档提供了SBS（Sequential Breakthrough System）训练系统的详细使用说明，包括不同训练模式的配置和使用方法。

## 目录

- [系统概述](#系统概述)
- [安装和准备](#安装和准备)
- [训练模式](#训练模式)
  - [标准训练模式](#标准训练模式)
  - [自监督训练模式](#自监督训练模式)
  - [强化学习模式](#强化学习模式)
  - [主动学习模式](#主动学习模式)
- [配置文件说明](#配置文件说明)
- [命令行参数](#命令行参数)
- [示例用法](#示例用法)
- [结果分析](#结果分析)
- [常见问题](#常见问题)

## 系统概述

SBS训练系统是一个统一的训练框架，支持多种训练模式，用于训练交易序列预测模型。系统特点包括：

- 统一的配置管理
- 多种训练模式支持
- 完整的训练、评估和可视化功能
- 灵活的数据加载和处理机制
- 详细的日志记录和结果追踪

## 安装和准备

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA（推荐用于GPU训练）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

根据不同的训练模式，准备相应的数据集：

- 标准训练：准备已标注的训练和验证数据
- 自监督训练：准备已标注和未标注的数据
- 强化学习：准备市场数据
- 主动学习：准备少量标注数据和大量未标注数据

## 训练模式

### 标准训练模式

标准训练模式使用有监督学习方法训练模型。

**运行命令**：

```bash
python src/sbs_train.py --config config/training/standard_training.yaml --mode standard
```

### 自监督训练模式

自监督训练模式同时使用标记数据和未标记数据，利用自监督学习方法提高模型泛化能力。

**运行命令**：

```bash
python src/sbs_train.py --config config/training/self_supervised_training.yaml --mode self_supervised
```

### 强化学习模式

强化学习模式使用奖励机制训练模型，通过交易结果反馈优化模型性能。

**运行命令**：

```bash
python src/sbs_train.py --config config/training/reinforcement_learning.yaml --mode rl
```

### 主动学习模式

主动学习模式通过不确定性估计选择最有价值的样本进行标注，减少人工标注成本。

**运行命令**：

```bash
python src/sbs_train.py --config config/training/active_learning.yaml --mode active_learning
```

## 配置文件说明

配置文件使用YAML格式，包含以下主要部分：

1. **基本配置**：名称、描述和版本
2. **数据配置**：数据路径和格式
3. **模型配置**：模型结构和参数
4. **优化器配置**：优化器类型和参数
5. **训练参数**：批量大小、轮数等
6. **设备配置**：训练设备选择
7. **日志和检查点**：结果保存路径
8. **评估配置**：评估指标和可视化选项

详细配置请参考 `config/training/` 目录下的示例配置文件。

## 命令行参数

以下是主要的命令行参数：

- `--config`：配置文件路径（必需）
- `--mode`：训练模式，可选 standard, self_supervised, rl, active_learning
- `--output_dir`：输出目录
- `--resume`：从检查点恢复训练
- `--seed`：随机种子
- `--data_path`：数据路径
- `--labeled_path`：已标记数据路径
- `--unlabeled_path`：未标记数据路径
- `--val_path`：验证数据路径
- `--batch_size`：批处理大小
- `--epochs`：训练轮数
- `--learning_rate`：学习率
- `--device`：训练设备
- `--log_level`：日志级别
- `--log_dir`：日志目录

## 示例用法

### 使用自定义数据路径运行标准训练

```bash
python src/sbs_train.py --config config/training/standard_training.yaml --mode standard --data_path /path/to/your/data --val_path /path/to/val/data
```

### 调整批量大小和学习率

```bash
python src/sbs_train.py --config config/training/reinforcement_learning.yaml --mode rl --batch_size 64 --learning_rate 0.0002
```

### 从检查点恢复训练

```bash
python src/sbs_train.py --config config/training/self_supervised_training.yaml --mode self_supervised --resume checkpoints/self_supervised/best_model.pth
```

## 结果分析

训练完成后，结果将保存在指定的输出目录中，包括：

- 模型检查点
- 训练日志
- 评估结果
- 可视化图表
- 交易历史（强化学习模式）

## 常见问题

### 内存不足

如果遇到内存不足问题，尝试减小批量大小或使用梯度累积。

### GPU显存不足

对于GPU显存不足问题，可以:
- 减小批量大小
- 使用混合精度训练
- 使用更小的模型

### 训练不稳定

如果训练过程不稳定，尝试:
- 减小学习率
- 增加梯度裁剪
- 检查数据预处理
- 调整模型参数 