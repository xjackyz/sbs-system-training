# SBS训练系统优化

## 优化概述

为了提高SBS系统的可维护性、可扩展性和易用性，我们对训练系统进行了全面优化。主要优化包括：

1. **统一训练器实现**：创建了`SBSTrainer`类，支持多种训练模式，减少代码重复。
2. **统一配置管理**：使用`ConfigManager`类，实现了配置的层次化访问和环境变量覆盖。
3. **统一训练入口**：使用`sbs_train.py`作为统一入口，支持多种训练模式。
4. **标注服务优化**：将JS版本的Label Studio服务替换为Python版本，保持语言一致性。
5. **训练脚本整合**：合并重复的训练脚本，减少维护负担。

## 系统组件关系

整个训练系统形成一个完整的循环，支持持续改进的交易策略：

```
数据收集 -> 数据处理 -> 模型训练 -> 模型评估 -> 交易预测 -> 交易执行 -> 结果收集 -> 回到训练
```

1. **数据收集**：收集K线图和市场数据
2. **数据处理**：处理和准备训练数据
3. **模型训练**：使用四种训练模式之一训练模型
   - 标准训练
   - 自监督训练
   - 强化学习训练
   - 主动学习训练
4. **模型评估**：评估模型性能和预测质量
5. **交易预测**：使用模型预测SBS序列和交易信号
6. **交易执行**：执行交易操作
7. **结果收集**：收集交易结果和奖励信息
8. **反馈到训练**：使用交易结果改进下一轮训练

## 训练模式说明

SBS系统支持四种训练模式，每种模式适用于不同的场景：

### 1. 标准训练 (Standard Training)

适用于有大量标注数据的情况，使用常规监督学习。

```
python scripts/sbs_train.py --mode standard --config config/training_config.yaml
```

### 2. 自监督训练 (Self-supervised Training)

适用于标注数据有限的情况，利用未标注数据进行预训练。

```
python scripts/sbs_train.py --mode self_supervised --config config/training_config.yaml
```

### 3. 强化学习训练 (Reinforcement Learning Training)

适用于需要优化实际交易策略的情况，使用奖励信号指导学习。

```
python scripts/sbs_train.py --mode reinforcement --config config/training_config.yaml
```

### 4. 主动学习训练 (Active Learning Training)

适用于人工标注资源有限的情况，选择最有价值的样本进行标注。

```
python scripts/sbs_train.py --mode active --config config/training_config.yaml
```

## 训练流程

1. 使用`ConfigManager`加载配置文件
2. 根据训练模式初始化`SBSTrainer`类
3. 准备训练数据和验证数据
4. 执行训练循环
5. 定期评估模型性能
6. 保存模型检查点
7. 记录训练指标和日志

## 奖励计算和交易跟踪

系统使用`SBSRewardCalculator`计算模型预测的奖励值，同时使用`TradeResultTracker`跟踪和评估交易结果。这两个组件紧密协作，确保模型能够从实际交易结果中学习和改进。

## 删除的冗余脚本

以下脚本已经合并到统一的训练入口，不再单独使用：

- `train.py`：旧版训练脚本
- `train_self_supervised.py`：自监督训练脚本
- `train_rl.py`：强化学习训练脚本
- `train_with_reward.py`：奖励训练脚本

## 下一步改进计划

1. 实现数据加载器模块，支持不同训练模式
2. 完善评估系统，增加更多指标
3. 优化奖励计算机制
4. 完善模型推理部分
5. 增加训练可视化功能 