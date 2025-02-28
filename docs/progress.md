# 项目进度记录

## 项目结构

### 核心组件

1. **自监督训练器** (`src/self_supervised/trainer/self_supervised_trainer.py`)
   - `SelfSupervisedTrainer`: 主要训练类
   - `MemoryTracker`: 内存使用监控
   - `DataIncrementalLoader`: 增量数据加载
   - `ChartDataset`: 图表数据集管理
   - `EarlyStopping`: 早停机制

2. **通知系统** (`src/notification/discord_notifier.py`)
   - `DiscordNotifier`: Discord 通知实现
   - `DiscordConfig`: Discord 配置管理
   - 支持三种通知类型：信号、监控、上传

3. **工具类** (`src/self_supervised/utils/`)
   - `signal_tracker.py`: 信号跟踪器
   - `reward_mechanism.py`: 奖励机制
   - `progress_notifier.py`: 进度通知

### 配置文件

1. **环境变量** (`.env`)
   - Discord Webhook URLs:
     - `DISCORD_WEBHOOK_MONITOR`
     - `DISCORD_WEBHOOK_SIGNAL`
     - `DISCORD_WEBHOOK_DEBUG`
   - Discord 配置:
     - `DISCORD_BOT_TOKEN`
     - `DISCORD_CLIENT_ID`
     - `DISCORD_BOT_AVATAR`

2. **模型路径**
   - 本地模型路径: `models/llava-sbs/`

## 已完成的功能

1. **自监督训练系统**
   - 实现了完整的训练流程
   - 支持增量数据加载
   - 内存使用监控和优化
   - 学习率调度和早停机制

2. **Discord 通知系统**
   - 支持异步和同步消息发送
   - 多类型通知支持（信号、监控、上传）
   - 自动重试机制
   - 完整的错误处理

3. **数据管理**
   - 增量数据加载
   - 数据完整性检查
   - 内存优化支持

## 使用说明

### 1. Discord 通知配置
```python
# 在 SelfSupervisedTrainer 中已自动配置
trainer = SelfSupervisedTrainer(...)
# Discord 通知器会自动初始化

# 发送通知示例
# 异步方式
await trainer.notifier.send_monitor_message({"status": "training", "progress": "50%"})
# 同步方式
trainer.notifier.send_message_sync("训练开始", webhook_type='monitor')
```

### 2. 模型训练
```python
trainer = SelfSupervisedTrainer(
    model=your_model,
    data_dir="path/to/data",
    save_dir="path/to/save",
    device="cuda"  # 或 "cpu"
)

# 配置学习率调度器
trainer.setup_lr_scheduler(scheduler_type='plateau')

# 开始训练
trainer.train(
    num_epochs=10,
    batch_size=32,
    validate_every=1,
    learning_rate=0.001
)
```

## 注意事项

1. **环境变量配置**
   - 确保所有 Discord Webhook URLs 正确配置
   - 检查模型路径是否正确（`models/llava-sbs/`）

2. **内存管理**
   - 使用 `MemoryTracker` 监控内存使用
   - 启用增量数据加载以优化内存使用

3. **错误处理**
   - 所有关键操作都有错误处理和日志记录
   - Discord 通知失败会自动重试

## 待优化项目

1. **性能优化**
   - [ ] 优化数据加载性能
   - [ ] 改进内存管理策略

2. **功能扩展**
   - [ ] 添加更多训练指标监控
   - [ ] 扩展通知系统功能

## 问题解决方案

1. **Discord 通知失败**
   - 检查环境变量配置
   - 查看日志中的错误信息
   - 使用 `tests/webhook/test_webhook.py` 测试

2. **内存使用过高**
   - 启用增量数据加载
   - 调整批次大小
   - 检查内存跟踪器日志

## 遇到的错误

- 错误1：描述
- 错误2：描述

## 错误修复方案

- 错误1的修复方案
- 错误2的修复方案

## 实现需求和逻辑

- 每个部分的实现需求和逻辑描述

## 可能出现的问题及解决方案

- 问题1：描述及解决方案
- 问题2：描述及解决方案

class NQRewardMechanism:
    def __init__(self):
        self.point_value = 20  # NQ每点20美元
        self.config = {
            'min_reward': -1000,  # 最大单次损失限制
            'max_reward': 2000,   # 最大单次收益限制
            'time_decay_factor': 20,  # 时间衰减因子
            'volatility_weight': 0.2  # 波动率权重
        }

    def calculate_trade_reward(self, signal_info: Dict) -> float:
        # 基础点数收益
        price_diff = signal_info['exit_price'] - signal_info['entry_price']
        if signal_info['position_type'] == 'short':
            price_diff = -price_diff
            
        # 基础美元收益
        base_reward = price_diff * self.point_value
        
        # 风险调整
        stop_loss = abs(signal_info['stop_loss'] - signal_info['entry_price'])
        risk_reward_ratio = abs(price_diff) / stop_loss if stop_loss > 0 else 1.0
        
        # 计算最终奖励
        final_reward = base_reward * self._calculate_multipliers(signal_info, risk_reward_ratio)
        
        return np.clip(final_reward, self.config['min_reward'], self.config['max_reward'])

    def _calculate_multipliers(self, signal_info: Dict, risk_reward_ratio: float) -> float:
        # 时间衰减
        duration = signal_info.get('duration_minutes', 0)
        time_multiplier = np.exp(-duration / self.config['time_decay_factor'])
        
        # 波动率调整
        volatility = signal_info.get('market_volatility', 1.0)
        vol_multiplier = 1.0 + (volatility - 1.0) * self.config['volatility_weight']
        
        # 风险收益比调整
        rr_multiplier = min(risk_reward_ratio / 2.0, 2.0)
        
        return time_multiplier * vol_multiplier * rr_multiplier

training_config = {
    # 显存优化
    'gpu_settings': {
        'batch_size': 48,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'max_seq_length': 512,
        'memory_efficient_attention': True
    },
    
    # 基准测试配置（2024最后一周）
    'baseline_validation': {
        'period': '2024-12-23/2024-12-31',
        'metrics': ['accuracy', 'profit_loss', 'max_drawdown', 'sharpe_ratio']
    },
    
    # 初始训练配置（2008-2010）
    'initial_training': {
        'years': [2008, 2009, 2010],
        'epochs_per_year': 5,
        'learning_rates': {
            2008: 1e-4,
            2009: 5e-5,
            2010: 2e-5
        }
    },
    
    # 验证和检查点
    'validation': {
        'frequency': 100,  # 批次
        'metrics_threshold': {
            'min_profit': 1000,  # 美元
            'max_drawdown': -2000,  # 美元
            'min_accuracy': 0.55,
            'min_sharpe': 1.2
        }
    }
} 