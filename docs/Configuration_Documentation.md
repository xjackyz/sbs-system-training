# 配置文档

## 配置文件结构
项目的配置文件主要包括 `training_config.py` 和 `.env` 文件。

### 1. training_config.py
该文件包含训练过程中的各种配置参数。

#### 主要配置项
- **gpu_settings**: GPU相关设置
  - `batch_size`: 批处理大小。
  - `gradient_accumulation_steps`: 梯度累积步数。
  - `mixed_precision`: 使用的混合精度类型。
  - `max_seq_length`: 最大序列长度。
  - `memory_efficient_attention`: 是否启用内存高效注意力。
  - `gradient_checkpointing`: 是否启用梯度检查点。

- **baseline_validation**: 基准验证配置
  - `period`: 验证的时间范围。
  - `metrics`: 需要监控的指标。

- **initial_training**: 初始训练配置
  - `years`: 训练年份。
  - `epochs_per_year`: 每年的训练轮数。
  - `learning_rates`: 每年的学习率。

- **validation**: 验证和检查点配置
  - `frequency`: 验证频率。
  - `metrics_threshold`: 各种指标的阈值。

### 2. .env 文件
该文件用于存储环境变量和敏感信息。

#### 主要配置项
- **系统配置**
  - `ENVIRONMENT`: 环境类型（development/production）。
  - `DEBUG`: 是否启用调试模式。
  - `LOG_LEVEL`: 日志级别。

- **设备配置**
  - `USE_GPU`: 是否使用GPU。
  - `NUM_WORKERS`: 数据加载的工作线程数。

- **模型配置**
  - `MODEL_PATH`: 模型文件路径。
  - `BATCH_SIZE`: 批处理大小。

- **网络配置**
  - `MIRROR_URL`: 镜像URL。
  - `VERIFY_SSL`: 是否验证SSL。

## 注意事项
- 确保在使用前正确配置所有参数。
- 不要将敏感信息（如API密钥）硬编码在代码中，使用环境变量进行管理。 