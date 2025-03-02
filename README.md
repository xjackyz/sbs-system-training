# SBS交易系统 (Sequential Breakthrough System)

<div align="center">
  <img src="docs/images/sbs_logo.png" alt="SBS Logo" width="200"/> <!-- 示意图，需要创建 -->
  
  <p>
    <strong>基于视觉的智能交易分析系统</strong><br>
    识别市场结构突破 · 预测交易信号 · 优化交易决策
  </p>
</div>

## 📑 目录

- [系统概述](#-系统概述)
- [核心特点](#-核心特点)
- [安装指南](#-安装指南)
- [快速开始](#-快速开始)
- [SBS交易序列详解](#-sbs交易序列详解)
- [系统架构](#-系统架构)
- [模块详解](#-模块详解)
- [使用示例](#-使用示例)
- [配置指南](#-配置指南)
- [常见问题](#-常见问题)
- [开发指南](#-开发指南)
- [许可协议](#-许可协议)

## 🔍 系统概述

SBS（Sequential Breakthrough System）是一个基于视觉的智能交易分析系统，核心使用经过专门微调的LLaVA（Large Language and Vision Assistant）多模态模型。系统能够"看懂"交易图表，直接识别市场结构突破点和交易形态，模拟专业交易员的视觉分析能力。

SBS系统只依赖原始K线数据和两条简单移动平均线（SMA20和SMA200）作为辅助指标，摒弃了复杂的技术指标组合，专注于价格行为和市场结构分析。

![SBS系统概览](docs/images/system_overview.png) <!-- 示意图，需要创建 -->

## ✨ 核心特点

- **视觉理解优先**：直接从K线图表中识别交易模式，不依赖复杂的数值指标
- **结构化分析**：识别市场结构突破和关键点位序列（12345序列）
- **简约指标使用**：仅使用SMA20和SMA200作为辅助判断趋势方向
- **多模态学习**：结合视觉和语言能力解读市场
- **自适应策略**：通过主动学习持续优化交易策略
- **多样形态识别**：能识别双顶双底、流动性获取(Liquidate)和SCE(单蜡烛入场)等关键形态
- **全面可视化**：丰富的图表分析和交易信号标注

## 📦 安装指南

### 系统要求

- Python 3.8+
- CUDA 11.7+ (推荐GPU训练)
- 至少16GB RAM (推荐32GB+)
- 至少8GB GPU显存 (推荐16GB+用于模型训练)

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/yourusername/sbs_system.git
cd sbs_system
```

2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 下载预训练模型

```bash
python scripts/download_pretrained_models.py
```

## 🚀 快速开始

### 准备训练数据

```bash
# 处理图表图像并准备训练数据
python scripts/prepare_image_data.py --src_dir data/raw --output_dir data/processed
```

### 模型微调

```bash
# 对LLaVA模型进行微调
python scripts/train_llava_sbs.py --config config/training_config.yaml
```

### 主动学习优化

```bash
# 选择最具信息价值的样本进行标注
python scripts/active_learning.py --model_path models/llava-sbs/final --num_samples 10
```

### 预测交易信号

```bash
# 分析交易图表
python scripts/predict_with_llava.py --image path/to/chart.png --prompt "图片中是否显示了市场结构的突破？"
```

## 📈 SBS交易序列详解

SBS交易系统基于"12345"序列模式进行交易，这是一种市场结构分析方法，包含5个关键点位：

![SBS交易序列示意图](docs/images/sbs_sequence.png) <!-- 示意图，需要创建 -->

### 5个关键点位

1. **点1**：突破后的第一次回调形成的点，即突破后第一个明显的回调高点或低点
2. **点2**：由点1创造出的极值点（上升趋势时的最高高点或下降趋势时的最低低点），作为主要盈利目标
3. **点3**：在点1附近获取流动性的点，即价格回调突破点1所创造的高/低点
4. **点4**：确认点，通常与点3形成双底/双顶或通过流动性获取确认，此时价格显示出结构性的反转迹象
5. **点5**：趋势继续，价格回到点2的位置

### 入场确认信号

系统识别四种主要入场确认信号：

- **结构突破**：价格突破前一个高点/低点，由实体K线确认
- **双顶/双底**：在点3和点4之间形成的反转形态，需要顶点或底点接近并有明显回调
- **流动性获取(Liquidate)**：价格短暂突破关键位置后没有发生结构反转，而是迅速反转回原有趋势
- **SCE(Single Candle Entry)**：一根蜡烛收盘后高点低点均高于和低于前一根不同颜色的蜡烛，后续第一根蜡烛为同样颜色，通常发生在点4

### 交易执行

- **入场点**：点4确认后，符合入场条件时（通常是点3的回调被突破后）
- **止损位**：点4下方(多单)/上方(空单)，防止因假突破造成较大亏损
- **止盈位**：主要目标为点2位置，或点2到点3之间的61.8%斐波那契回撤位作为第一止盈位

### 趋势确认

SMA20和SMA200均线可以帮助确认市场趋势：
- 价格高于SMA20和SMA200：上升趋势，适合做多
- 价格低于SMA20和SMA200：下降趋势，适合做空

## 🏗️ 系统架构

SBS系统由多个协同工作的模块组成，下图展示了各模块之间的关系：

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  LLaVA-SBS模型  │◄───┤   图表处理器    │◄───┤   图表数据输入  │
└────────┬────────┘    └────────┬────────┘    └─────────────────┘
         │                      │                       ▲
         ▼                      ▼                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  SBSTrainer     │◄───┤  SBSDataset     │◄───┤   交易图表源    │
└────────┬────────┘    └────────┬────────┘    └─────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ SBSActiveLearner◄───┤    结果分析器    │    │  交易信号生成器 │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┴──────────────────────┘
```

## 📚 模块详解

### 1. 核心模型模块

#### 1.1 LLaVA-SBS模型

**功能**：多模态视觉语言模型，负责K线图分析和交易形态识别。

**主要组件**：
- 视觉编码器(ViT-L-14-336px)：处理图表视觉特征
- 语言模型：基于视觉特征生成交易分析
- 多模态连接层：连接视觉和语言理解能力

**核心能力**：
- 识别市场结构突破
- 检测双顶/双底形态
- 分析流动性获取点
- 识别SCE入场信号
- 标记SBS序列关键点位

#### 1.2 SBSDataset

**功能**：处理交易图表图像和相应的提示文本

**主要特点**：
- 加载和预处理交易图表图像
- 匹配图像与对应的提示文本
- 支持图像尺寸调整和标准化
- 提供批量数据加载功能

#### 1.3 SBSActiveLearningDataset

**功能**：支持主动学习的数据集，扩展基本数据集以支持不确定性采样和主动学习策略

**核心功能**：
- 更新样本权重用于主动学习采样
- 标记已标注样本
- 获取未标记样本索引
- 生成样本元数据信息

### 2. 训练与优化模块

#### 2.1 SBSTrainer

**功能**：定制的训练器，继承自Huggingface的Trainer，添加特定于SBS任务的训练功能

**核心功能**：
- 计算训练损失
- 评估模型性能
- 保存模型和训练配置
- 提供训练过程监控

#### 2.2 SBSActiveLearner

**功能**：主动学习器，用于实施主动学习策略，选择最具信息价值的样本进行标注

**核心功能**：
- 选择样本进行标注
- 计算样本不确定性
- 不同选择策略支持(uncertainty/random)
- 更新模型权重

### 3. 使用与预测模块

#### 3.1 预测工具

**功能**：使用微调后的LLaVA-SBS模型分析交易图表

**核心功能**：
- 加载微调后的模型
- 处理输入交易图表
- 生成分析结果和交易建议
- 支持多种分析提示模板

## 🌟 使用示例

### 微调LLaVA-SBS模型

```python
# 加载配置
with open('config/training_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 设置随机种子
seed = config.get('seed', 42)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 准备模型
processor, model = prepare_llava_model(config)

# 准备数据集
train_dataset, val_dataset = prepare_datasets(config, processor)

# 开始训练
training_args = TrainingArguments(
    output_dir=config['paths']['model_dir'] + '/checkpoints',
    per_device_train_batch_size=config['training']['batch_size'],
    learning_rate=float(config['training']['learning_rate']),
    num_train_epochs=config['training']['epochs'],
    fp16=True
)

# 创建训练器
trainer = SBSTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer
)

# 执行训练
trainer.train()

# 保存模型
final_model_path = config['paths']['model_dir'] + '/final'
model.save_pretrained(final_model_path)
processor.save_pretrained(final_model_path)
```

### 主动学习样本选择

```python
# 加载模型和处理器
processor, model = load_model_and_processor(model_path, device='cuda')

# 准备数据集
dataset = SBSActiveLearningDataset(
    data_dir='data/active_learning',
    processor=processor,
    image_size=(336, 336)
)

# 创建主动学习器
active_learner = SBSActiveLearner(
    model=model,
    dataset=dataset,
    processor=processor,
    device='cuda',
    batch_size=4
)

# 选择样本
selected_indices = active_learner.select_samples(
    num_samples=10,
    strategy='uncertainty'  # 'uncertainty' 或 'random'
)

# 标记选中的样本
dataset.mark_as_labeled(selected_indices)

# 获取样本元数据
for idx in selected_indices:
    metadata = dataset.get_sample_metadata(idx)
    print(f"样本 {idx}: {metadata['image_path']}")
```

### 预测与交易图表分析

```python
# 加载模型和处理器
processor, model = load_model_and_processor(model_path, device='cuda')

# 处理图像
image = process_image(image_path, processor, image_size=(336, 336))

# 设置提示词
prompt = "图片中是否显示了市场结构的突破？前一个高点或低点是否被实体K线突破？请指出具体位置。"

# 分析图表
result = analyze_chart(image, processor, model, prompt)

# 输出结果
print("分析结果:")
print(result)
```

## ⚙️ 配置指南

SBS系统使用YAML格式的配置文件，所有配置文件位于`config/`目录。

### 主要配置文件

- **训练配置**: `config/training_config.yaml`
- **主动学习配置**: `config/active_learning_config.yaml`
- **预测配置**: `config/predict_config.yaml`

### 训练配置示例

```yaml
# 基本配置
environment: development
seed: 42

# 设备配置
device:
  use_gpu: true
  num_workers: 4
  accelerator: "auto"
  precision: "16-mixed"

# 模型配置
model:
  path: "liuhaotian/llava-v1.5-7b"
  type: "llava"
  image_size: [336, 336]
  max_length: 512

# 训练配置
training:
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  epochs: 3
  
  # 学习率调度器
  scheduler:
    type: "cosine"
    warmup_steps: 100
  
  # LoRA配置
  lora:
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "v_proj"]

# 路径配置
paths:
  data_dir: "data"
  model_dir: "models/llava-sbs"
  log_dir: "logs"
```

### 主动学习配置示例

```yaml
# 主动学习配置
active_learning:
  batch_size: 4
  num_samples: 10
  strategy: "uncertainty"
  uncertainty_threshold: 0.7
  diversity_weight: 0.5
```

## ❓ 常见问题

### 训练相关

**Q: LLaVA-SBS模型训练需要什么样的计算资源？**

A: 微调至少需要一张16GB显存的GPU，理想情况下建议使用24GB以上显存的GPU。可以通过开启量化(4bit或8bit)和梯度累积来减少显存需求。

**Q: 如何提高模型识别准确率？**

A: 增加高质量的标注数据，通过主动学习选择最具信息价值的样本进行标注，细化形态定义，确保训练数据覆盖各种市场条件。

**Q: 训练数据如何准备？**

A: 交易图表应按336x336像素大小标准化，并为每张图表准备对应的提示文本，描述图表中的市场结构和模式。使用`prepare_image_data.py`脚本可以自动化这一过程。

### 预测相关

**Q: 系统支持哪些市场的分析？**

A: 系统适用于任何提供K线图表的市场，包括股票、外汇和加密货币市场，只要图表中包含K线和SMA20/SMA200均线即可。

**Q: 如何提高入场信号的可靠性？**

A: 结合多个入场确认信号，尤其在SMA20和SMA200的相对位置支持交易方向时。例如，在上升趋势中(价格在均线上方)寻找多头信号，下降趋势中(价格在均线下方)寻找空头信号。

**Q: 如何解释模型的预测结果？**

A: 模型输出包含对交易图表的自然语言分析，指出是否存在市场结构突破、双顶/双底形态、流动性获取点，以及SBS序列中的各个关键点位。根据这些分析来确定入场点、止损位和止盈位。

### 系统相关

**Q: 如何为不同市场微调模型？**

A: 针对特定市场收集标注图表，运行增量微调训练。例如，如果主要交易加密货币，可以准备更多加密货币市场的图表；如果交易外汇，则准备更多外汇市场的图表。

**Q: 系统是否适用于不同的时间周期？**

A: 是的，SBS系统适用于各种时间周期的图表，从分钟级到日线级别。建议为特定时间周期微调模型以获得最佳结果。

**Q: 如何评估模型性能？**

A: 可以使用以下指标：
- 序列识别准确率：模型正确识别SBS序列点位的准确率
- 入场信号精度：模型正确识别入场时机的准确度
- 回测表现：基于模型建议的交易胜率和盈亏比

## 👨‍💻 开发指南

### 代码风格

项目遵循以下代码规范：

- 遵循PEP 8规范
- 使用4空格缩进
- 最大行长度限制为120字符
- 类名使用CamelCase
- 函数和变量名使用snake_case
- 常量使用大写字母
- 文档字符串使用中文
- 注释应清晰解释复杂逻辑

### 导入规范

- 标准库导入在最前
- 第三方库导入其次
- 本地模块导入最后
- 每组导入之间空一行

### 目录结构

```
sbs_system/
├── app/                # 应用程序主要代码
├── config/            # 配置文件
├── data/              # 数据文件
│   ├── raw/           # 原始图表
│   ├── processed/     # 处理后的训练数据
│   └── active_learning/ # 主动学习数据
├── docs/              # 文档
├── logs/              # 日志文件
├── models/            # 模型相关代码
├── prompts/           # 提示词模板
├── scripts/           # 工具脚本
├── src/               # 源代码
│   ├── data_utils/    # 数据处理工具
│   ├── training/      # 训练相关代码
│   └── utils/         # 工具类
├── tests/             # 测试代码
└── requirements.txt   # 依赖项
```

### 版本控制

- 提交信息使用中文
- 提交信息需要清晰描述改动
- 每个提交专注于单一功能或修复
- 分支管理:
  - main: 主分支，保持稳定
  - develop: 开发分支
  - feature/*: 功能分支
  - bugfix/*: 修复分支

## 📄 许可协议

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">
  <p>
    SBS交易系统开发团队 © 2023<br>
    <a href="https://github.com/yourusername/sbs_system">GitHub仓库</a> · 
    <a href="https://docs.sbs-system.com">文档网站</a> · 
    <a href="https://discord.gg/sbs-system">Discord社区</a>
  </p>
</div>