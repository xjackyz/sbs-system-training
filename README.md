# SBS System (Step-by-Step System)

## 项目概述
SBS System是一个基于LLaVA的自适应学习系统，用于图像分析和市场预测。系统采用分布式训练架构，支持自动标注和主动学习。

## 特性
- 🚀 基于LLaVA的多模态分析
- 📈 自适应学习和在线更新
- 🌐 分布式训练支持
- 🎯 主动学习采样
- 📊 自动标注系统
- 🔄 实时性能监控
- 🛡️ 多级缓存机制

## 安装

### 环境要求
- Python 3.9+
- CUDA 11.7+
- Redis 6.0+
- PostgreSQL 13+

### 快速开始
```bash
# 克隆仓库
git clone git@github.com:xjackyz/sbs-system-training.git
cd sbs-system-training

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.template .env
# 编辑 .env 文件设置必要的环境变量

# 运行训练
python scripts/train.py --config config/training_config.yaml
```

## 系统架构

### 核心组件

#### 1. 自适应训练器 (AdaptiveTrainer)
自适应训练模块负责动态调整训练参数和模型架构。

主要功能：
- 动态学习率调整
- GPU内存自适应的批次大小调整
- 在线模型更新
- 性能监控和错误处理
- 训练状态管理
- 指标分析和可视化

#### 2. 分布式训练 (Ray Trainer)
基于Ray框架的分布式训练实现。

主要功能：
- 多GPU训练支持
- 数据并行处理
- 动态资源分配
- 容错和恢复机制
- 训练进度同步
- 结果合并和验证

#### 3. 缓存管理 (Redis Manager)
多级缓存系统实现。

主要功能：
- L1/L2缓存管理
- 数据预加载
- 内存优化
- 性能监控
- 错误处理
- 统计分析

#### 4. 标注系统 (Label Studio Integration)
Label Studio集成模块。

主要功能：
- 项目管理
- 数据导入导出
- 标注质量控制
- 用户管理
- 自动化工作流
- 性能监控

### 工具脚本

#### 1. 训练脚本
```bash
python scripts/train.py --config config/training_config.yaml
```

#### 2. 标注工具
```bash
python scripts/label_sbs_sequences.py --data path/to/data.csv
```

#### 3. 渲染工具
```bash
python scripts/render_klines.py --input data.csv --output-dir output/
```

#### 4. 可视化工具
```bash
python scripts/visualize_training.py --stats-dir logs/rewards/
```

## 配置说明

### 训练配置
配置文件位于 `config/training_config.yaml`，包含：
- 基础训练参数
- 模型架构设置
- 优化器配置
- 数据处理参数
- 分布式训练设置
- 监控和日志配置

### 环境变量
环境变量模板位于 `.env.template`，包含：
- API密钥
- 数据库连接
- 缓存设置
- 安全配置
- 监控参数

## 开发规范

### 代码风格
- 遵循PEP 8规范
- 使用类型注解
- 编写完整的文档字符串
- 保持代码简洁清晰

### 测试规范
- 单元测试覆盖
- 集成测试
- 性能测试
- 代码质量检查

### Git工作流
- 主分支：main
- 开发分支：develop
- 功能分支：feature/*
- 修复分支：bugfix/*

## 监控和日志

### 性能监控
- Prometheus指标收集
- Grafana仪表板
- 资源使用监控
- 训练进度跟踪

### 日志系统
- 结构化日志
- 多级日志
- 错误追踪
- 性能分析

## 贡献指南
1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 许可证
[MIT License](LICENSE)

## 联系方式
- 项目维护者：[Your Name]
- 邮箱：[Your Email]
- 问题反馈：[Issues Page]