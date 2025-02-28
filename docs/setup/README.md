# SBS系统安装指南

本文档提供SBS交易分析系统的安装和设置说明。

## 系统要求

### 硬件要求
- CPU: 4核或更多
- RAM: 16GB或更多（推荐32GB）
- GPU: NVIDIA GPU，至少12GB显存（推荐16GB+）
- 存储: 最少50GB可用空间

### 软件要求
- 操作系统: Ubuntu 20.04+，macOS 12+，或Windows 10/11
- CUDA: 11.7或更高版本（如使用NVIDIA GPU）
- Python: 3.10或更高版本
- Docker（可选）: 20.10.x或更高版本

## 安装方法

### 方法1：使用Docker（推荐）

1. 安装Docker和Docker Compose
```bash
# Ubuntu
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl enable docker
sudo systemctl start docker

# 添加当前用户到docker组
sudo usermod -aG docker $USER
# 需要重新登录生效
```

2. 克隆仓库
```bash
git clone https://github.com/yourusername/sbs_system.git
cd sbs_system
```

3. 创建环境变量文件
```bash
cp .env.example .env
# 编辑.env文件，设置必要的环境变量
nano .env
```

4. 启动Docker容器
```bash
docker-compose up -d
```

### 方法2：直接安装

1. 安装Python依赖
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或者 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

2. 下载模型（可选）
```bash
# 创建模型目录
mkdir -p models
cd models

# 下载LLaVA模型（示例）
git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1.6-7b
```

3. 设置配置
```bash
cp .env.example .env
# 编辑.env文件
nano .env
```

4. 运行应用
```bash
python run.py
```

## 配置说明

### 主要配置文件
- `.env`: 环境变量配置
- `config/system/system_config.yaml`: 系统配置
- `config/training/training_config.yaml`: 训练配置
- `config/bot/bot_config.yaml`: 机器人配置

### 配置示例

#### 系统配置（system_config.yaml）

```yaml
system:
  name: SBS Trading Analysis System
  version: 1.0.0
  debug: false
  log_level: INFO
  
model:
  path: models/llava-v1.6-7b
  max_tokens: 1024
  temperature: 0.7
  
api:
  host: 0.0.0.0
  port: 5000
  allow_cors: true
```

## 验证安装

安装完成后，可以通过以下步骤验证系统是否正常工作：

1. 检查系统状态
```bash
python scripts/check_system.py
```

2. 运行测试脚本
```bash
python tests/run_tests.py --basic
```

3. 打开Web界面
在浏览器中访问 `http://localhost:5000`（或您配置的其他端口）

## 常见问题

### 如果遇到CUDA错误

确认CUDA版本与PyTorch兼容：
```bash
python -c "import torch; print(torch.version.cuda)"
```

### 如果内存不足

调整配置文件中的内存限制参数：
```yaml
# 在system_config.yaml中
resources:
  max_memory: "8G"  # 根据您的系统调整
```

## 下一步

成功安装后，请参阅[使用指南](../usage/README.md)了解如何使用系统。 