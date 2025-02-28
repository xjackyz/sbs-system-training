# 部署指南

## 部署步骤

### 1. 环境准备
确保您的服务器满足以下要求：
- **操作系统**: Linux (推荐 Ubuntu 20.04 或更高版本)
- **硬件要求**:
  - GPU: 显存 ≥ 24GB (推荐 A5000, A6000, A100)
  - CPU: 8核或以上
  - 内存: 32GB或以上
  - 存储: 100GB或以上SSD

### 2. 安装依赖
在服务器上安装必要的依赖：
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

### 3. 克隆代码仓库
```bash
git clone https://github.com/your-repo/sbs_system.git
cd sbs_system
```

### 4. 创建虚拟环境并激活
```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. 安装 Python 依赖
```bash
pip install -r requirements.txt
```

### 6. 配置环境变量
复制 `.env.example` 文件为 `.env` 并根据需要进行修改。

### 7. 数据准备
确保数据目录结构正确，并将训练和验证数据放置在相应的目录中。

### 8. 启动服务
```bash
python scripts/train.py
```

## 注意事项
- 确保在生产环境中使用合适的配置，避免使用开发模式。
- 定期检查系统日志，确保系统正常运行。
- 监控系统性能，及时处理潜在问题。 