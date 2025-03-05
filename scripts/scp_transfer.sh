#!/bin/bash

# SBS系统传输脚本（使用标准SSH/SCP） - 将本地SBS系统传输到远程服务器

# 服务器信息
SERVER_HOST="link.lanyun.net"
SERVER_PORT="32501"
SERVER_USER="root"
REMOTE_DIR="/root/sbs_system"

# 新增：本地接收目录
LOCAL_RECEIVE_DIR="/Users/jackyzhang/Desktop/sbs_system_received"

# 显示传输信息
echo "========================================"
echo "  SBS系统传输工具 (SCP版本)"
echo "========================================"
echo "将从本地传输到: ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}"
echo "端口: ${SERVER_PORT}"
echo "========================================"
echo "注意: 过程中会多次要求输入密码"
echo "密码: 0zlu9mqer9rikj72"
echo "========================================"

# 1. 创建远程目录结构
echo "正在创建远程目录结构..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${REMOTE_DIR}/{app,config,data/{raw,processed},docs,logs,models/{checkpoints,llava-sbs},prompts,scripts,src,tests}"

# 2. 使用tar打包本地文件，显式排除不需要的文件
echo "正在打包本地文件..."
tar --exclude=".git" \
    --exclude="venv" \
    --exclude=".venv" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".DS_Store" \
    --exclude="data/raw/*" \
    --exclude="models/checkpoints/*" \
    --exclude="sbs_system.tar.gz" \
    --exclude="logs/*" \
    --exclude=".pytest_cache" \
    --exclude="*.log" \
    -czf sbs_system.tar.gz \
    app/ \
    config/ \
    data/ \
    docs/ \
    logs/ \
    models/ \
    prompts/ \
    scripts/ \
    src/ \
    tests/ \
    requirements.txt \
    README.md

# 3. 使用SCP传输打包文件
echo "正在传输文件，请稍候..."
scp -P ${SERVER_PORT} sbs_system.tar.gz ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/

# 4. 在远程服务器上解压文件
echo "正在远程解压文件..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "cd ${REMOTE_DIR} && tar -xzf sbs_system.tar.gz && rm sbs_system.tar.gz"

# 5. 在远程服务器上设置正确的权限
echo "正在设置权限..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "cd ${REMOTE_DIR} && chmod +x scripts/*.py scripts/*.sh"

# 6. 清理本地临时文件
echo "清理本地临时文件..."
rm sbs_system.tar.gz

# 7. 在远程服务器上设置Python环境
echo "正在设置Python环境..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "cd ${REMOTE_DIR} && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt"

# 8. 配置WandB
echo "正在配置WandB..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "cd ${REMOTE_DIR} && \
    source venv/bin/activate && \
    wandb login 135573f7b891c2086c6e99c3c22fa3ec5543445d"

# 9. 创建必要的空目录和文件
echo "正在创建必要的目录和文件..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "cd ${REMOTE_DIR} && \
    mkdir -p data/raw data/processed && \
    mkdir -p models/checkpoints models/llava-sbs && \
    mkdir -p logs/training logs/self_supervised && \
    touch logs/training/.gitkeep logs/self_supervised/.gitkeep && \
    mkdir -p backups"

# 10. 运行服务器设置脚本
echo "正在运行服务器设置脚本..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "cd ${REMOTE_DIR} && ./scripts/server_setup.sh"

# 新增：从远程服务器传回所有内容
echo "正在从远程服务器传回所有内容..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "tar -czf - -C ${REMOTE_DIR} ." | tar -xzf - -C ${LOCAL_RECEIVE_DIR}

# 结束
echo "========================================"
echo "传输完成!"
echo "您现在可以通过SSH连接到服务器使用SBS系统:"
echo "ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
echo ""
echo "连接后，请执行以下命令启动系统:"
echo "cd ${REMOTE_DIR}"
echo "./start_sbs.sh"
echo ""
echo "要在本地查看训练状态，请使用以下命令设置SSH隧道:"
echo "ssh -L 8080:localhost:8080 -L 8000:localhost:8000 -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
echo "========================================" 