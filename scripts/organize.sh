#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "开始整理SBS系统目录..."

# 创建新的临时目录
temp_dir="sbs_system_new"
mkdir -p $temp_dir

# 创建标准目录结构
directories=(
    # 核心功能模块
    "src/self_supervised"      # 自监督学习
    "src/notification"         # 通知系统
    "src/utils"               # 工具函数
    "src/bot"                 # Discord机器人
    "src/web"                 # Web界面
    "src/api"                 # API服务
    "src/monitor"             # 监控系统
    "src/analysis"            # 分析模块
    "src/trading"             # 交易模块
    
    # 配置目录
    "config/training"         # 训练配置
    "config/model"            # 模型配置
    "config/system"           # 系统配置
    "config/bot"             # 机器人配置
    
    # 数据目录
    "data/raw"               # 原始数据
    "data/processed"         # 处理后数据
    "data/validation"        # 验证数据
    "data/charts"           # 图表数据
    "data/signals"          # 信号数据
    
    # 模型目录
    "models/llava-sbs"       # LLaVA模型
    "models/checkpoints"     # 检查点
    
    # 提示词目录
    "prompts/base"           # 基础提示词
    "prompts/advanced"       # 高级提示词
    "prompts/templates"      # 模板
    
    # Web应用
    "app/static"            # 静态文件
    "app/templates"         # 模板文件
    "app/components"        # 组件
    
    # 其他目录
    "docs/api"              # API文档
    "docs/guides"           # 使用指南
    "tests/unit"            # 单元测试
    "tests/integration"     # 集成测试
    "logs/training"         # 训练日志
    "logs/system"           # 系统日志
)

# 创建目录
for dir in "${directories[@]}"; do
    mkdir -p "$temp_dir/$dir"
done

# 复制核心文件
echo "复制核心文件..."

# 1. 复制源代码
if [ -d "src" ]; then
    cp -r src/* "$temp_dir/src/" 2>/dev/null || true
fi

# 2. 复制Web应用
if [ -d "app" ]; then
    cp -r app/* "$temp_dir/app/" 2>/dev/null || true
fi

# 3. 复制机器人代码
if [ -d "sbs_bot" ]; then
    cp -r sbs_bot/* "$temp_dir/src/bot/" 2>/dev/null || true
fi

# 4. 复制配置文件
if [ -d "config" ]; then
    cp -r config/* "$temp_dir/config/" 2>/dev/null || true
fi

# 5. 复制监控代码
if [ -d "monitoring" ]; then
    cp -r monitoring/* "$temp_dir/src/monitor/" 2>/dev/null || true
fi

# 6. 复制其他目录
directories_to_copy=(
    "scripts"
    "docs"
    "tests"
    "prompts"
)

for dir in "${directories_to_copy[@]}"; do
    if [ -d "$dir" ]; then
        cp -r "$dir"/* "$temp_dir/$dir/" 2>/dev/null || true
    fi
done

# 7. 复制重要文件
important_files=(
    "setup.py"
    "requirements.txt"
    ".env.example"
    "README.md"
    "migration_checklist.md"
    "LICENSE"
    ".gitignore"
    "Dockerfile"
    "docker-compose.yml"
)

for file in "${important_files[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$temp_dir/"
    fi
done

# 清理临时文件
echo "清理临时文件..."
find "$temp_dir" -name "*.pyc" -delete
find "$temp_dir" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$temp_dir" -name ".DS_Store" -delete
find "$temp_dir" -name "*.log" -delete
find "$temp_dir" -name "*.tmp" -delete

# 移动原目录到备份
timestamp=$(date +%Y%m%d_%H%M%S)
backup_dir="../sbs_system_backup_$timestamp"
echo "创建备份: $backup_dir"
mv ../"$(basename $(pwd))" "$backup_dir"

# 移动新目录到原位置
mv "$temp_dir" ../"$(basename $(pwd))"

echo -e "${GREEN}目录整理完成！${NC}"
echo "原目录已备份到: $backup_dir"
echo "新的整理后的目录已创建"

# 创建说明文件
cat > README_FIRST.md << EOL
# SBS系统功能说明

## 系统架构
\`\`\`
sbs_system/
├── src/                    # 源代码
│   ├── self_supervised/   # 自监督学习模块
│   ├── notification/      # 通知系统
│   ├── utils/            # 工具函数
│   ├── bot/              # Discord机器人
│   ├── web/              # Web界面
│   ├── api/              # API服务
│   ├── monitor/          # 监控系统
│   ├── analysis/         # 分析模块
│   └── trading/          # 交易模块
├── app/                   # Web应用
│   ├── static/           # 静态资源
│   ├── templates/        # 页面模板
│   └── components/       # UI组件
├── config/               # 配置文件
│   ├── training/        # 训练配置
│   ├── model/           # 模型配置
│   ├── system/          # 系统配置
│   └── bot/             # 机器人配置
├── data/                 # 数据目录
│   ├── raw/             # 原始数据
│   ├── processed/       # 处理后数据
│   ├── validation/      # 验证数据
│   ├── charts/          # 图表数据
│   └── signals/         # 信号数据
└── [其他目录...]         # 其他支持目录
\`\`\`

## 功能模块

1. **自监督学习模块**
   - 模型训练和优化
   - 数据预处理
   - 验证和评估

2. **交易分析系统**
   - 市场结构分析
   - 信号生成
   - 风险评估

3. **实时监控系统**
   - 性能监控
   - 资源使用监控
   - 告警系统

4. **通知系统**
   - Discord集成
   - 实时通知
   - 状态报告

5. **Web界面**
   - 数据可视化
   - 系统控制面板
   - 实时监控

6. **API服务**
   - RESTful API
   - WebSocket支持
   - 数据接口

## 快速开始

1. 环境配置：
   \`\`\`bash
   cp .env.example .env
   # 编辑 .env 文件设置必要的环境变量
   \`\`\`

2. 安装依赖：
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. 启动服务：
   \`\`\`bash
   # 启动所有服务
   ./scripts/start.sh
   
   # 或单独启动特定服务
   ./scripts/start.sh --service web
   ./scripts/start.sh --service bot
   ./scripts/start.sh --service training
   \`\`\`

## 配置说明

1. **训练配置**
   - 编辑 \`config/training/config.py\`
   - 设置GPU和模型参数
   - 配置数据路径

2. **系统配置**
   - 编辑 \`config/system/config.py\`
   - 设置服务端口
   - 配置日志级别

3. **Discord配置**
   - 编辑 \`config/bot/config.py\`
   - 设置Bot Token
   - 配置通知规则

## 注意事项
- 确保模型文件已放入正确位置
- 检查数据目录权限
- 确保所有服务端口可用
- 定期备份重要数据
EOL

echo "已创建说明文件: README_FIRST.md" 