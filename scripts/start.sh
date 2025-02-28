#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 帮助信息
show_help() {
    echo "SBS系统启动脚本"
    echo
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  --all                 启动所有服务"
    echo "  --service <name>      启动指定服务"
    echo "  --help               显示此帮助信息"
    echo
    echo "可用服务:"
    echo "  - web                Web界面"
    echo "  - bot                Discord机器人"
    echo "  - training           训练系统"
    echo "  - monitor            监控系统"
    echo "  - api                API服务"
}

# 环境检查
check_environment() {
    echo "检查环境..."
    
    # 检查Python版本
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if (( $(echo "$python_version >= 3.8" | bc -l) )); then
        echo -e "${GREEN}✓ Python版本检查通过: $python_version${NC}"
    else
        echo -e "${RED}✗ Python版本不满足要求: 需要3.8+${NC}"
        exit 1
    fi
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        echo -e "${GREEN}✓ CUDA版本检查通过: $cuda_version${NC}"
    else
        echo -e "${YELLOW}! 未检测到CUDA，将使用CPU模式${NC}"
    fi
    
    # 检查环境变量
    if [ ! -f ".env" ]; then
        echo -e "${RED}✗ 未找到.env文件${NC}"
        exit 1
    fi
    
    # 检查必要目录
    directories=(
        "data/raw"
        "data/processed"
        "data/validation"
        "models/llava-sbs"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            echo "创建目录: $dir"
            mkdir -p "$dir"
        fi
    done
}

# 启动Web服务
start_web() {
    echo "启动Web服务..."
    python -m src.web.app &
    echo $! > .web.pid
}

# 启动Discord机器人
start_bot() {
    echo "启动Discord机器人..."
    python -m src.bot.main &
    echo $! > .bot.pid
}

# 启动训练系统
start_training() {
    echo "启动训练系统..."
    python -m src.self_supervised.trainer.train &
    echo $! > .training.pid
}

# 启动监控系统
start_monitor() {
    echo "启动监控系统..."
    python -m src.monitor.main &
    echo $! > .monitor.pid
}

# 启动API服务
start_api() {
    echo "启动API服务..."
    python -m src.api.main &
    echo $! > .api.pid
}

# 启动所有服务
start_all() {
    start_web
    start_bot
    start_monitor
    start_api
    # 训练系统需要手动启动
    echo -e "${YELLOW}注意: 训练系统需要手动启动，使用 --service training${NC}"
}

# 停止服务
stop_service() {
    service=$1
    if [ -f ".$service.pid" ]; then
        pid=$(cat ".$service.pid")
        kill $pid 2>/dev/null
        rm ".$service.pid"
        echo "已停止 $service 服务"
    fi
}

# 停止所有服务
stop_all() {
    services=("web" "bot" "training" "monitor" "api")
    for service in "${services[@]}"; do
        stop_service $service
    done
}

# 主函数
main() {
    # 解析参数
    case "$1" in
        --help)
            show_help
            exit 0
            ;;
        --all)
            check_environment
            start_all
            ;;
        --service)
            if [ -z "$2" ]; then
                echo -e "${RED}错误: 未指定服务名称${NC}"
                show_help
                exit 1
            fi
            check_environment
            case "$2" in
                web)
                    start_web
                    ;;
                bot)
                    start_bot
                    ;;
                training)
                    start_training
                    ;;
                monitor)
                    start_monitor
                    ;;
                api)
                    start_api
                    ;;
                *)
                    echo -e "${RED}错误: 未知的服务 '$2'${NC}"
                    show_help
                    exit 1
                    ;;
            esac
            ;;
        --stop)
            if [ -z "$2" ]; then
                stop_all
            else
                stop_service "$2"
            fi
            ;;
        *)
            echo -e "${RED}错误: 未知的选项 '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
}

# 清理函数
cleanup() {
    echo "正在停止所有服务..."
    stop_all
    exit 0
}

# 注册清理函数
trap cleanup SIGINT SIGTERM

# 运行主函数
main "$@" 