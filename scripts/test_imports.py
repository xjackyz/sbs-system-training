#!/usr/bin/env python
"""
导入路径测试脚本

此脚本用于测试导入路径更新是否成功。
包含各种旧格式的导入语句，用于验证update_imports.py脚本的功能。
"""

# 添加当前目录到Python路径
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 测试导入 - 应该被更新的路径
from app.core.llava.analyzer import LLaVAAnalyzer
from monitoring.system_monitor import SystemMonitor, MonitorConfig
from src.data.data_loader import DataLoader
from src.analysis.backtester import Backtester, BacktestConfig
from src.analysis.signal_generator import SignalGenerator
from src.utils.logger import setup_logger
import src.analysis.sbs_analyzer

# 测试导入 - 不需要更新的标准库
import json
from pathlib import Path
from typing import Dict, List, Optional

def test_imports():
    """测试导入是否正常工作"""
    print("测试导入路径...")
    
    # 创建各种类的实例
    analyzer = LLaVAAnalyzer({})
    monitor = SystemMonitor({})
    loader = DataLoader({})
    backtester = Backtester({})
    signal_gen = SignalGenerator({})
    
    print("所有导入都正常工作！")
    
    # 返回实例列表，避免未使用警告
    return [analyzer, monitor, loader, backtester, signal_gen]

if __name__ == "__main__":
    # 设置日志
    logger = setup_logger("import_test")
    logger.info("开始导入测试")
    
    try:
        instances = test_imports()
        logger.info(f"成功导入和实例化了 {len(instances)} 个类")
        print("测试成功!")
    except Exception as e:
        logger.error(f"导入测试失败: {e}")
        print(f"测试失败: {e}") 