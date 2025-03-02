#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
系统测试脚本
用于验证SBS系统的核心功能
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.data_collector import DataCollector
from src.analysis.llava_analyzer import LLaVAAnalyzer
from src.signal_generator import SignalGenerator

logger = setup_logger('system_test')

def test_data_collection():
    """测试数据收集功能"""
    logger.info("开始测试数据收集...")
    try:
        config = load_config()
        collector = DataCollector(config=config)
        data = collector.collect_market_data(
            symbol="BTCUSDT",
            interval="1d",
        )
        assert len(data) > 0, "数据收集失败：没有获取到数据"
        logger.info(f"数据收集成功：获取到 {len(data)} 条记录")
        return True
    except Exception as e:
        logger.error(f"数据收集测试失败: {str(e)}")
        return False

def test_market_analysis():
    """测试市场分析功能"""
    logger.info("开始测试市场分析...")
    try:
        config = load_config()
        analyzer = LLaVAAnalyzer(config=config)
        # 使用测试数据
        test_data = [
            {"timestamp": "2024-01-01", "open": 100, "high": 110, "low": 90, "close": 105},
            {"timestamp": "2024-01-02", "open": 105, "high": 115, "low": 95, "close": 110}
        ]
        analysis = analyzer.analyze_market_data(test_data)
        assert analysis is not None, "分析失败：没有生成分析结果"
        logger.info("市场分析测试成功")
        return True
    except Exception as e:
        logger.error(f"市场分析测试失败: {str(e)}")
        return False

def test_signal_generation():
    """测试信号生成功能"""
    logger.info("开始测试信号生成...")
    try:
        config = load_config()
        generator = SignalGenerator(config=config)
        # 使用测试分析结果
        test_analysis = [
            {
                "timestamp": "2024-01-01",
                "trend": "上升",
                "strength": 0.8,
                "confidence": 0.9
            }
        ]
        signals = generator.generate_signals(test_analysis)
        assert signals is not None, "信号生成失败：没有生成信号"
        logger.info(f"信号生成成功：生成了 {len(signals)} 个信号")
        return True
    except Exception as e:
        logger.error(f"信号生成测试失败: {str(e)}")
        return False

def main():
    """运行所有测试"""
    logger.info("开始系统测试...")
    
    # 记录测试结果
    results = {
        "数据收集": test_data_collection(),
        "市场分析": test_market_analysis(),
        "信号生成": test_signal_generation()
    }
    
    # 输出测试报告
    logger.info("\n测试报告:")
    for test_name, passed in results.items():
        status = "通过" if passed else "失败"
        logger.info(f"{test_name}: {status}")
    
    # 检查是否所有测试都通过
    all_passed = all(results.values())
    if all_passed:
        logger.info("所有测试通过！系统准备就绪。")
        return 0
    else:
        logger.error("部分测试失败，请检查日志获取详细信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 