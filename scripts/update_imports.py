#!/usr/bin/env python
"""
导入路径更新脚本

此脚本用于更新项目中的Python导入路径，以适应新的目录结构。
它会递归遍历指定目录中的所有.py文件，并根据映射关系更新导入语句。
"""

import os
import re
import argparse
from pathlib import Path

# 导入路径映射：原路径 -> 新路径
PATH_MAPPINGS = {
    # 旧的模型路径 -> 新的结构
    'src.model.llava_analyzer': 'app.core.llava.analyzer',
    'src.model.evaluator.evaluator': 'src.analysis.evaluator',
    'src.model.analyzer': 'app.core.llava.analyzer',
    'src.model.model_optimizer': 'src.analysis.model_optimizer',
    'src.model.sbs_analyzer': 'src.analysis.sbs_analyzer',
    'LLaVAAnalyzer': 'LLaVAAnalyzer',  # 类名保持不变
    
    # 监控系统路径更新
    'src.monitor.system_monitor': 'monitoring.system_monitor',
    'src.monitor.discord_notifier': 'src.notification.discord_notifier',
    'src.monitor.performance_tracker': 'monitoring.performance_tracker',
    'SystemMonitor': 'SystemMonitor',  # 类名保持不变
    'MonitorConfig': 'MonitorConfig',  # 类名保持不变
    
    # 数据处理
    'src.data.collector': 'src.data.collector',
    'src.data.tradingview': 'src.data.tradingview',
    'src.data.data_loader': 'src.data.data_loader',
    'src.data.dataset': 'src.data.dataset',
    'DataLoader': 'DataLoader',  # 类名保持不变
    'DataCollector': 'DataCollector',  # 类名保持不变
    'CollectorConfig': 'CollectorConfig',  # 类名保持不变
    
    # 测试路径
    'test_proxy': 'tests.proxy.test_proxy',
    'test_discord_bot': 'tests.discord.test_discord_bot',
    'test_webhook': 'tests.webhook.test_webhook',
    'test_signal_flow': 'tests.signal.test_signal_flow',
    'test_llava': 'tests.llm.test_llava',
    'test_bot': 'tests.bot.test_bot',
    'test_system_e2e': 'tests.e2e.test_system_e2e',
    
    # 图像处理
    'src.image.processor': 'src.image.processor',  # 保持不变
    'ImageProcessor': 'ImageProcessor',  # 类名保持不变
    
    # 其他模块
    'src.preprocessing.data_processor': 'src.data.data_processor',
    'src.signal.signal_generator': 'src.analysis.signal_generator',
    'src.backtest.backtester': 'src.analysis.backtester',
    'src.backtest.portfolio': 'src.analysis.portfolio',
    'src.backtest.performance': 'src.analysis.performance',
    'SignalGenerator': 'SignalGenerator',  # 类名保持不变
    'SignalConfig': 'SignalConfig',  # 类名保持不变
    'Backtester': 'Backtester',  # 类名保持不变
    'BacktestConfig': 'BacktestConfig',  # 类名保持不变
    'Portfolio': 'Portfolio',  # 类名保持不变
    'PerformanceAnalyzer': 'PerformanceAnalyzer',  # 类名保持不变
    
    # 自监督学习模块
    'src.self_supervised.trainer.self_supervised_trainer': 'src.self_supervised.trainer.self_supervised_trainer',
    'src.self_supervised.model.sequence_model': 'src.self_supervised.model.sequence_model',
    'src.self_supervised.reinforcement.ppo_agent': 'src.self_supervised.reinforcement.ppo_agent',
    'src.self_supervised.reinforcement.trading_env': 'src.self_supervised.reinforcement.trading_env',
    'src.self_supervised.validator.sequence_evaluator': 'src.self_supervised.validator.sequence_evaluator',
    'src.self_supervised.validator.performance_validator': 'src.self_supervised.validator.performance_validator',
    'src.self_supervised.utils.output_formatter': 'src.self_supervised.utils.output_formatter',
    'src.self_supervised.data_generator.sequence_generator': 'src.self_supervised.data_generator.sequence_generator',
    'SelfSupervisedTrainer': 'SelfSupervisedTrainer',  # 类名保持不变
    'SequenceModel': 'SequenceModel',  # 类名保持不变
    'ModelConfig': 'ModelConfig',  # 类名保持不变
    'PPOAgent': 'PPOAgent',  # 类名保持不变
    'PPOConfig': 'PPOConfig',  # 类名保持不变
    'TradingEnvironment': 'TradingEnvironment',  # 类名保持不变
    'EnvConfig': 'EnvConfig',  # 类名保持不变
    'RewardConfig': 'RewardConfig',  # 类名保持不变
    'OutputFormatter': 'OutputFormatter',  # 类名保持不变
    'OutputRequirements': 'OutputRequirements',  # 类名保持不变
    'SequenceGenerator': 'SequenceGenerator',  # 类名保持不变
    
    # 主应用
    'src.main': 'src.main',  # 保持不变
    'SBSSystem': 'SBSSystem',  # 类名保持不变
    'SystemConfig': 'SystemConfig',  # 类名保持不变
    
    # 工具函数
    'src.utils.system_validator': 'src.utils.system_validator',
    'src.utils.trading_alerts': 'src.notification.trading_alerts',
    'src.utils.memory_monitor': 'monitoring.memory_monitor',
    'SystemValidator': 'SystemValidator',  # 类名保持不变
    'TradingAlertManager': 'TradingAlertManager',  # 类名保持不变
    'MemoryMonitor': 'MemoryMonitor',  # 类名保持不变
    
    # 调度器
    'src.scheduler.task_scheduler': 'src.scheduler.task_scheduler',
    'TaskScheduler': 'TaskScheduler',  # 类名保持不变
    
    # 配置
    'config.sbs_prompt': 'config.sbs_prompt',  # 保持不变
    'SBS_PROMPT': 'SBS_PROMPT',  # 常量保持不变
    
    # app 目录
    'app.core.discord.bot': 'app.core.discord.bot',  # 保持不变
    'app.utils.logger': 'app.utils.logger',  # 保持不变
    'run_bot': 'run_bot',  # 函数名保持不变
    'setup_logger': 'setup_logger',  # 函数名保持不变
    
    # sbs_bot 目录
    'sbs_bot.src.bot.discord_bot': 'sbs_bot.src.bot.discord_bot',  # 保持不变
    'sbs_bot.src.model.sbs_analyzer': 'sbs_bot.src.model.sbs_analyzer',  # 保持不变
    'sbs_bot.src.notification.discord_notifier': 'sbs_bot.src.notification.discord_notifier',  # 保持不变
    'sbs_bot.src.utils.logger': 'sbs_bot.src.utils.logger',  # 保持不变
}

# 导入语句匹配模式
IMPORT_PATTERNS = [
    r'from\s+([\w\.]+)\s+import',  # from xxx import
    r'import\s+([\w\.]+)'          # import xxx
]


def update_imports_in_file(file_path, dry_run=False):
    """更新单个文件中的导入语句
    
    Args:
        file_path: 文件路径
        dry_run: 是否只打印而不实际修改
        
    Returns:
        bool: 是否有更改
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 对每个匹配模式进行处理
    for pattern in IMPORT_PATTERNS:
        # 查找所有导入语句
        matches = re.finditer(pattern, content)
        
        # 替换导入路径
        for match in matches:
            import_path = match.group(1)
            
            # 检查是否需要更新
            for old_path, new_path in PATH_MAPPINGS.items():
                if import_path == old_path or import_path.startswith(old_path + '.'):
                    # 创建替换后的路径
                    new_import_path = import_path.replace(old_path, new_path, 1)
                    
                    # 替换导入语句
                    old_import = match.group(0)
                    new_import = old_import.replace(import_path, new_import_path, 1)
                    
                    if dry_run:
                        print(f"Would replace: {old_import} -> {new_import} in {file_path}")
                    else:
                        content = content.replace(old_import, new_import)
    
    # 检查是否有更改
    has_changes = content != original_content
    
    # 如果有更改且不是dry_run，则写回文件
    if has_changes and not dry_run:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated imports in {file_path}")
    
    return has_changes


def update_imports_in_directory(directory, dry_run=False):
    """递归更新目录中所有Python文件的导入语句
    
    Args:
        directory: 目录路径
        dry_run: 是否只打印而不实际修改
        
    Returns:
        int: 更新的文件数量
    """
    directory = Path(directory)
    updated_files = 0
    
    for path in directory.glob('**/*.py'):
        try:
            if update_imports_in_file(path, dry_run):
                updated_files += 1
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    return updated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='更新Python文件中的导入路径')
    parser.add_argument('directories', metavar='DIR', type=str, nargs='+',
                      help='要处理的目录')
    parser.add_argument('--dry-run', action='store_true',
                      help='只打印要更改的内容，不实际修改文件')
    
    args = parser.parse_args()
    
    total_updated = 0
    for directory in args.directories:
        print(f"\n处理目录: {directory}")
        updated = update_imports_in_directory(directory, args.dry_run)
        total_updated += updated
        print(f"在 {directory} 中{'发现' if args.dry_run else '更新了'} {updated} 个文件")
    
    action = "发现需要更新" if args.dry_run else "更新了"
    print(f"\n总共{action} {total_updated} 个文件") 