#!/usr/bin/env python
"""
内存管理和分批处理示例
展示如何使用TradeResultTracker的内存管理和分批处理功能处理大量交易数据
"""

import sys
import os
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path
import time
import gc
import argparse
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.self_supervised.utils.trade_tracker import TradeResultTracker

def generate_large_trade_dataset(num_trades=10000, output_file=None):
    """生成大量模拟交易数据用于测试内存管理"""
    print(f"生成{num_trades}笔交易数据...")
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT']
    timeframes = ['5m', '15m', '1h', '4h', 'daily']
    
    start_date = datetime.now() - timedelta(days=365)
    
    trades = []
    for i in tqdm(range(num_trades)):
        # 创建随机交易数据
        symbol = random.choice(symbols)
        direction = random.choice(['多', '空'])
        entry_price = random.uniform(100, 10000)
        
        # 设置止损止盈
        if direction == '多':
            stop_loss = entry_price * (1 - random.uniform(0.01, 0.05))
            take_profit = entry_price * (1 + random.uniform(0.02, 0.1))
        else:
            stop_loss = entry_price * (1 + random.uniform(0.01, 0.05))
            take_profit = entry_price * (1 - random.uniform(0.02, 0.1))
            
        # 设置时间
        entry_time = start_date + timedelta(minutes=random.randint(0, 525600))  # 随机一年内
        
        # 模拟交易持续时间
        duration_minutes = random.randint(5, 4320)  # 5分钟到3天
        exit_time = entry_time + timedelta(minutes=duration_minutes)
        
        # 设置结果
        is_win = random.random() < 0.5
        
        if is_win:
            # 盈利交易
            if direction == '多':
                exit_price = take_profit * random.uniform(0.9, 1.0)
            else:
                exit_price = take_profit * random.uniform(1.0, 1.1)
            exit_reason = 'take_profit'
            profit_percentage = abs((exit_price - entry_price) / entry_price * 100)
            if direction == '空':
                profit_percentage = -profit_percentage
        else:
            # 亏损交易
            if direction == '多':
                exit_price = stop_loss * random.uniform(0.9, 1.0)
            else:
                exit_price = stop_loss * random.uniform(1.0, 1.1)
            exit_reason = 'stop_loss'
            profit_percentage = -abs((exit_price - entry_price) / entry_price * 100)
            if direction == '空':
                profit_percentage = -profit_percentage
                
        # 添加SBS数据
        has_sbs = random.random() < 0.7  # 70%的交易有SBS数据
        
        metadata = {}
        if has_sbs:
            sbs_data = {
                'liquidation': random.random() < 0.3,
                'double_pattern': random.random() < 0.4,
                'sce': random.random() < 0.3,
                'complete_sequence': random.random() < 0.5
            }
            metadata['sbs_data'] = sbs_data
            
        # 创建交易记录
        trade = {
            'trade_id': f"trade_{i+1}",
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': entry_time.isoformat(),
            'exit_time': exit_time.isoformat(),
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'profit_percentage': profit_percentage,
            'status': 'closed',
            'timeframe': random.choice(timeframes),
            'risk_percentage': random.uniform(0.5, 2.0),
            'metadata': metadata,
            'max_profit_percentage': profit_percentage if is_win else profit_percentage * random.uniform(0.2, 0.8),
            'max_drawdown': abs(profit_percentage) * random.uniform(0.1, 0.5) if not is_win else 0,
            'duration': duration_minutes
        }
        
        trades.append(trade)
    
    # 如果指定了输出文件，保存数据
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(trades, f, ensure_ascii=False, indent=2)
        
        print(f"交易数据已保存至: {output_path}")
    
    return trades

def test_memory_management(trades, batch_size=1000):
    """测试内存管理功能"""
    print("\n测试内存管理功能...")
    
    # 创建带内存管理的交易跟踪器
    config = {
        'memory_management_enabled': True,
        'max_trades_in_memory': 2000,
        'trade_archive_threshold': 500,
        'cleanup_interval': 1000,
        'storage_dir': 'data/memory_test'
    }
    
    tracker = TradeResultTracker(config)
    
    # 记录初始内存使用
    initial_memory = get_memory_usage()
    print(f"初始内存使用: {initial_memory:.2f} MB")
    
    # 使用分批处理添加交易
    print(f"分批导入{len(trades)}笔交易，批次大小: {batch_size}...")
    
    # 定义批处理函数
    def process_batch(batch):
        for trade in batch:
            # 添加交易（已关闭的）
            trade_copy = trade.copy()
            tracker.completed_trades.append(trade_copy)
            tracker.trade_history.append(trade_copy)
            
            # 每添加一批交易后检查内存清理
            tracker.trade_count_since_cleanup += 1
        
        # 检查是否需要清理
        if tracker.trade_count_since_cleanup >= tracker.cleanup_interval:
            tracker._memory_cleanup()
            
        return len(batch)
    
    # 开始计时
    start_time = time.time()
    
    # 分批处理
    total_processed = sum(tracker.process_trades_in_batches(trades, batch_size, process_batch))
    
    # 结束计时
    elapsed = time.time() - start_time
    
    # 记录最终内存使用
    final_memory = get_memory_usage()
    
    print(f"导入完成！共处理{total_processed}笔交易，耗时: {elapsed:.2f}秒")
    print(f"最终内存使用: {final_memory:.2f} MB")
    print(f"内存增长: {final_memory - initial_memory:.2f} MB")
    print(f"内存中保留的交易数: {len(tracker.completed_trades)}")
    print(f"已归档的交易数: {total_processed - len(tracker.completed_trades)}")
    
    # 测试加载归档
    print("\n测试加载归档交易...")
    start_time = time.time()
    loaded_count = tracker.load_archived_trades()
    elapsed = time.time() - start_time
    
    print(f"加载归档完成，共加载{loaded_count}笔交易，耗时: {elapsed:.2f}秒")
    print(f"当前内存中交易数: {len(tracker.completed_trades)}")
    
    # 测试按日期加载归档
    if loaded_count > 0:
        print("\n测试按日期范围加载归档...")
        # 清空当前交易
        tracker.completed_trades = []
        tracker._memory_cleanup()
        
        # 生成一个3个月的日期范围
        six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        start_time = time.time()
        loaded_count = tracker.load_archived_trades(
            start_date=six_months_ago, 
            end_date=three_months_ago
        )
        elapsed = time.time() - start_time
        
        print(f"按日期范围加载完成，共加载{loaded_count}笔交易，耗时: {elapsed:.2f}秒")
        print(f"当前内存中交易数: {len(tracker.completed_trades)}")
    
    return tracker

def test_batch_processing(trades, batch_size=1000):
    """测试分批处理功能"""
    print("\n测试分批处理功能...")
    
    # 创建标准交易跟踪器
    tracker = TradeResultTracker()
    
    # 创建耗时处理函数
    def slow_processing(batch):
        result = {
            'total': len(batch),
            'profitable': sum(1 for t in batch if t.get('profit_percentage', 0) > 0),
            'avg_profit': sum(t.get('profit_percentage', 0) for t in batch) / len(batch) if batch else 0
        }
        # 模拟耗时处理
        time.sleep(0.01)  
        return result
    
    # 开始计时
    start_time = time.time()
    
    # 分批处理并合并结果
    results = tracker.process_trades_in_batches(trades, batch_size, slow_processing)
    
    # 结束计时
    elapsed = time.time() - start_time
    
    print(f"分批处理完成，共处理{len(trades)}笔交易，批次大小: {batch_size}，耗时: {elapsed:.2f}秒")
    print(f"批次数量: {len(results)}")
    print(f"总盈利交易数: {sum(r.get('profitable', 0) for r in results)}")
    
    # 对比不使用分批的处理时间
    print("\n对比不使用分批的处理时间...")
    start_time = time.time()
    
    # 直接处理所有交易
    result = slow_processing(trades)
    
    elapsed_no_batch = time.time() - start_time
    
    print(f"不分批处理耗时: {elapsed_no_batch:.2f}秒")
    print(f"速度提升: {elapsed_no_batch / max(0.001, elapsed):.2f}倍")
    
    return results

def get_memory_usage():
    """获取当前内存使用量（MB）"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # 转换为MB

def run_demos():
    """运行所有演示"""
    parser = argparse.ArgumentParser(description='TradeResultTracker内存管理和分批处理演示')
    parser.add_argument('--num_trades', type=int, default=10000, help='生成的交易数量')
    parser.add_argument('--batch_size', type=int, default=500, help='处理批次大小')
    parser.add_argument('--save_data', action='store_true', help='保存生成的交易数据')
    args = parser.parse_args()
    
    print("=" * 50)
    print("TradeResultTracker内存管理和分批处理演示")
    print("=" * 50)
    
    # 生成测试数据
    data_file = 'data/memory_test/large_trade_dataset.json' if args.save_data else None
    trades = generate_large_trade_dataset(args.num_trades, data_file)
    
    # 测试内存管理
    tracker = test_memory_management(trades, args.batch_size)
    
    # 测试分批处理
    test_batch_processing(trades[:5000], args.batch_size)  # 使用部分数据以加快测试
    
    print("\n演示完成！")

if __name__ == "__main__":
    run_demos() 