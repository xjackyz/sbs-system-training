#!/usr/bin/env python
"""
探索机制示例
展示如何使用TradeResultTracker的探索机制进行交易
"""

import sys
import os
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.self_supervised.utils.trade_tracker import TradeResultTracker

def generate_random_price_series(start_price=100, length=100, volatility=0.02):
    """生成随机价格序列，用于模拟交易"""
    price_series = [start_price]
    for i in range(1, length):
        change = np.random.normal(0, volatility)
        # 加入一些趋势
        if i < length / 2:
            change += 0.001  # 上升趋势
        else:
            change -= 0.001  # 下降趋势
        new_price = price_series[-1] * (1 + change)
        price_series.append(new_price)
    return price_series

def create_random_trade(tracker, trade_id, price_series, index):
    """创建随机交易"""
    current_price = price_series[index]
    direction = random.choice(['多', '空'])
    
    # 随机设置止损和止盈
    if direction == '多':
        stop_loss = current_price * (1 - random.uniform(0.01, 0.05))
        take_profit = current_price * (1 + random.uniform(0.02, 0.1))
    else:
        stop_loss = current_price * (1 + random.uniform(0.01, 0.05))
        take_profit = current_price * (1 - random.uniform(0.02, 0.1))
    
    # 创建交易
    trade = tracker.add_trade(
        trade_id=trade_id,
        symbol='EXAMPLE',
        direction=direction,
        entry_price=current_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_percentage=1.0,
        metadata={'index': index}
    )
    
    return trade

def simulate_price_movement(tracker, trade, price_series, start_index, max_steps=20):
    """模拟价格移动并更新交易状态"""
    current_index = start_index
    max_index = min(start_index + max_steps, len(price_series) - 1)
    
    while current_index < max_index:
        current_index += 1
        current_price = price_series[current_index]
        
        # 更新交易状态
        updated_trade = tracker.update_trade(
            trade_id=trade['trade_id'],
            current_price=current_price,
            current_time=datetime.now().isoformat()
        )
        
        # 如果交易已关闭，退出循环
        if updated_trade is None:
            break
    
    # 如果交易仍然活跃，手动关闭它
    if trade['trade_id'] in tracker.active_trades:
        tracker.close_trade(
            trade_id=trade['trade_id'],
            exit_price=price_series[current_index],
            exit_reason='manual_close',
            exit_time=datetime.now().isoformat()
        )

def main():
    # 配置探索参数
    config = {
        'storage_dir': 'data/exploration_example',
        'exploration_enabled': True,
        'exploration_rate': 0.3,  # 30%的交易会应用探索策略
        'exploration_decay': 0.98,  # 探索率衰减
        'min_exploration_rate': 0.05  # 最小探索率
    }
    
    # 创建交易跟踪器
    tracker = TradeResultTracker(config)
    
    # 生成随机价格序列
    price_series = generate_random_price_series(length=500)
    
    # 执行多次交易以比较标准交易和探索交易
    num_trades = 100
    
    print(f"开始执行{num_trades}次交易...")
    print(f"初始探索率: {tracker.exploration_rate:.2f}")
    
    # 执行交易
    for i in range(num_trades):
        # 创建交易
        trade_id = f"TRADE-{i+1:04d}"
        trade = create_random_trade(tracker, trade_id, price_series, i)
        
        # 模拟价格移动
        simulate_price_movement(tracker, trade, price_series, i)
        
        # 每10次交易打印一次状态
        if (i + 1) % 10 == 0:
            print(f"已完成 {i+1}/{num_trades} 次交易，当前探索率: {tracker.exploration_rate:.4f}")
    
    # 生成详细报告
    report_path = tracker.generate_detailed_report()
    
    # 打印统计信息
    stats = tracker.get_trade_stats()
    exploration_metrics = tracker.get_exploration_metrics()
    exploration_analysis = tracker._analyze_exploration_effectiveness()
    
    print("\n===== 交易统计 =====")
    print(f"总交易数: {stats['completed_trades']}")
    print(f"整体胜率: {stats['win_rate']:.2f}")
    print(f"平均利润: {stats['avg_profit']:.2f}%")
    print(f"利润因子: {stats['profit_factor']:.2f}")
    
    print("\n===== 探索统计 =====")
    print(f"探索交易数: {exploration_metrics['total_explorations']}")
    print(f"成功探索数: {exploration_metrics['successful_explorations']}")
    print(f"探索成功率: {exploration_metrics['success_rate']:.2f}")
    print(f"最终探索率: {exploration_metrics['exploration_rate']:.4f}")
    
    print("\n===== 探索VS标准 =====")
    if 'exploration_win_rate' in exploration_analysis and 'standard_win_rate' in exploration_analysis:
        print(f"探索交易胜率: {exploration_analysis['exploration_win_rate']:.2f}")
        print(f"标准交易胜率: {exploration_analysis['standard_win_rate']:.2f}")
        print(f"胜率差异: {exploration_analysis['exploration_benefit']['win_rate_difference']:.4f}")
        print(f"利润差异: {exploration_analysis['exploration_benefit']['profit_difference']:.4f}%")
        print(f"总体收益比: {exploration_analysis['exploration_benefit']['overall_benefit']:.4f}")
    
    # 按类型分析探索效果
    if 'exploration_by_type' in exploration_analysis:
        print("\n===== 探索类型分析 =====")
        for e_type, data in exploration_analysis['exploration_by_type'].items():
            if data['count'] > 0:
                print(f"{e_type}: 数量={data['count']}, 胜率={data['win_rate']:.2f}, 平均利润={data['avg_profit']:.2f}%")
    
    print(f"\n详细报告已保存至: {report_path}")
    return exploration_analysis

if __name__ == "__main__":
    main() 