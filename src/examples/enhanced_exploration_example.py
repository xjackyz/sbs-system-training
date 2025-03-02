#!/usr/bin/env python
"""
增强型探索机制示例
展示如何使用TradeResultTracker的强化学习探索机制进行交易
"""

import sys
import os
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.self_supervised.utils.trade_tracker import TradeResultTracker
from src.self_supervised.utils.exploration_config import ExplorationConfig

def generate_market_data(num_days=30, volatility=0.02, trend=0.001):
    """生成模拟市场数据"""
    print(f"生成{num_days}天的市场数据...")
    
    # 生成价格序列
    prices = [100.0]
    volumes = []
    
    # 创建多个市场条件
    market_conditions = [
        {'name': '牛市', 'days': num_days // 3, 'base_trend': 0.002, 'base_volatility': 0.02},
        {'name': '震荡市', 'days': num_days // 3, 'base_trend': 0.0, 'base_volatility': 0.03},
        {'name': '熊市', 'days': num_days - (num_days // 3) * 2, 'base_trend': -0.002, 'base_volatility': 0.025}
    ]
    
    market_data = []
    
    for condition in market_conditions:
        condition_trend = condition['base_trend']
        condition_volatility = condition['base_volatility']
        
        for day in range(condition['days']):
            # 生成当天的价格
            price_change = np.random.normal(condition_trend, condition_volatility)
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
            
            # 生成成交量
            volume = random.uniform(500, 5000)
            volumes.append(volume)
            
            # 计算简单移动平均线
            if len(prices) >= 20:
                sma20 = sum(prices[-20:]) / 20
            else:
                sma20 = new_price
                
            if len(prices) >= 200:
                sma200 = sum(prices[-200:]) / 200
            else:
                sma200 = new_price
            
            # 创建市场数据点
            market_data.append({
                'day': len(prices) - 1,
                'price': new_price,
                'volume': volume,
                'volatility': condition_volatility,
                'market_condition': condition['name'],
                'timestamp': (datetime.now() - timedelta(days=num_days-len(prices)+1)).isoformat(),
                'sma20': sma20,
                'sma200': sma200,
                'price_to_sma20': new_price / sma20 if sma20 > 0 else 1.0,
                'price_to_sma200': new_price / sma200 if sma200 > 0 else 1.0,
                'sma20_to_sma200': sma20 / sma200 if sma200 > 0 else 1.0
            })
    
    return market_data

def simulate_trading(tracker, market_data, strategy_type="standard"):
    """模拟交易过程"""
    print(f"使用{strategy_type}策略开始模拟交易...")
    
    # 模拟交易参数
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    timeframes = ['15m', '1h', '4h']
    
    trades_executed = 0
    
    # 遍历市场数据，模拟交易决策
    for day, data in enumerate(market_data):
        # 每天有50%的概率考虑交易
        if random.random() < 0.5:
            continue
            
        # 模拟市场信号检测，30%的概率出现交易信号
        if random.random() < 0.3:
            # 决定是否进行探索
            is_exploration = tracker.should_explore(data)
            market_condition = data['market_condition']
            
            # 根据市场条件决定策略
            if market_condition == '牛市':
                direction_prob = 0.7  # 70%几率做多
            elif market_condition == '熊市':
                direction_prob = 0.3  # 30%几率做多
            else:
                direction_prob = 0.5  # 50%几率做多
            
            # 如果是探索交易，可能反向操作
            if is_exploration:
                # 探索时有30%的概率反向操作
                if random.random() < 0.3:
                    direction_prob = 1 - direction_prob
            
            # 确定方向
            direction = '多' if random.random() < direction_prob else '空'
            
            # 生成交易ID
            trade_id = f"trade_{strategy_type}_{day+1:04d}"
            
            # 当前价格
            price = data['price']
            
            # 设置止损止盈
            if direction == '多':
                stop_loss = price * (1 - random.uniform(0.01, 0.05))
                take_profit = price * (1 + random.uniform(0.03, 0.1))
            else:
                stop_loss = price * (1 + random.uniform(0.01, 0.05))
                take_profit = price * (1 - random.uniform(0.03, 0.1))
            
            # 随机生成SBS序列点位
            sequence_points = {
                'point1': price * (1 - random.uniform(0.05, 0.1) if direction == '多' else -random.uniform(0.05, 0.1)),
                'point2': price * (1 + random.uniform(0.1, 0.15) if direction == '多' else -random.uniform(0.1, 0.15)),
                'point3': price * (1 - random.uniform(0.03, 0.08) if direction == '多' else -random.uniform(0.03, 0.08)),
                'point4': price * (1 - random.uniform(0.01, 0.03) if direction == '多' else -random.uniform(0.01, 0.03)),
                'point5': price * (1 + random.uniform(0.15, 0.25) if direction == '多' else -random.uniform(0.15, 0.25))
            }
            
            # 随机确认信号
            confirmation_signals = {
                'sce': random.random() < 0.6,
                'liquidation': random.random() < 0.4,
                'double_pattern': random.random() < 0.3,
                'strength': random.uniform(0.3, 0.9)
            }
            
            # 添加元数据
            metadata = {
                'market_condition': market_condition,
                'market_data': data,
                'signal_strength': random.uniform(0.3, 0.9),
                'exploration_type': random.choice(['direction_change', 'stop_loss_adjust', 'take_profit_adjust', 'risk_adjust']) if is_exploration else None
            }
            
            # 添加交易
            trade = tracker.add_trade(
                trade_id=trade_id,
                symbol=random.choice(symbols),
                direction=direction,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=data['timestamp'],
                timeframe=random.choice(timeframes),
                risk_percentage=random.uniform(0.5, 2.0),
                sequence_points=sequence_points,
                confirmation_signal=confirmation_signals,
                market_data=data,
                is_exploration=is_exploration
            )
            
            trades_executed += 1
            
            # 模拟交易结果
            # 假设未来3-10个时间单位后交易结束
            future_days = min(random.randint(3, 10), len(market_data) - day - 1)
            if future_days > 0:
                future_data = market_data[day + future_days]
                
                # 计算价格变化
                price_change_percent = (future_data['price'] - price) / price * 100
                
                # 根据方向和市场情况调整利润
                if direction == '多':
                    expected_profit = price_change_percent
                else:
                    expected_profit = -price_change_percent
                
                # 探索交易的结果可能更加极端
                if is_exploration:
                    variation = random.uniform(0.7, 1.3)
                    expected_profit *= variation
                
                # 市场条件影响
                if market_condition == '牛市' and direction == '多':
                    expected_profit *= random.uniform(1.0, 1.2)
                elif market_condition == '熊市' and direction == '空':
                    expected_profit *= random.uniform(1.0, 1.2)
                
                # 计算出场价格
                if direction == '多':
                    exit_price = price * (1 + expected_profit / 100)
                else:
                    exit_price = price * (1 - expected_profit / 100)
                
                # 确定出场原因
                if expected_profit > 0:
                    exit_reason = 'take_profit'
                else:
                    exit_reason = 'stop_loss'
                
                # 关闭交易
                tracker.close_trade(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    exit_time=future_data['timestamp']
                )
    
    print(f"共执行{trades_executed}笔交易")
    return trades_executed

def run_demo(num_days=100, enable_rl=True):
    """运行探索机制演示"""
    print("=" * 50)
    print("增强型探索机制演示")
    print("=" * 50)
    
    # 生成市场数据
    market_data = generate_market_data(num_days=num_days)
    
    # 创建基本配置
    base_config = {
        'storage_dir': 'data/exploration_demo',
        'exploration_enabled': True,
        'exploration_rate': 0.3,
        'exploration_decay': 0.98,
        'exploration_min_rate': 0.05,
        'exploration_reward_threshold': 1.0,
        'exploration_boost_interval': 50,
        'exploration_success_threshold': 0.6,
        'exploration_failure_threshold': 0.3
    }
    
    # 如果启用强化学习，添加RL配置
    if enable_rl:
        rl_config = {
            'use_rl_for_exploration': True,
            'rl_learning_rate': 0.001,
            'gamma': 0.99,
            'batch_size': 32,
            'target_update_freq': 100,
            'replay_buffer_size': 10000
        }
        base_config.update(rl_config)
    
    # 创建交易跟踪器
    tracker = TradeResultTracker(base_config)
    
    # 模拟交易
    trades_count = simulate_trading(tracker, market_data, "enhanced_exploration")
    
    # 分析结果
    stats = tracker.get_trade_stats()
    exploration_metrics = tracker.get_exploration_metrics()
    exploration_analysis = tracker.analyze_exploration_effectiveness()
    
    # 输出结果
    print("\n===== 交易统计 =====")
    print(f"总交易数: {stats['total_trades']}")
    print(f"完成交易数: {stats['completed_trades']}")
    print(f"胜率: {stats['win_rate']:.2f}")
    print(f"平均利润: {stats['avg_profit']:.2f}%")
    print(f"利润因子: {stats['profit_factor']:.2f}")
    print(f"最大回撤: {stats['max_drawdown']:.2f}%")
    
    print("\n===== 探索指标 =====")
    print(f"探索交易数: {exploration_metrics.get('total_explorations', 0)}")
    print(f"探索成功率: {exploration_metrics.get('success_rate', 0):.2f}")
    print(f"当前探索率: {exploration_metrics.get('exploration_rate', 0):.4f}")
    
    # 可视化探索效果
    output_path = "data/exploration_demo/exploration_effectiveness.png"
    tracker.visualize_exploration_effectiveness(
        output_path=output_path,
        title="探索策略效果分析",
        show_plot=False
    )
    
    print(f"\n探索效果分析图表已保存至: {output_path}")
    
    # 尝试优化探索参数
    print("\n===== 优化探索参数 =====")
    print("正在使用optuna优化探索参数...")
    
    try:
        optimization_results = tracker.optimize_exploration_parameters(
            num_trials=20,
            optimization_metric='combined_score',
            test_trades_count=50,
            save_best_params=True
        )
        
        if optimization_results.get('success', False):
            print("\n优化成功！最佳参数:")
            for param, value in optimization_results['best_params'].items():
                print(f"  {param}: {value:.4f}")
            
            print(f"最佳得分: {optimization_results['best_value']:.4f}")
            
            # 使用优化后的参数再次模拟
            print("\n使用优化后的参数进行第二轮模拟...")
            
            # 生成新的市场数据
            new_market_data = generate_market_data(num_days=num_days)
            
            # 模拟交易
            simulate_trading(tracker, new_market_data, "optimized_exploration")
            
            # 获取新的统计数据
            new_stats = tracker.get_trade_stats()
            
            print("\n===== 优化后的交易统计 =====")
            print(f"总交易数: {new_stats['total_trades']}")
            print(f"完成交易数: {new_stats['completed_trades']}")
            print(f"胜率: {new_stats['win_rate']:.2f}")
            print(f"平均利润: {new_stats['avg_profit']:.2f}%")
            print(f"利润因子: {new_stats['profit_factor']:.2f}")
            
            # 对比改进
            print("\n===== 优化前后对比 =====")
            print(f"胜率: {stats['win_rate']:.2f} -> {new_stats['win_rate']:.2f} ({(new_stats['win_rate']/stats['win_rate']-1)*100:.1f}%)")
            print(f"平均利润: {stats['avg_profit']:.2f}% -> {new_stats['avg_profit']:.2f}% ({(new_stats['avg_profit']/stats['avg_profit']-1)*100:.1f}%)")
            print(f"利润因子: {stats['profit_factor']:.2f} -> {new_stats['profit_factor']:.2f} ({(new_stats['profit_factor']/stats['profit_factor']-1)*100:.1f}%)")
        else:
            print(f"优化失败: {optimization_results.get('message', '未知错误')}")
    except Exception as e:
        print(f"优化过程出错: {e}")
        print("可能需要安装optuna库: pip install optuna")
    
    print("\n演示完成！")
    return tracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='增强型探索机制演示')
    parser.add_argument('--days', type=int, default=100, help='模拟的市场数据天数')
    parser.add_argument('--no-rl', action='store_true', help='禁用强化学习探索决策')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs('data/exploration_demo', exist_ok=True)
    
    # 运行演示
    tracker = run_demo(num_days=args.days, enable_rl=not args.no_rl) 