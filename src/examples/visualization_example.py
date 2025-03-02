#!/usr/bin/env python
"""
交易结果和探索效果可视化示例
展示如何使用TradeResultTracker的可视化功能分析交易结果和探索效果
"""

import sys
import os
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.self_supervised.utils.trade_tracker import TradeResultTracker

def generate_sample_trades(num_trades=200):
    """生成样本交易数据用于可视化"""
    print(f"生成{num_trades}笔样本交易数据...")
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT']
    timeframes = ['5m', '15m', '1h', '4h', 'daily']
    
    # 设置一个基准时间
    start_date = datetime.now() - timedelta(days=180)
    
    trades = []
    # 创建一个时间序列，让交易有时间顺序性
    current_date = start_date
    
    # 生成一些价格序列数据以计算SMA
    prices = [1000.0]
    for i in range(10000):  # 足够多的数据点来计算SMA
        price_change = np.random.normal(0, 0.02)
        prices.append(prices[-1] * (1 + price_change))
    
    for i in range(num_trades):
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
            
        # 随机但有序列性的时间
        # 模拟每笔交易之间的时间间隔
        time_delta = random.randint(30, 1440)  # 30分钟到1天之间的间隔
        current_date += timedelta(minutes=time_delta)
        entry_time = current_date
        
        # 模拟交易持续时间
        duration_minutes = random.randint(5, 1440)  # 5分钟到1天
        exit_time = entry_time + timedelta(minutes=duration_minutes)
        
        # 模拟交易结果，但使用一个更有模式的盈亏分布
        # 周期性地改变胜率，模拟市场变化
        cycle_position = (i % 50) / 50.0  # 0到1之间的循环位置
        win_rate = 0.3 + 0.4 * np.sin(cycle_position * 2 * np.pi)  # 胜率在30%到70%之间波动
        
        is_win = random.random() < win_rate
        
        # 模拟探索状态
        # 每20笔交易进入一次探索阶段
        is_exploration = ((i // 20) % 2) == 1
        
        # 探索阶段的胜率略低
        if is_exploration and random.random() < 0.3:
            is_win = not is_win  # 30%概率翻转结果，使探索期胜率偏低
            
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
                
        # 计算模拟的SMA值
        price_idx = i % (len(prices) - 200)  # 循环使用价格数据
        sma20 = sum(prices[price_idx:price_idx+20]) / 20
        sma200 = sum(prices[price_idx:price_idx+200]) / 200
        
        # 添加SBS数据，使其与交易结果有相关性
        has_sbs = True
        
        metadata = {
            'sma20': sma20,
            'sma200': sma200,
            'price_to_sma20': entry_price / sma20,
            'price_to_sma200': entry_price / sma200,
            'sma20_to_sma200': sma20 / sma200
        }
        
        # 创建与交易结果相关的SBS数据
        if has_sbs:
            # 成功交易中，完整序列和确认信号更频繁
            complete_sequence_prob = 0.7 if is_win else 0.3
            confirmation_prob = 0.8 if is_win else 0.4
            
            sbs_data = {
                'liquidation': random.random() < (0.4 if is_win else 0.2),
                'double_pattern': random.random() < (0.6 if is_win else 0.2),
                'sce': random.random() < (0.7 if is_win else 0.3),
                'complete_sequence': random.random() < complete_sequence_prob,
                'point1': random.uniform(entry_price * 0.95, entry_price * 1.05),
                'point2': random.uniform(entry_price * 0.9, entry_price * 1.1),
                'point3': random.uniform(entry_price * 0.93, entry_price * 1.07),
                'point4': random.uniform(entry_price * 0.92, entry_price * 1.08),
                'point5': random.uniform(entry_price * 0.9, entry_price * 1.1),
                'confirmation_strength': random.uniform(0.1, 0.9)
            }
            
            # 如果是探索交易，添加探索标签
            if is_exploration:
                sbs_data['exploration'] = True
                sbs_data['exploration_type'] = random.choice(['new_pattern', 'timeframe_test', 'parameter_adjust'])
                sbs_data['exploration_result'] = 'success' if is_win else 'failure'
            
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
            'exploration': is_exploration,
            'max_profit_percentage': profit_percentage if is_win else profit_percentage * random.uniform(0.2, 0.8),
            'max_drawdown': abs(profit_percentage) * random.uniform(0.1, 0.5) if not is_win else random.uniform(0, 1.0),
            'duration': duration_minutes
        }
        
        trades.append(trade)
    
    return trades

def run_visualization_demos(trades):
    """运行可视化演示"""
    print("\n开始可视化演示...")
    
    # 创建交易跟踪器
    config = {
        'enable_exploration': True,
        'exploration_rate': 0.2,
        'min_exploration_rate': 0.05,
        'exploration_decay': 0.95,
        'exploration_reward_threshold': 1.0,
        'visualization_dir': 'data/visualization'
    }
    
    tracker = TradeResultTracker(config)
    
    # 添加交易记录
    for trade in trades:
        tracker.add_trade(trade)
        
        # 如果交易有SBS数据，记录SBS序列
        if 'sbs_data' in trade.get('metadata', {}):
            sbs_data = trade['metadata']['sbs_data']
            
            # 如果有完整的点位数据，记录SBS序列
            if all(f'point{i}' in sbs_data for i in range(1, 6)):
                points = [sbs_data[f'point{i}'] for i in range(1, 6)]
                
                confirmation_signals = {}
                if 'liquidation' in sbs_data:
                    confirmation_signals['liquidation'] = sbs_data['liquidation']
                if 'double_pattern' in sbs_data:
                    confirmation_signals['double_pattern'] = sbs_data['double_pattern']
                if 'sce' in sbs_data:
                    confirmation_signals['sce'] = sbs_data['sce']
                if 'confirmation_strength' in sbs_data:
                    confirmation_signals['strength'] = sbs_data['confirmation_strength']
                
                tracker.record_sbs_sequence(
                    trade_id=trade['trade_id'],
                    symbol=trade['symbol'],
                    timeframe=trade['timeframe'],
                    direction=trade['direction'],
                    points=points,
                    confirmation_signals=confirmation_signals,
                    profit_percentage=trade['profit_percentage'],
                    is_exploration=trade.get('exploration', False)
                )
    
    # 生成各种可视化
    output_dir = 'data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. 生成交易业绩可视化...")
    tracker.visualize_performance(
        output_path=f"{output_dir}/performance_overview.png",
        title="交易业绩概览",
        show_plot=False
    )
    
    print("2. 生成盈亏走势图...")
    tracker.visualize_equity_curve(
        output_path=f"{output_dir}/equity_curve.png",
        title="账户权益曲线",
        show_plot=False
    )
    
    print("3. 生成交易分布图...")
    tracker.visualize_trade_distribution(
        output_path=f"{output_dir}/trade_distribution.png",
        title="交易盈亏分布",
        show_plot=False
    )
    
    print("4. 生成交易月度热力图...")
    tracker.visualize_monthly_heatmap(
        output_path=f"{output_dir}/monthly_heatmap.png",
        title="月度交易业绩热力图",
        show_plot=False
    )
    
    print("5. 生成探索效果分析图...")
    tracker.visualize_exploration_effectiveness(
        output_path=f"{output_dir}/exploration_effectiveness.png",
        title="探索策略效果分析",
        show_plot=False
    )
    
    print("6. 生成SBS序列模式分析图...")
    tracker.visualize_sbs_patterns(
        output_path=f"{output_dir}/sbs_patterns.png",
        title="SBS交易模式分析",
        show_plot=False
    )
    
    print("7. 生成交易质量分析图...")
    tracker.visualize_trade_quality(
        output_path=f"{output_dir}/trade_quality.png",
        title="交易质量分析",
        show_plot=False
    )
    
    print("8. 生成确认信号分析图...")
    tracker.visualize_confirmation_signals(
        output_path=f"{output_dir}/confirmation_signals.png",
        title="确认信号效果分析",
        show_plot=False
    )
    
    print("9. 生成交易报告...")
    report_path = f"{output_dir}/trading_report.html"
    tracker.generate_html_report(
        output_path=report_path,
        title="交易业绩详细报告"
    )
    
    print(f"\n可视化演示完成！所有图表已保存到 {output_dir} 目录")
    print(f"详细HTML报告保存在: {report_path}")
    
    return tracker

def show_sample_plots(trades):
    """展示一些自定义示例图表"""
    print("\n创建自定义示例图表...")
    
    # 将交易列表转换为DataFrame
    df = pd.DataFrame([
        {
            'trade_id': t['trade_id'],
            'symbol': t['symbol'],
            'profit': t['profit_percentage'],
            'direction': t['direction'],
            'entry_time': datetime.fromisoformat(t['entry_time'].replace('Z', '+00:00') if t['entry_time'].endswith('Z') else t['entry_time']),
            'timeframe': t['timeframe'],
            'duration': t['duration'],
            'exploration': t.get('exploration', False),
            'has_sbs': 'sbs_data' in t.get('metadata', {})
        } for t in trades
    ])
    
    # 设置可视化样式
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 10})
    
    # 创建输出目录
    output_dir = 'data/visualization/custom'
    os.makedirs(output_dir, exist_ok=True)
    
    # 示例1: 不同品种的盈亏对比
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='symbol', y='profit', data=df, palette='viridis')
    plt.title('不同交易品种的盈亏分布')
    plt.xlabel('交易品种')
    plt.ylabel('盈亏百分比')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/symbol_profit_boxplot.png", dpi=100)
    plt.close()
    
    # 示例2: 不同时间框架的胜率
    plt.figure(figsize=(10, 6))
    win_rates = []
    for tf in df['timeframe'].unique():
        tf_data = df[df['timeframe'] == tf]
        win_rate = len(tf_data[tf_data['profit'] > 0]) / len(tf_data) * 100 if len(tf_data) > 0 else 0
        win_rates.append({'Timeframe': tf, 'Win Rate': win_rate})
    
    win_rate_df = pd.DataFrame(win_rates)
    sns.barplot(x='Timeframe', y='Win Rate', data=win_rate_df, palette='coolwarm')
    plt.title('不同时间框架的胜率对比')
    plt.xlabel('时间框架')
    plt.ylabel('胜率 (%)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/timeframe_winrate.png", dpi=100)
    plt.close()
    
    # 示例3: 交易持续时间与盈亏关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='duration', y='profit', hue='direction', data=df, palette={'多': 'green', '空': 'red'}, alpha=0.7)
    plt.title('交易持续时间与盈亏关系')
    plt.xlabel('持续时间 (分钟)')
    plt.ylabel('盈亏百分比')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.legend(title='方向')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/duration_profit.png", dpi=100)
    plt.close()
    
    # 示例4: 探索交易与常规交易的对比
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='exploration', y='profit', data=df, palette=['blue', 'orange'], inner='quartile')
    plt.title('探索交易与常规交易盈亏对比')
    plt.xlabel('是否为探索交易')
    plt.ylabel('盈亏百分比')
    plt.xticks([0, 1], ['常规交易', '探索交易'])
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exploration_comparison.png", dpi=100)
    plt.close()
    
    # 示例5: 时间序列上的探索策略效果
    df['date'] = df['entry_time'].dt.date
    daily_data = df.groupby(['date', 'exploration']).agg(
        avg_profit=('profit', 'mean'),
        count=('trade_id', 'count')
    ).reset_index()
    
    pivot_data = daily_data.pivot(index='date', columns='exploration', values='avg_profit').fillna(0)
    pivot_data.columns = ['常规交易', '探索交易']
    
    plt.figure(figsize=(12, 6))
    pivot_data.plot(kind='line', marker='o', ax=plt.gca())
    plt.title('时间序列上的探索策略与常规策略盈亏对比')
    plt.xlabel('日期')
    plt.ylabel('平均盈亏百分比')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exploration_time_series.png", dpi=100)
    plt.close()
    
    print(f"自定义示例图表已保存到 {output_dir} 目录")

def run_demos():
    """运行所有演示"""
    parser = argparse.ArgumentParser(description='TradeResultTracker可视化功能演示')
    parser.add_argument('--num_trades', type=int, default=200, help='生成的交易数量')
    args = parser.parse_args()
    
    print("=" * 50)
    print("TradeResultTracker可视化功能演示")
    print("=" * 50)
    
    # 生成测试数据
    trades = generate_sample_trades(args.num_trades)
    
    # 运行可视化演示
    tracker = run_visualization_demos(trades)
    
    # 创建自定义示例图表
    show_sample_plots(trades)
    
    print("\n演示完成！")

if __name__ == "__main__":
    run_demos() 