#!/usr/bin/env python
"""
Optuna超参数优化示例
展示如何使用TradeResultTracker的参数优化功能寻找最佳的探索策略参数
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
import json
from pathlib import Path
import logging

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.self_supervised.utils.trade_tracker import TradeResultTracker

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('optuna_optimization')

def generate_trade_data(num_trades=500, random_seed=42):
    """生成用于优化的交易数据集"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    logger.info(f"生成{num_trades}笔交易数据用于优化...")
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT']
    timeframes = ['5m', '15m', '1h', '4h', 'daily']
    
    # 创建具有多种模式的交易数据，以便测试各种探索参数
    start_date = datetime.now() - timedelta(days=365)
    current_date = start_date
    
    # 生成基本价格序列用于计算SMA
    prices = [1000.0]
    for i in range(10000):  # 足够多的数据点来计算SMA
        price_change = np.random.normal(0, 0.02)
        prices.append(prices[-1] * (1 + price_change))
    
    trades = []
    
    # 创建3个不同的市场阶段
    market_conditions = [
        {'name': '牛市', 'base_win_rate': 0.65, 'duration': num_trades // 3},
        {'name': '震荡市', 'base_win_rate': 0.5, 'duration': num_trades // 3},
        {'name': '熊市', 'base_win_rate': 0.35, 'duration': num_trades - (num_trades // 3) * 2}
    ]
    
    trade_count = 0
    for condition in market_conditions:
        logger.info(f"生成{condition['name']}阶段的交易数据，基础胜率: {condition['base_win_rate']}")
        
        for i in range(condition['duration']):
            # 设置符合当前市场环境的交易参数
            base_win_rate = condition['base_win_rate']
            
            # 周期性地调整胜率，模拟市场波动
            cycle_position = (i % 30) / 30.0
            win_rate_adjustment = 0.15 * np.sin(cycle_position * 2 * np.pi)
            current_win_rate = base_win_rate + win_rate_adjustment
            
            # 随机选择交易品种和方向
            symbol = random.choice(symbols)
            direction = random.choice(['多', '空'])
            
            # 在熊市中，空单更可能盈利；在牛市中，多单更可能盈利
            if (condition['name'] == '牛市' and direction == '多') or \
               (condition['name'] == '熊市' and direction == '空'):
                current_win_rate += 0.1
            
            # 设置价格
            entry_price = random.uniform(100, 10000)
            
            # 设置止损止盈
            if direction == '多':
                stop_loss = entry_price * (1 - random.uniform(0.01, 0.05))
                take_profit = entry_price * (1 + random.uniform(0.02, 0.1))
            else:
                stop_loss = entry_price * (1 + random.uniform(0.01, 0.05))
                take_profit = entry_price * (1 - random.uniform(0.02, 0.1))
            
            # 设置时间
            time_delta = random.randint(30, 720)  # 30分钟到12小时的间隔
            current_date += timedelta(minutes=time_delta)
            entry_time = current_date
            
            # 随机持续时间
            duration_minutes = random.randint(10, 2880)  # 10分钟到2天
            exit_time = entry_time + timedelta(minutes=duration_minutes)
            
            # 确定交易结果
            is_win = random.random() < current_win_rate
            
            # 设置价格和盈亏
            if is_win:
                if direction == '多':
                    exit_price = take_profit * random.uniform(0.95, 1.0)
                else:
                    exit_price = take_profit * random.uniform(1.0, 1.05)
                profit_percentage = abs((exit_price - entry_price) / entry_price * 100)
                if direction == '空':
                    profit_percentage = -profit_percentage
                exit_reason = 'take_profit'
            else:
                if direction == '多':
                    exit_price = stop_loss * random.uniform(0.95, 1.0)
                else:
                    exit_price = stop_loss * random.uniform(1.0, 1.05)
                profit_percentage = -abs((exit_price - entry_price) / entry_price * 100)
                if direction == '空':
                    profit_percentage = -profit_percentage
                exit_reason = 'stop_loss'
            
            # 计算SMA值
            price_idx = trade_count % (len(prices) - 200)  # 循环使用价格数据
            sma20 = sum(prices[price_idx:price_idx+20]) / 20
            sma200 = sum(prices[price_idx:price_idx+200]) / 200
            
            # 创建SBS数据，盈利交易更可能有完整序列
            complete_sequence_prob = 0.8 if is_win else 0.2
            
            metadata = {
                'sbs_data': {
                    'liquidation': random.random() < (0.5 if is_win else 0.2),
                    'double_pattern': random.random() < (0.6 if is_win else 0.3),
                    'sce': random.random() < (0.7 if is_win else 0.3),
                    'complete_sequence': random.random() < complete_sequence_prob,
                    'confirmation_strength': random.uniform(0.2, 0.9),
                    'market_condition': condition['name'],
                    'point1': entry_price * (1 - random.uniform(0.01, 0.03)),
                    'point2': entry_price * (1 + random.uniform(0.01, 0.05)) if direction == '多' else entry_price * (1 - random.uniform(0.01, 0.05)),
                    'point3': entry_price * (1 - random.uniform(0.005, 0.02)),
                    'point4': entry_price * (1 - random.uniform(0.001, 0.01)),
                    'point5': entry_price * (1 + random.uniform(0.02, 0.08)) if direction == '多' else entry_price * (1 - random.uniform(0.02, 0.08))
                },
                'sma20': sma20,
                'sma200': sma200,
                'price_to_sma20': entry_price / sma20,
                'price_to_sma200': entry_price / sma200,
                'sma20_to_sma200': sma20 / sma200
            }
            
            # 创建交易记录
            trade = {
                'trade_id': f"trade_{trade_count + 1}",
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
                'max_profit_percentage': profit_percentage if is_win else profit_percentage * random.uniform(0.3, 0.7),
                'max_drawdown': 0 if is_win else abs(profit_percentage) * random.uniform(0.5, 1.0),
                'duration': duration_minutes,
                'market_condition': condition['name']
            }
            
            trades.append(trade)
            trade_count += 1
    
    return trades

def save_trade_data(trades, output_file='data/optimization/trade_dataset.json'):
    """保存交易数据到JSON文件"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)
    
    logger.info(f"交易数据已保存至: {output_path}")

def load_trade_data(input_file='data/optimization/trade_dataset.json'):
    """从JSON文件加载交易数据"""
    input_path = Path(input_file)
    
    if not input_path.exists():
        logger.error(f"交易数据文件不存在: {input_path}")
        return None
    
    with open(input_path, 'r', encoding='utf-8') as f:
        trades = json.load(f)
    
    logger.info(f"已从{input_path}加载{len(trades)}笔交易数据")
    return trades

def run_optimization(trades, num_trials=100, output_dir='data/optimization'):
    """运行Optuna优化并可视化结果"""
    logger.info(f"开始Optuna优化，将进行{num_trials}次试验...")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建交易跟踪器并添加交易数据
    logger.info("初始化交易跟踪器并添加交易数据...")
    tracker = TradeResultTracker({
        'enable_exploration': True,
        'visualization_dir': str(output_path)
    })
    
    # 添加交易并记录序列
    for trade in trades:
        tracker.add_trade(trade)
        
        # 记录SBS序列
        if 'sbs_data' in trade.get('metadata', {}):
            sbs_data = trade['metadata']['sbs_data']
            if all(f'point{i}' in sbs_data for i in range(1, 6)):
                points = [sbs_data[f'point{i}'] for i in range(1, 6)]
                
                confirmation_signals = {
                    'liquidation': sbs_data.get('liquidation', False),
                    'double_pattern': sbs_data.get('double_pattern', False),
                    'sce': sbs_data.get('sce', False),
                    'strength': sbs_data.get('confirmation_strength', 0.5)
                }
                
                tracker.record_sbs_sequence(
                    trade_id=trade['trade_id'],
                    symbol=trade['symbol'],
                    timeframe=trade['timeframe'],
                    direction=trade['direction'],
                    points=points,
                    confirmation_signals=confirmation_signals,
                    profit_percentage=trade['profit_percentage']
                )
    
    # 运行优化
    logger.info("开始探索参数优化...")
    optimization_results = tracker.optimize_exploration_parameters(
        num_trials=num_trials,
        optimization_metric='combined_score',
        test_trades_count=100,
        save_best_params=True
    )
    
    # 保存优化结果
    results_file = output_path / 'optimization_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        # 转换NumPy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        formatted_results = {
            'best_params': {k: convert_numpy(v) for k, v in optimization_results['best_params'].items()},
            'best_value': float(optimization_results['best_value']),
            'optimization_history': [
                {k: convert_numpy(v) for k, v in trial.items()}
                for trial in optimization_results['optimization_history']
            ]
        }
        
        json.dump(formatted_results, f, indent=2)
    
    logger.info(f"优化结果已保存到: {results_file}")
    
    # 可视化优化结果
    visualize_optimization_results(optimization_results, output_dir)
    
    # 使用最佳参数的交易跟踪器
    logger.info("使用最佳参数创建交易跟踪器并分析结果...")
    best_params = optimization_results['best_params']
    
    best_tracker = TradeResultTracker({
        'enable_exploration': True,
        'exploration_rate': best_params['exploration_rate'],
        'min_exploration_rate': best_params['min_exploration_rate'],
        'exploration_decay': best_params['exploration_decay'],
        'exploration_reward_threshold': best_params['exploration_reward_threshold'],
        'visualization_dir': str(output_path)
    })
    
    # 添加交易数据到最佳参数跟踪器
    for trade in trades:
        best_tracker.add_trade(trade)
        
        # 记录SBS序列
        if 'sbs_data' in trade.get('metadata', {}):
            sbs_data = trade['metadata']['sbs_data']
            if all(f'point{i}' in sbs_data for i in range(1, 6)):
                points = [sbs_data[f'point{i}'] for i in range(1, 6)]
                
                confirmation_signals = {
                    'liquidation': sbs_data.get('liquidation', False),
                    'double_pattern': sbs_data.get('double_pattern', False),
                    'sce': sbs_data.get('sce', False),
                    'strength': sbs_data.get('confirmation_strength', 0.5)
                }
                
                best_tracker.record_sbs_sequence(
                    trade_id=trade['trade_id'],
                    symbol=trade['symbol'],
                    timeframe=trade['timeframe'],
                    direction=trade['direction'],
                    points=points,
                    confirmation_signals=confirmation_signals,
                    profit_percentage=trade['profit_percentage']
                )
    
    # 分析和可视化结果
    logger.info("生成最佳参数配置的业绩报告...")
    best_tracker.visualize_exploration_effectiveness(
        output_path=str(output_path / 'best_exploration_effectiveness.png'),
        title="最佳探索参数配置的探索效果分析",
        show_plot=False
    )
    
    best_tracker.visualize_performance(
        output_path=str(output_path / 'best_performance_overview.png'),
        title="最佳参数配置的交易业绩概览",
        show_plot=False
    )
    
    best_tracker.generate_html_report(
        output_path=str(output_path / 'best_params_report.html'),
        title="最佳探索参数配置的交易业绩报告"
    )
    
    return optimization_results, best_tracker

def visualize_optimization_results(results, output_dir):
    """可视化优化结果"""
    logger.info("可视化优化结果...")
    output_path = Path(output_dir)
    
    # 提取参数和分数历史
    history = results['optimization_history']
    params = ['exploration_rate', 'min_exploration_rate', 'exploration_decay', 
              'exploration_reward_threshold']
    
    # 转换历史记录为DataFrame
    df = pd.DataFrame([
        {**{param: trial['params'][param] for param in params}, 
         'value': trial['value'], 
         'trial': i} 
        for i, trial in enumerate(history)
    ])
    
    # 设置可视化样式
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 10})
    
    # 1. 参数重要性分析
    plt.figure(figsize=(10, 6))
    param_importances = {}
    
    for param in params:
        # 计算相关系数
        correlation = df[param].corr(df['value'])
        param_importances[param] = abs(correlation)
    
    # 绘制参数重要性
    importances_df = pd.DataFrame({
        'Parameter': list(param_importances.keys()),
        'Importance': list(param_importances.values())
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(x='Parameter', y='Importance', data=importances_df, palette='viridis')
    plt.title('参数重要性分析（基于与优化目标的相关性）')
    plt.xlabel('参数')
    plt.ylabel('重要性（相关系数绝对值）')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(str(output_path / 'parameter_importance.png'), dpi=100)
    plt.close()
    
    # 2. 参数收敛图
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(df['trial'], df['exploration_rate'], c=df['value'], cmap='viridis')
    plt.colorbar(label='目标值')
    plt.title('exploration_rate 收敛')
    plt.xlabel('试验次数')
    plt.ylabel('exploration_rate')
    
    plt.subplot(2, 2, 2)
    plt.scatter(df['trial'], df['min_exploration_rate'], c=df['value'], cmap='viridis')
    plt.colorbar(label='目标值')
    plt.title('min_exploration_rate 收敛')
    plt.xlabel('试验次数')
    plt.ylabel('min_exploration_rate')
    
    plt.subplot(2, 2, 3)
    plt.scatter(df['trial'], df['exploration_decay'], c=df['value'], cmap='viridis')
    plt.colorbar(label='目标值')
    plt.title('exploration_decay 收敛')
    plt.xlabel('试验次数')
    plt.ylabel('exploration_decay')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['trial'], df['exploration_reward_threshold'], c=df['value'], cmap='viridis')
    plt.colorbar(label='目标值')
    plt.title('exploration_reward_threshold 收敛')
    plt.xlabel('试验次数')
    plt.ylabel('exploration_reward_threshold')
    
    plt.tight_layout()
    plt.savefig(str(output_path / 'parameter_convergence.png'), dpi=100)
    plt.close()
    
    # 3. 优化历史曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['trial'], df['value'], 'o-', alpha=0.7)
    plt.plot(df['trial'], df['value'].cummax(), 'r-', label='累计最大值')
    plt.title('优化分数历史')
    plt.xlabel('试验次数')
    plt.ylabel('优化目标值')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path / 'optimization_history.png'), dpi=100)
    plt.close()
    
    # 4. 参数对比热力图
    plt.figure(figsize=(12, 10))
    # exploration_rate vs min_exploration_rate
    plt.subplot(2, 2, 1)
    pivot1 = df.pivot_table(
        values='value', 
        index=pd.cut(df['exploration_rate'], 5), 
        columns=pd.cut(df['min_exploration_rate'], 5),
        aggfunc='mean'
    )
    sns.heatmap(pivot1, cmap='viridis', annot=True, fmt='.3f')
    plt.title('exploration_rate vs min_exploration_rate')
    
    # exploration_rate vs exploration_decay
    plt.subplot(2, 2, 2)
    pivot2 = df.pivot_table(
        values='value', 
        index=pd.cut(df['exploration_rate'], 5), 
        columns=pd.cut(df['exploration_decay'], 5),
        aggfunc='mean'
    )
    sns.heatmap(pivot2, cmap='viridis', annot=True, fmt='.3f')
    plt.title('exploration_rate vs exploration_decay')
    
    # exploration_decay vs exploration_reward_threshold
    plt.subplot(2, 2, 3)
    pivot3 = df.pivot_table(
        values='value', 
        index=pd.cut(df['exploration_decay'], 5), 
        columns=pd.cut(df['exploration_reward_threshold'], 5),
        aggfunc='mean'
    )
    sns.heatmap(pivot3, cmap='viridis', annot=True, fmt='.3f')
    plt.title('exploration_decay vs exploration_reward_threshold')
    
    # min_exploration_rate vs exploration_reward_threshold
    plt.subplot(2, 2, 4)
    pivot4 = df.pivot_table(
        values='value', 
        index=pd.cut(df['min_exploration_rate'], 5), 
        columns=pd.cut(df['exploration_reward_threshold'], 5),
        aggfunc='mean'
    )
    sns.heatmap(pivot4, cmap='viridis', annot=True, fmt='.3f')
    plt.title('min_exploration_rate vs exploration_reward_threshold')
    
    plt.tight_layout()
    plt.savefig(str(output_path / 'parameter_interactions.png'), dpi=100)
    plt.close()
    
    logger.info(f"优化可视化结果已保存到: {output_path}")

def run_demos():
    """运行所有演示"""
    parser = argparse.ArgumentParser(description='TradeResultTracker探索参数优化演示')
    parser.add_argument('--num_trades', type=int, default=500, help='生成的交易数量')
    parser.add_argument('--num_trials', type=int, default=50, help='Optuna优化的试验次数')
    parser.add_argument('--random_seed', type=int, default=42, help='随机数种子')
    parser.add_argument('--load_data', action='store_true', help='从文件加载现有交易数据')
    parser.add_argument('--output_dir', type=str, default='data/optimization', help='输出目录')
    args = parser.parse_args()
    
    print("=" * 50)
    print("TradeResultTracker探索参数优化演示")
    print("=" * 50)
    
    # 获取交易数据
    trades = None
    if args.load_data:
        trades = load_trade_data(f"{args.output_dir}/trade_dataset.json")
    
    if trades is None:
        trades = generate_trade_data(args.num_trades, args.random_seed)
        save_trade_data(trades, f"{args.output_dir}/trade_dataset.json")
    
    # 运行优化
    results, best_tracker = run_optimization(
        trades, 
        num_trials=args.num_trials,
        output_dir=args.output_dir
    )
    
    # 输出最佳参数
    print("\n" + "=" * 30)
    print("最佳探索参数配置:")
    print("=" * 30)
    for param, value in results['best_params'].items():
        print(f"{param}: {value}")
    print("=" * 30)
    print(f"最佳分数: {results['best_value']:.4f}")
    print("=" * 30)
    
    print(f"\n优化演示完成！所有结果已保存到 {args.output_dir} 目录")

if __name__ == "__main__":
    run_demos() 