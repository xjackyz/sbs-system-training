"""
可视化工具模块
用于绘制学习曲线和监控训练进度
"""

import os
from typing import Dict, List, Optional
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
from ..utils.logger import setup_logger

logger = setup_logger('visualization')

class LearningVisualizer:
    """学习曲线可视化工具"""
    
    def __init__(self, save_dir: str = 'logs/visualization'):
        """
        初始化可视化工具
        
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_learning_curves(self, stats_dir: str = 'logs/rewards') -> str:
        """
        绘制学习曲线
        
        Args:
            stats_dir: 统计数据目录
            
        Returns:
            str: 保存的图表路径
        """
        try:
            # 加载统计数据
            stats_data = self._load_stats_data(stats_dir)
            if not stats_data:
                logger.warning("没有找到统计数据")
                return ""
                
            # 创建图表
            fig = plt.figure(figsize=(15, 10))
            
            # 1. 准确度曲线
            ax1 = plt.subplot(2, 1, 1)
            self._plot_accuracy_curves(ax1, stats_data)
            
            # 2. 累积利润曲线
            ax2 = plt.subplot(2, 1, 2)
            self._plot_profit_curves(ax2, stats_data)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.save_dir / f'learning_curves_{timestamp}.png'
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"学习曲线已保存至: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"绘制学习曲线失败: {str(e)}")
            return ""
            
    def _load_stats_data(self, stats_dir: str) -> List[Dict]:
        """
        加载统计数据
        
        Args:
            stats_dir: 统计数据目录
            
        Returns:
            List[Dict]: 统计数据列表
        """
        stats_data = []
        stats_dir = Path(stats_dir)
        
        # 按日期排序加载所有统计文件
        for stats_file in sorted(stats_dir.glob("stats_*.json")):
            try:
                with open(stats_file, 'r') as f:
                    daily_stats = json.load(f)
                stats_data.append(daily_stats)
            except Exception as e:
                logger.error(f"加载统计文件 {stats_file} 失败: {str(e)}")
                continue
                
        return stats_data
        
    def _plot_accuracy_curves(self, ax: plt.Axes, stats_data: List[Dict]) -> None:
        """
        绘制准确度相关曲线
        
        Args:
            ax: matplotlib轴对象
            stats_data: 统计数据列表
        """
        dates = []
        accuracy_rates = []
        false_positive_rates = []
        false_negative_rates = []
        
        for daily_stats in stats_data:
            date = daily_stats['date']
            total_trades = daily_stats['total_trades']
            
            if total_trades > 0:
                # 计算各类率
                false_positives = daily_stats['false_positives']
                false_negatives = daily_stats['false_negatives']
                correct_trades = total_trades - false_positives - false_negatives
                
                accuracy_rate = correct_trades / total_trades
                false_positive_rate = false_positives / total_trades
                false_negative_rate = false_negatives / total_trades
                
                dates.append(date)
                accuracy_rates.append(accuracy_rate)
                false_positive_rates.append(false_positive_rate)
                false_negative_rates.append(false_negative_rate)
        
        # 绘制曲线
        ax.plot(dates, accuracy_rates, 'g-', label='准确率', marker='o')
        ax.plot(dates, false_positive_rates, 'r--', label='误报率', marker='s')
        ax.plot(dates, false_negative_rates, 'b--', label='漏报率', marker='^')
        
        # 设置图表
        ax.set_title('识别准确度趋势', fontsize=12)
        ax.set_xlabel('日期')
        ax.set_ylabel('比率')
        ax.grid(True)
        ax.legend()
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45)
        
    def _plot_profit_curves(self, ax: plt.Axes, stats_data: List[Dict]) -> None:
        """
        绘制利润相关曲线
        
        Args:
            ax: matplotlib轴对象
            stats_data: 统计数据列表
        """
        dates = []
        daily_profits = []
        cumulative_profits = []
        win_rates = []
        
        cumulative_pnl = 0
        for daily_stats in stats_data:
            date = daily_stats['date']
            daily_pnl = daily_stats['pnl']
            cumulative_pnl += daily_pnl
            
            total_trades = daily_stats['total_trades']
            profitable_trades = daily_stats['profitable_trades']
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            dates.append(date)
            daily_profits.append(daily_pnl)
            cumulative_profits.append(cumulative_pnl)
            win_rates.append(win_rate)
        
        # 创建双y轴
        ax2 = ax.twinx()
        
        # 绘制曲线
        ax.bar(dates, daily_profits, alpha=0.3, color='gray', label='日利润')
        ax.plot(dates, cumulative_profits, 'b-', label='累积利润', linewidth=2)
        ax2.plot(dates, win_rates, 'r--', label='胜率', linewidth=1)
        
        # 设置图表
        ax.set_title('利润趋势', fontsize=12)
        ax.set_xlabel('日期')
        ax.set_ylabel('利润($)')
        ax2.set_ylabel('胜率')
        
        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 网格
        ax.grid(True)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45)
        
    def plot_reward_distribution(self, stats_dir: str = 'logs/rewards') -> str:
        """
        绘制奖励分布图
        
        Args:
            stats_dir: 统计数据目录
            
        Returns:
            str: 保存的图表路径
        """
        try:
            # 加载统计数据
            stats_data = self._load_stats_data(stats_dir)
            if not stats_data:
                return ""
                
            # 收集所有奖励数据
            profit_rewards = []
            accuracy_rewards = []
            total_rewards = []
            
            for daily_stats in stats_data:
                for reward_info in daily_stats['rewards']:
                    profit_rewards.append(reward_info['profit_reward'])
                    accuracy_rewards.append(reward_info['accuracy_reward'])
                    total_rewards.append(reward_info['total_reward'])
                    
            # 创建图表
            fig = plt.figure(figsize=(15, 5))
            
            # 1. 利润奖励分布
            ax1 = plt.subplot(1, 3, 1)
            sns.histplot(profit_rewards, ax=ax1, bins=30)
            ax1.set_title('利润奖励分布')
            
            # 2. 准确度奖励分布
            ax2 = plt.subplot(1, 3, 2)
            sns.histplot(accuracy_rewards, ax=ax2, bins=30)
            ax2.set_title('准确度奖励分布')
            
            # 3. 总奖励分布
            ax3 = plt.subplot(1, 3, 3)
            sns.histplot(total_rewards, ax=ax3, bins=30)
            ax3.set_title('总奖励分布')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.save_dir / f'reward_distribution_{timestamp}.png'
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"奖励分布图已保存至: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"绘制奖励分布图失败: {str(e)}")
            return ""
            
    def plot_training_progress(self, 
                             stats_dir: str = 'logs/rewards',
                             window_size: int = 7) -> str:
        """
        绘制训练进度图表
        
        Args:
            stats_dir: 统计数据目录
            window_size: 移动平均窗口大小
            
        Returns:
            str: 保存的图表路径
        """
        try:
            # 加载统计数据
            stats_data = self._load_stats_data(stats_dir)
            if not stats_data:
                return ""
                
            # 创建图表
            fig = plt.figure(figsize=(15, 10))
            
            # 1. 移动平均曲线
            ax1 = plt.subplot(2, 1, 1)
            self._plot_moving_averages(ax1, stats_data, window_size)
            
            # 2. 性能指标热图
            ax2 = plt.subplot(2, 1, 2)
            self._plot_performance_heatmap(ax2, stats_data)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.save_dir / f'training_progress_{timestamp}.png'
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"训练进度图表已保存至: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"绘制训练进度图表失败: {str(e)}")
            return ""
            
    def _plot_moving_averages(self, 
                             ax: plt.Axes, 
                             stats_data: List[Dict],
                             window_size: int) -> None:
        """
        绘制移动平均曲线
        
        Args:
            ax: matplotlib轴对象
            stats_data: 统计数据列表
            window_size: 移动平均窗口大小
        """
        dates = []
        accuracy_ma = []
        profit_ma = []
        
        # 收集数据
        daily_accuracy = []
        daily_profits = []
        
        for daily_stats in stats_data:
            dates.append(daily_stats['date'])
            
            # 计算日准确率
            total = daily_stats['total_trades']
            if total > 0:
                correct = total - daily_stats['false_positives'] - daily_stats['false_negatives']
                accuracy = correct / total
            else:
                accuracy = 0
            daily_accuracy.append(accuracy)
            
            # 收集日利润
            daily_profits.append(daily_stats['pnl'])
        
        # 计算移动平均
        for i in range(len(dates)):
            if i < window_size - 1:
                accuracy_ma.append(np.mean(daily_accuracy[:i+1]))
                profit_ma.append(np.mean(daily_profits[:i+1]))
            else:
                accuracy_ma.append(np.mean(daily_accuracy[i-window_size+1:i+1]))
                profit_ma.append(np.mean(daily_profits[i-window_size+1:i+1]))
        
        # 绘制曲线
        ax.plot(dates, accuracy_ma, 'g-', label=f'{window_size}日准确率均值')
        ax2 = ax.twinx()
        ax2.plot(dates, profit_ma, 'b-', label=f'{window_size}日利润均值')
        
        # 设置图表
        ax.set_title('移动平均趋势', fontsize=12)
        ax.set_xlabel('日期')
        ax.set_ylabel('准确率')
        ax2.set_ylabel('利润($)')
        
        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45)
        
    def _plot_performance_heatmap(self, ax: plt.Axes, stats_data: List[Dict]) -> None:
        """
        绘制性能指标热图
        
        Args:
            ax: matplotlib轴对象
            stats_data: 统计数据列表
        """
        # 准备数据
        performance_data = []
        for daily_stats in stats_data:
            total_trades = daily_stats['total_trades']
            if total_trades > 0:
                correct = total_trades - daily_stats['false_positives'] - daily_stats['false_negatives']
                accuracy = correct / total_trades
                win_rate = daily_stats['profitable_trades'] / total_trades
                avg_profit = daily_stats['pnl'] / total_trades
            else:
                accuracy = win_rate = avg_profit = 0
                
            performance_data.append({
                'date': daily_stats['date'],
                'accuracy': accuracy,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'total_trades': total_trades
            })
            
        # 创建数据框
        df = pd.DataFrame(performance_data)
        df = df.set_index('date')
        
        # 绘制热图
        sns.heatmap(df[['accuracy', 'win_rate', 'avg_profit', 'total_trades']].T,
                   cmap='RdYlGn',
                   center=0,
                   ax=ax)
        
        # 设置图表
        ax.set_title('性能指标热图')
        ax.set_xlabel('日期')
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45) 