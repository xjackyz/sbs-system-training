import os
import time
import json
import logging
import threading
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import aiohttp
import asyncio
from datetime import datetime
import discord
from discord import Webhook, File
import io
import platform
import psutil
import gc
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class ProgressNotifier:
    """训练进度通知器
    
    定期将训练进度发送到Discord频道，包括图表和统计信息
    """
    
    def __init__(self, discord_notifier=None, webhook_url: str = None, config_path: str = None, 
                notify_every: int = 5, charts_dir: str = "charts", interval_minutes: int = 5, save_dir: str = None):
        """初始化进度通知器
        
        Args:
            discord_notifier: Discord通知器实例
            webhook_url: Discord Webhook URL
            config_path: 配置文件路径，用于从文件加载webhook_url
            notify_every: 每多少轮通知一次
            charts_dir: 图表保存目录
            interval_minutes: 通知间隔（分钟）
            save_dir: 保存目录
        """
        self.webhook_url = webhook_url
        self.discord_notifier = discord_notifier
        self.notify_every = notify_every
        self.charts_dir = charts_dir
        self.interval_minutes = interval_minutes
        self.save_dir = save_dir
        self.logger = logging.getLogger('progress_notifier')
        
        # 创建图表目录
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # 加载配置文件
        if config_path and not webhook_url and not discord_notifier:
            self._load_config(config_path)
        
        # 初始化统计信息
        self.stats = {
            'start_time': time.time(),
            'last_notification_time': 0,
            'notification_count': 0,
            'last_metrics': {},
            'training_in_progress': False,
            'epochs_completed': 0,
            'total_epochs': 0,
            'estimated_completion_time': None,
            'epoch_times': [],
            'memory_usage': [],
            'gpu_usage': [],
            'signal_metrics': {
                'bullish': {'correct': 0, 'total': 0},
                'bearish': {'correct': 0, 'total': 0},
                'neutral': {'correct': 0, 'total': 0}
            }
        }
        
        # 通知队列和事件循环
        self.notification_queue = []
        self.loop = None
        self.notification_thread = None
        self.is_running = False
        
        # 状态跟踪
        self.performance_history = []
        self.error_count = 0
        self.warning_count = 0
        
        # 跟踪每轮的指标
        self.epoch_metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epochs': []
        }
        
        # 初始化通知线程
        if self.webhook_url or self.discord_notifier:
            self._init_notification_thread()
    
    def _load_config(self, config_path: str):
        """从配置文件加载Discord配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 提取webhook URL
            if 'discord' in config:
                if 'webhook_url' in config['discord']:
                    self.webhook_url = config['discord']['webhook_url']
                    self.logger.info("从配置文件加载了Discord webhook URL")
                else:
                    self.logger.warning("配置文件中缺少 webhook_url")
            else:
                self.logger.warning("配置文件中缺少 discord 部分")
        
        except Exception as e:
            self.logger.error(f"加载Discord配置时出错: {e}")
    
    def _init_notification_thread(self):
        """初始化通知线程"""
        if self.notification_thread is not None and self.notification_thread.is_alive():
            return
            
        self.is_running = True
        self.notification_thread = threading.Thread(target=self._run_notification_loop, daemon=True)
        self.notification_thread.start()
        self.logger.info("通知线程已启动")
    
    def _run_notification_loop(self):
        """运行通知事件循环"""
        # 创建新的事件循环
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # 运行事件循环
        try:
            while self.is_running:
                # 检查队列
                while self.notification_queue:
                    # 获取下一个通知
                    notification = self.notification_queue.pop(0)
                    
                    # 发送通知
                    self.loop.run_until_complete(self._send_discord_message(**notification))
                
                # 检查是否需要发送定时通知
                current_time = time.time()
                if (current_time - self.stats['last_notification_time']) > (self.interval_minutes * 60) and self.stats.get('training_in_progress', False):
                    # 发送定时通知
                    if hasattr(self, 'on_periodic_notification'):
                        self.on_periodic_notification()
                    
                    # 更新最后通知时间
                    self.stats['last_notification_time'] = current_time
                
                # 休眠一段时间
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"通知循环中出错: {e}")
        finally:
            self.loop.close()
            self.logger.info("通知线程已停止")
    
    def stop(self):
        """停止通知线程"""
        self.is_running = False
        if self.notification_thread and self.notification_thread.is_alive():
            self.notification_thread.join(timeout=5)
    
    async def _send_discord_message(self, message: str, charts: List[str] = None, files: List[str] = None, embed_color: int = None):
        """发送Discord消息
        
        Args:
            message: 消息内容
            charts: 图表文件路径列表
            files: 其他文件路径列表
            embed_color: 嵌入颜色
        """
        if not self.webhook_url:
            self.logger.warning("未设置webhook_url，无法发送通知")
            return
        
        try:
            # 分块发送长消息
            chunks = self._split_message(message)
            
            # 准备附件
            webhook_files = []
            
            # 添加图表
            if charts:
                for i, chart_path in enumerate(charts):
                    if os.path.exists(chart_path):
                        with open(chart_path, 'rb') as f:
                            chart_file = File(f, filename=f"chart_{i+1}.png")
                            webhook_files.append(chart_file)
            
            # 添加其他文件
            if files:
                for i, file_path in enumerate(files):
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            file_obj = File(f, filename=os.path.basename(file_path))
                            webhook_files.append(file_obj)
            
            # 使用aiohttp发送消息
            async with aiohttp.ClientSession() as session:
                webhook = Webhook.from_url(self.webhook_url, session=session)
                
                # 如果有附件，发送第一个消息块和附件
                if webhook_files:
                    first_chunk = chunks.pop(0) if chunks else "附件"
                    await webhook.send(content=first_chunk, files=webhook_files)
                
                # 发送剩余消息块
                for chunk in chunks:
                    if chunk.strip():  # 跳过空块
                        await webhook.send(content=chunk)
            
            # 更新统计信息
            self.stats['notification_count'] += 1
            self.stats['last_notification_time'] = time.time()
            
            self.logger.info(f"已发送Discord通知 (总数: {self.stats['notification_count']})")
            
            # 删除临时图表文件
            if charts:
                for chart_path in charts:
                    try:
                        if os.path.exists(chart_path) and 'temp_charts' in chart_path:
                            os.remove(chart_path)
                    except Exception as e:
                        self.logger.warning(f"删除临时图表文件时出错: {e}")
            
        except Exception as e:
            self.logger.error(f"发送Discord消息时出错: {e}")
            # 增加错误计数
            self.error_count += 1
    
    def _split_message(self, message: str, max_length: int = 1900) -> List[str]:
        """拆分长消息
        
        Discord消息有2000字符限制，我们保留一些余量
        
        Args:
            message: 要拆分的消息
            max_length: 每个块的最大长度
            
        Returns:
            消息块列表
        """
        if len(message) <= max_length:
            return [message]
        
        chunks = []
        current_chunk = ""
        
        # 按行拆分
        for line in message.split('\n'):
            # 如果这一行太长，需要进一步拆分
            if len(line) > max_length:
                # 先添加当前块
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # 拆分长行
                for i in range(0, len(line), max_length):
                    chunks.append(line[i:i+max_length])
            
            # 检查添加这一行是否会超出限制
            elif len(current_chunk) + len(line) + 1 > max_length:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def send_training_progress(self, epoch: int, metrics: Dict, learning_rate: float = None,
                              charts: List[str] = None, checkpoint_path: str = None, profit: float = None):
        """发送训练进度通知
        
        Args:
            epoch: 当前轮次
            metrics: 训练指标
            learning_rate: 当前学习率
            charts: 额外的图表文件路径列表
            checkpoint_path: 检查点路径
            profit: 当前盈利情况
        """
        # 更新统计数据
        self.update_training_stats(epoch, metrics, learning_rate)
        
        # 如果不是通知轮次，直接返回
        if epoch % self.notify_every != 0:
            return
            
        # 准备消息
        message = f"🔄 **训练进度 - 轮次 {epoch}** 🔄\n\n"
        
        # 添加指标信息
        message += "**训练指标:**\n"
        for key, value in metrics.items():
            message += f"- {key}: {value:.4f}\n"
        
        # 添加盈利情况
        if profit is not None:
            message += f"\n**当前盈利情况:** {profit:.2f}"
        
        # 添加学习率信息
        if learning_rate is not None:
            message += f"\n**当前学习率:** {learning_rate:.8f}"
        
        # 添加进度信息
        if self.stats['total_epochs'] > 0:
            progress = (epoch / self.stats['total_epochs']) * 100
            message += f"\n\n**训练进度:** {progress:.1f}% ({epoch}/{self.stats['total_epochs']}轮)"
        
        # 添加估计完成时间
        if self.stats['estimated_completion_time']:
            eta = datetime.fromtimestamp(self.stats['estimated_completion_time'])
            now = datetime.now()
            time_remaining = eta - now
            hours_remaining = time_remaining.total_seconds() / 3600
            
            message += f"\n**预计完成时间:** {eta.strftime('%Y-%m-%d %H:%M:%S')}"
            message += f"\n**剩余时间:** {hours_remaining:.1f}小时"
        
        # 添加内存使用信息
        if self.stats['memory_usage']:
            current_memory = self.stats['memory_usage'][-1][1]
            message += f"\n\n**当前内存使用:** {current_memory:.1f} MB"
        
        # 创建并添加图表
        all_charts = []
        
        # 1. 学习曲线图
        learning_curve_chart = self.create_learning_curve_chart()
        if learning_curve_chart:
            all_charts.append(learning_curve_chart)
        
        # 2. 内存使用图
        if len(self.stats['memory_usage']) > 2:
            memory_chart = self.create_memory_usage_chart()
            if memory_chart:
                all_charts.append(memory_chart)
        
        # 3. 添加信号性能图（如果有足够数据）
        has_signal_data = any(metrics['total'] > 0 for metrics in self.stats['signal_metrics'].values())
        if has_signal_data:
            signal_chart = self.create_signal_performance_chart()
            if signal_chart:
                all_charts.append(signal_chart)
        
        # 4. 添加用户提供的图表
        if charts:
            all_charts.extend(charts)
        
        # 发送通知
        if self.discord_notifier:
            self.discord_notifier.send_message(message, files=all_charts)
        else:
            self.notification_queue.append({
                'message': message,
                'charts': all_charts,
                'files': [],
                'embed_color': 0x3498db  # 蓝色
            })
    
    def set_total_epochs(self, total_epochs: int):
        """设置总训练轮次
        
        Args:
            total_epochs: 总训练轮次
        """
        self.stats['total_epochs'] = total_epochs
    
    def update_training_stats(self, epoch: int, metrics: Dict, learning_rate: float = None, 
                             signal_metrics: Dict = None, memory_usage: float = None):
        """更新训练统计信息
        
        用于跟踪性能演变，不会立即发送通知
        
        Args:
            epoch: 当前训练轮次
            metrics: 性能指标 (包含train_loss, train_accuracy, val_loss, val_accuracy等)
            learning_rate: 当前学习率
            signal_metrics: 信号类型性能指标
            memory_usage: 内存使用量(MB)
        """
        # 更新轮次信息
        self.stats['epochs_completed'] = epoch
        
        # 记录轮次时间
        if self.stats.get('epoch_start_time'):
            epoch_time = time.time() - self.stats['epoch_start_time']
            self.stats['epoch_times'].append(epoch_time)
            
            # 估计完成时间
            if self.stats['total_epochs'] > epoch:
                remaining_epochs = self.stats['total_epochs'] - epoch
                avg_epoch_time = np.mean(self.stats['epoch_times'][-min(5, len(self.stats['epoch_times'])):])
                estimated_remaining_time = avg_epoch_time * remaining_epochs
                self.stats['estimated_completion_time'] = time.time() + estimated_remaining_time
        
        # 记录新轮次开始时间
        self.stats['epoch_start_time'] = time.time()
        
        # 保存性能历史
        self.performance_history.append({
            'epoch': epoch,
            'metrics': metrics,
            'learning_rate': learning_rate,
            'timestamp': time.time()
        })
        
        # 更新学习曲线数据
        if metrics:
            self.epoch_metrics['epochs'].append(epoch)
            
            for key in ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']:
                if key in metrics:
                    self.epoch_metrics[key].append(metrics[key])
            
            # 记录学习率
            if learning_rate is not None:
                self.epoch_metrics['learning_rate'].append(learning_rate)
        
        # 更新信号指标
        if signal_metrics:
            for signal_type, metrics in signal_metrics.items():
                if signal_type in self.stats['signal_metrics']:
                    if 'correct' in metrics:
                        self.stats['signal_metrics'][signal_type]['correct'] += metrics['correct']
                    if 'total' in metrics:
                        self.stats['signal_metrics'][signal_type]['total'] += metrics['total']
        
        # 记录内存使用
        if memory_usage is not None:
            self.stats['memory_usage'].append((time.time(), memory_usage))
        else:
            # 自动获取内存使用
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # 转换为MB
            self.stats['memory_usage'].append((time.time(), memory_mb))
        
        # 检查性能下降
        if len(self.performance_history) >= 5:  # 至少有5个数据点才检查
            self._check_performance_degradation()
    
    def _check_performance_degradation(self, key_metric: str = 'val_loss', window: int = 5, threshold: float = 0.2):
        """检查性能下降
        
        Args:
            key_metric: 要监控的关键指标
            window: 滑动窗口大小
            threshold: 下降阈值
        """
        # 获取最近N个数据点
        recent_history = self.performance_history[-window:]
        
        # 提取指标值
        metric_values = [point['metrics'].get(key_metric) for point in recent_history if point['metrics'].get(key_metric) is not None]
        
        # 如果数据不足，返回
        if len(metric_values) < window:
            return
        
        # 对于损失值，较低更好；对于准确率等，较高更好
        is_loss_metric = 'loss' in key_metric.lower()
        
        if is_loss_metric:
            # 计算趋势（对于损失，上升趋势是性能下降）
            trend = np.polyfit(range(len(metric_values)), metric_values, 1)[0]
            
            # 如果趋势为正且较大，发送警告
            if trend > 0 and trend > threshold:
                self.send_performance_degradation(
                    metrics={key_metric: metric_values[-1], 'trend': trend},
                    threshold=threshold,
                    suggestion="考虑降低学习率或增加正则化"
                )
        else:
            # 计算趋势（对于准确率，下降趋势是性能下降）
            trend = np.polyfit(range(len(metric_values)), metric_values, 1)[0]
            
            # 如果趋势为负且较大，发送警告
            if trend < 0 and abs(trend) > threshold:
                self.send_performance_degradation(
                    metrics={key_metric: metric_values[-1], 'trend': trend},
                    threshold=threshold,
                    suggestion="考虑调整学习率或检查数据分布"
                )
    
    def _format_time_delta(self, seconds: float) -> str:
        """格式化时间差
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间字符串
        """
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}小时 {minutes}分钟"
        elif minutes > 0:
            return f"{minutes}分钟 {seconds}秒"
        else:
            return f"{seconds}秒"
    
    async def on_periodic_notification(self):
        """定期通知回调
        每30分钟发送一次训练进度更新
        """
        while True:
            await asyncio.sleep(self.notify_every * 60)  # 每notify_every分钟发送一次
            # 计算当前训练进度
            current_epoch = self.epoch_metrics['epochs'][-1] if self.epoch_metrics['epochs'] else 0
            total_epochs = self.stats['total_epochs']
            remaining_time = (total_epochs - current_epoch) * (self.stats['epoch_times'][-1] / 60) if self.stats['epoch_times'] else 0  # 估算剩余时间
            message = f"当前训练进度: {current_epoch}/{total_epochs} 预计剩余时间: {self._format_time_delta(remaining_time)}"
            await self._send_discord_message(message)
            self.logger.info(f"发送阶段性通知: {message}")

    def create_learning_curve_chart(self) -> str:
        """创建学习曲线图表
        
        Returns:
            图表文件路径
        """
        if not self.epoch_metrics['epochs']:
            return None
            
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # 绘制损失
        ax1.plot(self.epoch_metrics['epochs'], self.epoch_metrics['train_loss'], 'b-', label='训练损失')
        if self.epoch_metrics['val_loss']:
            ax1.plot(self.epoch_metrics['epochs'][:len(self.epoch_metrics['val_loss'])], 
                    self.epoch_metrics['val_loss'], 'r-', label='验证损失')
        ax1.set_title('训练和验证损失')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制准确率
        ax2.plot(self.epoch_metrics['epochs'], self.epoch_metrics['train_accuracy'], 'b-', label='训练准确率')
        if self.epoch_metrics['val_accuracy']:
            ax2.plot(self.epoch_metrics['epochs'][:len(self.epoch_metrics['val_accuracy'])], 
                    self.epoch_metrics['val_accuracy'], 'r-', label='验证准确率')
        ax2.set_title('训练和验证准确率')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('准确率')
        ax2.legend()
        ax2.grid(True)
        
        # 保存图表
        fig.tight_layout()
        chart_path = os.path.join(self.charts_dir, f"learning_curve_{int(time.time())}.png")
        plt.savefig(chart_path)
        plt.close(fig)
        
        return chart_path

    def create_signal_performance_chart(self) -> str:
        """创建信号性能图表
        
        Returns:
            图表文件路径
        """
        signal_data = self.stats['signal_metrics']
        
        # 计算各类信号的准确率
        signal_accuracies = {}
        for signal_type, metrics in signal_data.items():
            if metrics['total'] > 0:
                signal_accuracies[signal_type] = metrics['correct'] / metrics['total']
            else:
                signal_accuracies[signal_type] = 0
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 获取信号类型和对应的准确率
        signal_types = list(signal_accuracies.keys())
        accuracies = [signal_accuracies[t] for t in signal_types]
        
        # 使用不同颜色
        colors = ['green' if t == 'bullish' else 'red' if t == 'bearish' else 'blue' for t in signal_types]
        
        # 绘制柱状图
        bars = ax.bar(signal_types, accuracies, color=colors)
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2%}', ha='center', va='bottom')
        
        # 设置标题和标签
        ax.set_title('不同信号类型的准确率')
        ax.set_xlabel('信号类型')
        ax.set_ylabel('准确率')
        ax.set_ylim(0, 1.1)  # 设置y轴上限
        
        # 添加样本数量注释
        for i, signal_type in enumerate(signal_types):
            ax.text(i, 0.05, f"n={signal_data[signal_type]['total']}", ha='center')
        
        # 保存图表
        chart_path = os.path.join(self.charts_dir, f"signal_performance_{int(time.time())}.png")
        plt.savefig(chart_path)
        plt.close(fig)
        
        return chart_path

    def create_memory_usage_chart(self) -> str:
        """创建内存使用图表
        
        Returns:
            图表文件路径
        """
        if not self.stats['memory_usage']:
            return None
            
        # 解析数据
        timestamps = [t[0] for t in self.stats['memory_usage']]
        memory_mb = [t[1] for t in self.stats['memory_usage']]
        
        # 转换时间戳为相对时间（分钟）
        start_time = timestamps[0]
        relative_times = [(t - start_time) / 60 for t in timestamps]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(relative_times, memory_mb, 'b-')
        
        # 设置标题和标签
        ax.set_title('训练过程内存使用')
        ax.set_xlabel('时间（分钟）')
        ax.set_ylabel('内存使用（MB）')
        ax.grid(True)
        
        # 保存图表
        chart_path = os.path.join(self.charts_dir, f"memory_usage_{int(time.time())}.png")
        plt.savefig(chart_path)
        plt.close(fig)
        
        return chart_path

    def create_training_summary_chart(self) -> str:
        """创建训练摘要图表
        
        Returns:
            图表文件路径
        """
        # 创建包含多个指标的摘要图表
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 左上: 损失曲线
        if self.epoch_metrics['train_loss']:
            axs[0, 0].plot(self.epoch_metrics['epochs'], self.epoch_metrics['train_loss'], 'b-', label='训练')
            if self.epoch_metrics['val_loss']:
                axs[0, 0].plot(self.epoch_metrics['epochs'][:len(self.epoch_metrics['val_loss'])], 
                              self.epoch_metrics['val_loss'], 'r-', label='验证')
            axs[0, 0].set_title('损失曲线')
            axs[0, 0].set_xlabel('轮次')
            axs[0, 0].set_ylabel('损失')
            axs[0, 0].legend()
            axs[0, 0].grid(True)
        
        # 2. 右上: 准确率曲线
        if self.epoch_metrics['train_accuracy']:
            axs[0, 1].plot(self.epoch_metrics['epochs'], self.epoch_metrics['train_accuracy'], 'b-', label='训练')
            if self.epoch_metrics['val_accuracy']:
                axs[0, 1].plot(self.epoch_metrics['epochs'][:len(self.epoch_metrics['val_accuracy'])], 
                              self.epoch_metrics['val_accuracy'], 'r-', label='验证')
            axs[0, 1].set_title('准确率曲线')
            axs[0, 1].set_xlabel('轮次')
            axs[0, 1].set_ylabel('准确率')
            axs[0, 1].legend()
            axs[0, 1].grid(True)
        
        # 3. 左下: 学习率变化
        if self.epoch_metrics['learning_rate']:
            axs[1, 0].plot(self.epoch_metrics['epochs'][:len(self.epoch_metrics['learning_rate'])], 
                          self.epoch_metrics['learning_rate'], 'g-')
            axs[1, 0].set_title('学习率变化')
            axs[1, 0].set_xlabel('轮次')
            axs[1, 0].set_ylabel('学习率')
            axs[1, 0].set_yscale('log')  # 对数刻度
            axs[1, 0].grid(True)
        
        # 4. 右下: 信号性能饼图
        signal_data = self.stats['signal_metrics']
        labels = []
        sizes = []
        colors = ['green', 'red', 'blue']
        
        for i, (signal_type, metrics) in enumerate(signal_data.items()):
            if metrics['total'] > 0:
                labels.append(f"{signal_type}\n{metrics['correct']}/{metrics['total']}")
                sizes.append(metrics['total'])
        
        if sizes:
            axs[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                         startangle=90)
            axs[1, 1].axis('equal')  # 等比例
            axs[1, 1].set_title('信号分布')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(self.charts_dir, f"training_summary_{int(time.time())}.png")
        plt.savefig(chart_path)
        plt.close(fig)
        
        return chart_path

    def send_training_started(self, config: Dict, model_info: Dict = None):
        """发送训练开始通知
        
        Args:
            config: 训练配置
            model_info: 模型信息
        """
        # 设置训练进行中标志
        self.stats['training_in_progress'] = True
        
        # 重置开始时间
        self.stats['start_time'] = time.time()
        
        # 构建消息
        message = f"🚀 **训练已开始** 🚀\n\n"
        
        # 添加配置信息
        message += "**训练配置:**\n"
        for key, value in config.items():
            if key in ['learning_rate', 'batch_size', 'epochs', 'validation_interval', 'checkpoint_interval', 'early_stopping_patience']:
                message += f"- {key}: {value}\n"
        
        # 添加模型信息
        if model_info:
            message += "\n**模型信息:**\n"
            for key, value in model_info.items():
                if key in ['name', 'type', 'parameters', 'layers']:
                    message += f"- {key}: {value}\n"
        
        # 添加到通知队列
        self.notification_queue.append({
            'message': message,
            'embed_color': 0x2ecc71  # 绿色
        })
    
    def send_training_completed(self, training_time: float, final_metrics: Dict, best_metrics: Dict = None):
        """发送训练完成通知
        
        Args:
            training_time: 训练时间（秒）
            final_metrics: 最终指标
            best_metrics: 最佳指标
        """
        # 设置训练进行中标志
        self.stats['training_in_progress'] = False
        
        # 构建消息
        message = f"✅ **训练已完成** ✅\n\n"
        message += f"**总训练时间:** {self._format_time_delta(training_time)}\n\n"
        
        # 添加最终指标
        message += "**最终指标:**\n"
        for key, value in final_metrics.items():
            message += f"- {key}: {value:.4f}\n"
        
        # 添加最佳指标
        if best_metrics:
            message += "\n**最佳指标:**\n"
            for key, value in best_metrics.items():
                message += f"- {key}: {value:.4f}\n"
        
        # 添加到通知队列
        self.notification_queue.append({
            'message': message,
            'embed_color': 0x2ecc71  # 绿色
        })
    
    def send_training_error(self, error_message: str, traceback_str: str = None):
        """发送训练错误通知
        
        Args:
            error_message: 错误消息
            traceback_str: 异常追踪字符串
        """
        # 构建消息
        message = f"❌ **训练中发生错误** ❌\n\n"
        message += f"**错误:** {error_message}\n"
        
        # 添加追踪信息（如果有）
        if traceback_str:
            message += "\n**详细信息:**\n```\n"
            # 限制追踪信息长度
            if len(traceback_str) > 1000:
                message += traceback_str[:997] + "..."
            else:
                message += traceback_str
            message += "\n```"
        
        # 添加到通知队列
        self.notification_queue.append({
            'message': message,
            'embed_color': 0xe74c3c  # 红色
        })
    
    def send_performance_degradation(self, metrics: Dict, threshold: float, suggestion: str = None):
        """发送性能下降通知
        
        Args:
            metrics: 性能指标
            threshold: 下降阈值
            suggestion: 建议措施
        """
        # 构建消息
        message = f"⚠️ **性能下降警告** ⚠️\n\n"
        message += "**当前指标:**\n"
        
        for key, value in metrics.items():
            message += f"- {key}: {value:.4f}\n"
        
        message += f"\n**性能下降超过阈值:** {threshold:.2f}\n"
        
        # 添加建议
        if suggestion:
            message += f"\n**建议:** {suggestion}"
        
        # 添加到通知队列
        self.notification_queue.append({
            'message': message,
            'embed_color': 0xf39c12  # 橙色
        })
        
        # 增加警告计数
        self.warning_count += 1


# 单例实例
_notifier_instance = None

def get_notifier(webhook_url=None, interval_minutes=30, **kwargs) -> ProgressNotifier:
    """获取进度通知器实例
    
    Args:
        webhook_url: Discord webhook URL
        interval_minutes: 通知间隔（分钟）
        **kwargs: 其他参数
        
    Returns:
        进度通知器实例
    """
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = ProgressNotifier(
            webhook_url=webhook_url, 
            interval_minutes=interval_minutes,
            **kwargs
        )
    return _notifier_instance 