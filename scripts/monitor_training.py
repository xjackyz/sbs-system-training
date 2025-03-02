#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练监控脚本
用于实时监控训练进度、性能指标和资源使用情况
"""

import os
import sys
import time
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger
from src.cache.redis_manager import RedisManager
from src.self_supervised.utils.visualization import LearningVisualizer

# 设置日志
logger = setup_logger('training_monitor')

class TrainingMonitor:
    def __init__(self, config_path: str):
        """初始化训练监控器"""
        self.config = self._load_config(config_path)
        self.cache_manager = RedisManager(config=self.config)
        self.visualizer = LearningVisualizer(save_dir='logs/monitoring')
        self.metrics_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def monitor_training(self, interval: int = 60):
        """持续监控训练过程"""
        try:
            while True:
                # 收集指标
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 更新可视化
                self._update_visualizations()
                
                # 检查异常
                self._check_anomalies(metrics)
                
                # 生成报告
                self._generate_report()
                
                # 等待下一次收集
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("监控已停止")
            self._generate_final_report()
            
    def _collect_metrics(self) -> Dict:
        """收集各项指标"""
        metrics = {
            'timestamp': datetime.now(),
            'system': self._get_system_metrics(),
            'training': self._get_training_metrics(),
            'performance': self._get_performance_metrics()
        }
        return metrics
        
    def _get_system_metrics(self) -> Dict:
        """获取系统资源指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用
        memory = psutil.virtual_memory()
        memory_used = memory.used / (1024 * 1024 * 1024)  # GB
        memory_total = memory.total / (1024 * 1024 * 1024)  # GB
        
        # GPU使用率
        gpus = GPUtil.getGPUs()
        gpu_metrics = []
        for gpu in gpus:
            gpu_metrics.append({
                'id': gpu.id,
                'load': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            })
            
        return {
            'cpu_percent': cpu_percent,
            'memory_used': memory_used,
            'memory_total': memory_total,
            'gpu_metrics': gpu_metrics
        }
        
    def _get_training_metrics(self) -> Dict:
        """获取训练相关指标"""
        # 从Redis缓存获取最新训练结果
        latest_results = self.cache_manager.get(
            f"training_results_{datetime.now().strftime('%Y%m%d')}"
        )
        
        if not latest_results:
            return {}
            
        return {
            'loss': latest_results.get('loss'),
            'accuracy': latest_results.get('accuracy'),
            'learning_rate': latest_results.get('learning_rate'),
            'batch_size': latest_results.get('batch_size'),
            'epoch': latest_results.get('epoch'),
            'validation_metrics': latest_results.get('validation_metrics')
        }
        
    def _get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        latest_results = self.cache_manager.get(
            f"training_results_{datetime.now().strftime('%Y%m%d')}"
        )
        
        if not latest_results:
            return {}
            
        return {
            'training_speed': latest_results.get('samples_per_second'),
            'gpu_utilization': latest_results.get('gpu_utilization'),
            'memory_utilization': latest_results.get('memory_utilization'),
            'cache_hit_rate': self.cache_manager.get_stats().get('hit_rate')
        }
        
    def _update_visualizations(self):
        """更新可视化图表"""
        if not self.metrics_history:
            return
            
        # 转换为DataFrame
        df = pd.DataFrame([
            {
                'timestamp': m['timestamp'],
                'loss': m['training'].get('loss'),
                'accuracy': m['training'].get('accuracy'),
                'learning_rate': m['training'].get('learning_rate'),
                'gpu_utilization': m['performance'].get('gpu_utilization'),
                'memory_utilization': m['performance'].get('memory_utilization'),
                'training_speed': m['performance'].get('training_speed')
            }
            for m in self.metrics_history
        ])
        
        # 绘制训练指标
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失和准确率
        ax1 = axes[0, 0]
        ax1.plot(df['timestamp'], df['loss'], label='Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Loss')
        
        ax2 = axes[0, 1]
        ax2.plot(df['timestamp'], df['accuracy'], label='Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Accuracy')
        
        # GPU和内存使用
        ax3 = axes[1, 0]
        ax3.plot(df['timestamp'], df['gpu_utilization'], label='GPU')
        ax3.plot(df['timestamp'], df['memory_utilization'], label='Memory')
        ax3.set_title('Resource Utilization')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Utilization %')
        ax3.legend()
        
        # 训练速度
        ax4 = axes[1, 1]
        ax4.plot(df['timestamp'], df['training_speed'], label='Speed')
        ax4.set_title('Training Speed')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Samples/Second')
        
        plt.tight_layout()
        plt.savefig('logs/monitoring/training_metrics.png')
        plt.close()
        
    def _check_anomalies(self, metrics: Dict):
        """检查异常情况"""
        # GPU温度检查
        for gpu in metrics['system']['gpu_metrics']:
            if gpu['temperature'] > 80:
                logger.warning(f"GPU {gpu['id']} 温度过高: {gpu['temperature']}°C")
                
        # 内存使用检查
        memory_usage = metrics['system']['memory_used'] / metrics['system']['memory_total']
        if memory_usage > 0.95:
            logger.warning(f"内存使用率过高: {memory_usage*100:.1f}%")
            
        # 训练指标检查
        if metrics['training'].get('loss'):
            if metrics['training']['loss'] > 100:
                logger.warning(f"损失值异常: {metrics['training']['loss']}")
                
        # 性能检查
        if metrics['performance'].get('training_speed'):
            if metrics['performance']['training_speed'] < 1:
                logger.warning("训练速度异常偏低")
                
    def _generate_report(self):
        """生成监控报告"""
        if not self.metrics_history:
            return
            
        latest = self.metrics_history[-1]
        
        report = f"""
训练监控报告 - {latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

系统资源:
- CPU使用率: {latest['system']['cpu_percent']:.1f}%
- 内存使用: {latest['system']['memory_used']:.1f}GB / {latest['system']['memory_total']:.1f}GB
- GPU使用情况:
{self._format_gpu_info(latest['system']['gpu_metrics'])}

训练状态:
- 当前轮次: {latest['training'].get('epoch', 'N/A')}
- 损失值: {latest['training'].get('loss', 'N/A'):.4f}
- 准确率: {latest['training'].get('accuracy', 'N/A'):.4f}
- 学习率: {latest['training'].get('learning_rate', 'N/A'):.6f}
- 批次大小: {latest['training'].get('batch_size', 'N/A')}

性能指标:
- 训练速度: {latest['performance'].get('training_speed', 'N/A'):.2f} samples/s
- GPU利用率: {latest['performance'].get('gpu_utilization', 'N/A'):.1f}%
- 内存利用率: {latest['performance'].get('memory_utilization', 'N/A'):.1f}%
- 缓存命中率: {latest['performance'].get('cache_hit_rate', 'N/A'):.1f}%
"""
        
        # 保存报告
        report_path = Path('logs/monitoring/reports')
        report_path.mkdir(parents=True, exist_ok=True)
        
        with open(report_path / f"report_{latest['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
            f.write(report)
            
    def _format_gpu_info(self, gpu_metrics: List[Dict]) -> str:
        """格式化GPU信息"""
        return '\n'.join([
            f"  GPU {gpu['id']}: {gpu['load']:.1f}% 使用率, "
            f"{gpu['memory_used']}/{gpu['memory_total']}MB 内存, "
            f"{gpu['temperature']}°C"
            for gpu in gpu_metrics
        ])
        
    def _generate_final_report(self):
        """生成最终报告"""
        if not self.metrics_history:
            return
            
        start_time = self.metrics_history[0]['timestamp']
        end_time = self.metrics_history[-1]['timestamp']
        duration = end_time - start_time
        
        # 计算平均值和统计信息
        df = pd.DataFrame([
            {
                'loss': m['training'].get('loss'),
                'accuracy': m['training'].get('accuracy'),
                'gpu_utilization': m['performance'].get('gpu_utilization'),
                'training_speed': m['performance'].get('training_speed')
            }
            for m in self.metrics_history
        ])
        
        final_report = f"""
训练总结报告
{'='*50}

训练时长: {duration}

性能统计:
- 平均损失: {df['loss'].mean():.4f}
- 最终准确率: {df['accuracy'].iloc[-1]:.4f}
- 平均GPU利用率: {df['gpu_utilization'].mean():.1f}%
- 平均训练速度: {df['training_speed'].mean():.2f} samples/s

详细指标:
{df.describe().to_string()}
"""
        
        # 保存最终报告
        report_path = Path('logs/monitoring/final_report.txt')
        with open(report_path, 'w') as f:
            f.write(final_report)
            
        logger.info(f"最终报告已保存至: {report_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='训练监控工具')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                      help='配置文件路径')
    parser.add_argument('--interval', type=int, default=60,
                      help='监控间隔(秒)')
    args = parser.parse_args()
    
    monitor = TrainingMonitor(config_path=args.config)
    monitor.monitor_training(interval=args.interval)

if __name__ == '__main__':
    main() 