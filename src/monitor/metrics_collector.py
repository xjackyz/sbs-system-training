import psutil
import GPUtil
from typing import Dict, List
import time
import logging
from datetime import datetime
import numpy as np
from collections import deque

class MetricsCollector:
    def __init__(self, history_size: int = 3600):  # 默认保存1小时的历史数据
        self.history_size = history_size
        self.metrics_history = {
            'gpu': deque(maxlen=history_size),
            'cpu': deque(maxlen=history_size),
            'memory': deque(maxlen=history_size),
            'network': deque(maxlen=history_size),
            'disk': deque(maxlen=history_size),
            'training': deque(maxlen=history_size)
        }
        self.last_network_io = None
        self.last_disk_io = None
        self.last_collection_time = None

    def collect_gpu_metrics(self) -> Dict:
        """收集GPU指标"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {
                    'gpu_usage': 0,
                    'gpu_memory_used': 0,
                    'gpu_memory_total': 0,
                    'gpu_temperature': 0
                }
            
            gpu = gpus[0]  # 假设使用第一个GPU
            return {
                'gpu_usage': gpu.load * 100,
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_temperature': gpu.temperature
            }
        except Exception as e:
            logging.error(f"GPU指标收集失败: {e}")
            return {
                'gpu_usage': 0,
                'gpu_memory_used': 0,
                'gpu_memory_total': 0,
                'gpu_temperature': 0
            }

    def collect_cpu_metrics(self) -> Dict:
        """收集CPU指标"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_freq = psutil.cpu_freq()
            cpu_temp = psutil.sensors_temperatures().get('coretemp', [])
            return {
                'cpu_usage_per_core': cpu_percent,
                'cpu_usage_avg': sum(cpu_percent) / len(cpu_percent),
                'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
                'cpu_freq_max': cpu_freq.max if cpu_freq else 0,
                'cpu_temperature': max(temp.current for temp in cpu_temp) if cpu_temp else 0
            }
        except Exception as e:
            logging.error(f"CPU指标收集失败: {e}")
            return {
                'cpu_usage_per_core': [],
                'cpu_usage_avg': 0,
                'cpu_freq_current': 0,
                'cpu_freq_max': 0,
                'cpu_temperature': 0
            }

    def collect_memory_metrics(self) -> Dict:
        """收集内存指标"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                'memory_total': memory.total,
                'memory_used': memory.used,
                'memory_free': memory.free,
                'memory_percent': memory.percent,
                'swap_total': swap.total,
                'swap_used': swap.used,
                'swap_free': swap.free,
                'swap_percent': swap.percent
            }
        except Exception as e:
            logging.error(f"内存指标收集失败: {e}")
            return {
                'memory_total': 0,
                'memory_used': 0,
                'memory_free': 0,
                'memory_percent': 0,
                'swap_total': 0,
                'swap_used': 0,
                'swap_free': 0,
                'swap_percent': 0
            }

    def collect_network_metrics(self) -> Dict:
        """收集网络指标"""
        try:
            network_io = psutil.net_io_counters()
            current_time = time.time()
            
            if self.last_network_io and self.last_collection_time:
                time_diff = current_time - self.last_collection_time
                bytes_sent = (network_io.bytes_sent - self.last_network_io.bytes_sent) / time_diff
                bytes_recv = (network_io.bytes_recv - self.last_network_io.bytes_recv) / time_diff
            else:
                bytes_sent = 0
                bytes_recv = 0
            
            self.last_network_io = network_io
            self.last_collection_time = current_time
            
            return {
                'network_bytes_sent': bytes_sent,
                'network_bytes_recv': bytes_recv,
                'network_packets_sent': network_io.packets_sent,
                'network_packets_recv': network_io.packets_recv,
                'network_errin': network_io.errin,
                'network_errout': network_io.errout
            }
        except Exception as e:
            logging.error(f"网络指标收集失败: {e}")
            return {
                'network_bytes_sent': 0,
                'network_bytes_recv': 0,
                'network_packets_sent': 0,
                'network_packets_recv': 0,
                'network_errin': 0,
                'network_errout': 0
            }

    def collect_disk_metrics(self) -> Dict:
        """收集磁盘指标"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            current_time = time.time()
            
            if self.last_disk_io and self.last_collection_time:
                time_diff = current_time - self.last_collection_time
                read_speed = (disk_io.read_bytes - self.last_disk_io.read_bytes) / time_diff
                write_speed = (disk_io.write_bytes - self.last_disk_io.write_bytes) / time_diff
            else:
                read_speed = 0
                write_speed = 0
            
            self.last_disk_io = disk_io
            
            return {
                'disk_total': disk_usage.total,
                'disk_used': disk_usage.used,
                'disk_free': disk_usage.free,
                'disk_percent': disk_usage.percent,
                'disk_read_speed': read_speed,
                'disk_write_speed': write_speed,
                'disk_read_count': disk_io.read_count,
                'disk_write_count': disk_io.write_count
            }
        except Exception as e:
            logging.error(f"磁盘指标收集失败: {e}")
            return {
                'disk_total': 0,
                'disk_used': 0,
                'disk_free': 0,
                'disk_percent': 0,
                'disk_read_speed': 0,
                'disk_write_speed': 0,
                'disk_read_count': 0,
                'disk_write_count': 0
            }

    def collect_all_metrics(self) -> Dict:
        """收集所有指标"""
        timestamp = datetime.now().isoformat()
        metrics = {
            'timestamp': timestamp,
            'gpu': self.collect_gpu_metrics(),
            'cpu': self.collect_cpu_metrics(),
            'memory': self.collect_memory_metrics(),
            'network': self.collect_network_metrics(),
            'disk': self.collect_disk_metrics()
        }
        
        # 保存历史数据
        for category in metrics:
            if category != 'timestamp':
                self.metrics_history[category].append(metrics[category])
        
        return metrics

    def get_metrics_history(self, category: str, metric_name: str) -> List:
        """获取特定指标的历史数据"""
        if category not in self.metrics_history:
            return []
        
        history = self.metrics_history[category]
        return [data.get(metric_name, 0) for data in history]

    def get_metrics_statistics(self, category: str, metric_name: str) -> Dict:
        """获取特定指标的统计信息"""
        history = self.get_metrics_history(category, metric_name)
        if not history:
            return {
                'min': 0,
                'max': 0,
                'avg': 0,
                'std': 0
            }
        
        return {
            'min': min(history),
            'max': max(history),
            'avg': sum(history) / len(history),
            'std': np.std(history) if len(history) > 1 else 0
        }

    def get_system_health(self) -> Dict:
        """获取系统健康状态"""
        gpu_metrics = self.collect_gpu_metrics()
        cpu_metrics = self.collect_cpu_metrics()
        memory_metrics = self.collect_memory_metrics()
        
        return {
            'status': 'healthy',  # 或 'warning' 或 'critical'
            'warnings': [
                warning for warning in [
                    '显存使用率过高' if gpu_metrics['gpu_memory_used'] / gpu_metrics['gpu_memory_total'] > 0.9 else None,
                    'GPU温度过高' if gpu_metrics['gpu_temperature'] > 80 else None,
                    'CPU使用率过高' if cpu_metrics['cpu_usage_avg'] > 90 else None,
                    '内存使用率过高' if memory_metrics['memory_percent'] > 90 else None
                ] if warning is not None
            ]
        } 