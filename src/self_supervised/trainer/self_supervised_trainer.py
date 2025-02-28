import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import time
import traceback
from functools import wraps

from ..model.sequence_model import SequenceModel
from ..utils.signal_tracker import SignalTracker
from ..utils.reward_mechanism import RewardMechanism
from ..utils.progress_notifier import ProgressNotifier

# 添加新的导入
import gc
import psutil
import threading
from queue import Queue
from src.notification.discord_notifier import get_discord_notifier

class MemoryTracker:
    """内存使用跟踪器
    
    用于监控训练过程中的内存使用情况
    """
    
    def __init__(self, log_interval: int = 50):
        """初始化内存跟踪器
        
        Args:
            log_interval: 日志记录间隔（批次数）
        """
        self.log_interval = log_interval
        self.memory_usage = []
        self.step_count = 0
        self.logger = logging.getLogger('memory_tracker')
        self.logger.info("内存跟踪器已初始化")
    
    def track(self):
        """记录当前内存使用情况"""
        if self.step_count % self.log_interval == 0:
            # 获取当前进程的内存使用
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # 转换为MB
            
            # 记录内存使用
            self.memory_usage.append({
                'step': self.step_count,
                'memory_mb': memory_mb,
                'timestamp': time.time()
            })
            
            # 打印日志
            self.logger.info(f"步骤 {self.step_count}: 内存使用 {memory_mb:.2f} MB")
            
            # 检查是否需要进行垃圾回收
            if len(self.memory_usage) >= 5:
                recent_usage = [entry['memory_mb'] for entry in self.memory_usage[-5:]]
                if recent_usage[-1] > 1.2 * recent_usage[0]:  # 内存增长超过20%
                    self.logger.info("检测到内存持续增长，执行主动垃圾回收...")
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.step_count += 1
    
    def get_memory_stats(self) -> Dict:
        """获取内存使用统计信息
        
        Returns:
            统计信息字典
        """
        if not self.memory_usage:
            return {
                'avg_memory_mb': 0,
                'max_memory_mb': 0,
                'min_memory_mb': 0,
                'current_memory_mb': 0
            }
        
        memory_values = [entry['memory_mb'] for entry in self.memory_usage]
        
        # 获取当前内存使用
        process = psutil.Process(os.getpid())
        current_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        return {
            'avg_memory_mb': np.mean(memory_values),
            'max_memory_mb': max(memory_values),
            'min_memory_mb': min(memory_values),
            'current_memory_mb': current_memory_mb
        }
    
    def plot_memory_usage(self, save_path: str = 'logs/memory_usage.png'):
        """绘制内存使用情况图表
        
        Args:
            save_path: 图表保存路径
        """
        if not self.memory_usage:
            self.logger.warning("没有内存使用数据可以绘制")
            return None
        
        # 提取数据
        steps = [entry['step'] for entry in self.memory_usage]
        memory_values = [entry['memory_mb'] for entry in self.memory_usage]
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.plot(steps, memory_values, 'b-')
        plt.title('训练过程内存使用情况')
        plt.xlabel('步骤')
        plt.ylabel('内存使用 (MB)')
        plt.grid(True)
        
        # 添加平均值和最大值线
        avg_memory = np.mean(memory_values)
        max_memory = max(memory_values)
        
        plt.axhline(y=avg_memory, color='g', linestyle='--', label=f'平均: {avg_memory:.2f} MB')
        plt.axhline(y=max_memory, color='r', linestyle='--', label=f'最大: {max_memory:.2f} MB')
        
        plt.legend()
        
        # 保存图表
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"内存使用图表已保存至 {save_path}")
        
        return save_path


class DataIncrementalLoader:
    """增量数据加载器
    
    分批次加载和处理数据，减少内存占用
    """
    
    def __init__(self, data_path: str, batch_size: int = 1000, window_size: int = 100, 
                 stride: int = 20, skip_rows: int = None, prefetch: int = 2,
                 use_dali: bool = True, num_threads: int = 4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.stride = stride
        self.skip_rows = skip_rows
        self.prefetch = prefetch
        self.use_dali = use_dali
        self.num_threads = num_threads
        
        # DALI管道配置
        if self.use_dali:
            self.dali_pipeline = self._create_dali_pipeline()
            
        self._init_columns()
        self._start_prefetch_thread()
        
    def _create_dali_pipeline(self):
        try:
            from nvidia.dali import pipeline_def
            import nvidia.dali.fn as fn
            import nvidia.dali.types as types
            
            @pipeline_def
            def create_pipeline():
                data = fn.readers.csv(
                    file_root=os.path.dirname(self.data_path),
                    file_list=[os.path.basename(self.data_path)],
                    num_threads=self.num_threads,
                    prefetch_queue_depth=self.prefetch,
                    skip_lines=self.skip_rows or 0,
                    device="cpu"
                )
                return data
                
            return create_pipeline(
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                device_id=0
            )
        except ImportError:
            logging.warning("NVIDIA DALI未安装，将使用标准数据加载")
            self.use_dali = False
            return None

    def _get_data_length(self) -> int:
        """获取数据总行数
        
        Returns:
            数据文件的总行数
        """
        with open(self.data_path, 'r') as f:
            return sum(1 for _ in f)
    
    def _init_columns(self):
        """初始化数据列结构"""
        # 读取标题行以获取列名
        df_sample = pd.read_csv(self.data_path, nrows=1)
        self.columns = df_sample.columns.tolist()
        
        # 确定日期列
        date_cols = [col for col in self.columns if 'date' in col.lower() or 'time' in col.lower()]
        self.date_column = date_cols[0] if date_cols else None
        
        # 确定价格列
        price_cols = [col for col in self.columns if 'price' in col.lower() or 'close' in col.lower()]
        self.price_column = price_cols[0] if price_cols else None
        
        if not self.date_column:
            self.logger.warning("未找到日期列，这可能会影响某些功能")
        
        if not self.price_column:
            self.logger.warning("未找到价格列，这可能会影响某些功能")
    
    def _start_prefetch_thread(self):
        """启动数据预加载线程"""
        if self.prefetch <= 0:
            return
            
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        self.logger.info(f"启动数据预加载线程，预加载 {self.prefetch} 个批次")
    
    def _prefetch_worker(self):
        try:
            while not self.stop_flag:
                if self.use_dali and self.dali_pipeline:
                    # 使用DALI加载数据
                    pipe_output = self.dali_pipeline.run()
                    batch_data = self._process_dali_output(pipe_output)
                else:
                    # 标准数据加载
                    batch_data = self._load_next_batch()
                
                if batch_data is not None:
                    self.prefetch_queue.put(batch_data)
                    
                if len(self.prefetch_queue.queue) >= self.prefetch:
                    time.sleep(0.1)
                    
        except Exception as e:
            logging.error(f"预取线程错误: {e}")
            
    def _process_dali_output(self, pipe_output):
        """处理DALI输出数据"""
        try:
            # 转换DALI输出为pandas DataFrame
            data_dict = {}
            for col, tensor in zip(self.columns, pipe_output):
                data_dict[col] = tensor.as_cpu().as_array()
            return pd.DataFrame(data_dict)
        except Exception as e:
            logging.error(f"DALI数据处理错误: {e}")
            return None
    
    def stop(self):
        """停止预加载线程"""
        if self.prefetch_thread:
            self.stop_prefetch.set()
            self.prefetch_thread.join(timeout=5)
            self.logger.info("数据预加载线程已停止")
    
    def has_next_batch(self) -> bool:
        """检查是否还有下一批数据
        
        Returns:
            是否还有更多数据
        """
        return self.current_position < self.data_length
    
    def get_next_batch(self) -> Tuple[pd.DataFrame, int]:
        """获取下一批数据
        
        Returns:
            (数据批次DataFrame, 下一个起始位置)
        """
        if not self.has_next_batch():
            return None, self.current_position
        
        # 如果使用预加载
        if self.prefetch > 0 and not self.queue.empty():
            df_batch, batch_start, next_position = self.queue.get()
            self.current_position = next_position
            self._check_data_integrity(df_batch)  # 检查数据完整性
            return df_batch, next_position
        
        # 直接加载（未使用预加载或预加载队列为空）
        batch_start = self.current_position
        batch_end = min(batch_start + self.batch_size, self.data_length)
        
        try:
            # 读取数据块
            df_batch = pd.read_csv(
                self.data_path, 
                skiprows=range(1, batch_start),  # 跳过之前的行（但保留标题行）
                nrows=batch_end - batch_start,   # 读取的行数
                header=0                         # 第一行是标题
            )
            self._check_data_integrity(df_batch)  # 检查数据完整性
            # 更新位置
            self.current_position = batch_end
            return df_batch, batch_end
        except Exception as e:
            self.logger.error(f"加载数据批次时出错: {e}")
            return None, batch_start
    
    def _check_data_integrity(self, df: pd.DataFrame):
        """检查数据完整性和有效性
        
        Args:
            df: 数据批次DataFrame
        """
        if df.isnull().values.any():
            raise ValueError("数据中存在缺失值")
        if (df < 0).any().any():  # 假设负值是异常值
            raise ValueError("数据中存在异常值")
        
        # 检查列名和数据类型
        expected_columns = ['date', 'price']  # 示例列名
        for col in expected_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 检查数据类型
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise ValueError("日期列的数据类型不正确")
        if not pd.api.types.is_numeric_dtype(df['price']):
            raise ValueError("价格列的数据类型不正确")
    
    def get_windows_from_batch(self, df_batch: pd.DataFrame) -> List[pd.DataFrame]:
        """从数据批次中提取窗口
        
        根据窗口大小和步长从数据批次中提取多个窗口
        
        Args:
            df_batch: 数据批次DataFrame
            
        Returns:
            窗口DataFrame列表
        """
        windows = []
        
        # 如果批次数据不足一个窗口，返回空列表
        if len(df_batch) < self.window_size:
            return windows
        
        # 提取窗口
        for i in range(0, len(df_batch) - self.window_size + 1, self.stride):
            window = df_batch.iloc[i:i+self.window_size].copy()
            windows.append(window)
        
        return windows
    
    def process_all_data(self, processor_func, progress_callback=None):
        """处理所有数据
        
        使用提供的处理函数处理所有数据窗口
        
        Args:
            processor_func: 处理函数，接收窗口DataFrame作为输入
            progress_callback: 进度回调函数，接收处理进度作为输入
            
        Returns:
            处理结果列表
        """
        results = []
        total_windows = 0
        processed_windows = 0
        
        # 估计总窗口数
        estimated_total_windows = (self.data_length - self.window_size) // self.stride + 1
        
        # 处理所有批次
        with tqdm(total=estimated_total_windows, desc="处理数据窗口") as pbar:
            while self.has_next_batch():
                df_batch, _ = self.get_next_batch()
                
                if df_batch is None:
                    continue
                
                # 提取窗口
                windows = self.get_windows_from_batch(df_batch)
                total_windows += len(windows)
                
                # 处理每个窗口
                for window in windows:
                    try:
                        result = processor_func(window)
                        if result:
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"处理窗口时出错: {e}")
                    
                    processed_windows += 1
                    pbar.update(1)
                    
                    # 调用进度回调
                    if progress_callback:
                        progress_callback(processed_windows, estimated_total_windows)
                
                # 执行垃圾回收以减少内存占用
                if processed_windows % 1000 == 0:
                    gc.collect()
        
        self.logger.info(f"已处理 {processed_windows} 个窗口，生成 {len(results)} 个结果")
        return results


class ChartDataset(Dataset):
    """图表数据集"""
    
    def __init__(self, data_dir: str, transform=None):
        """初始化数据集
        
        Args:
            data_dir: 数据目录
            transform: 数据转换函数
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """加载样本数据"""
        samples = []
        
        # 遍历数据目录中的所有JSON文件
        for json_file in self.data_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    samples.append(data)
            except Exception as e:
                logging.warning(f"加载样本 {json_file} 失败: {e}")
        
        logging.info(f"加载了 {len(samples)} 个样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 提取图表数据和标签
        chart_data = sample.get('chart_data', {})
        labels = sample.get('labels', {})
        
        # 应用转换
        if self.transform:
            chart_data = self.transform(chart_data)
        
        return chart_data, labels


class EarlyStopping:
    """早停机制
    
    当模型性能不再提升时，提前结束训练
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """初始化早停
        
        Args:
            patience: 容忍多少个轮次性能不提升
            min_delta: 认为是提升的最小变化量
            mode: 'min'表示指标越小越好，'max'表示指标越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = logging.getLogger('early_stopping')
        self.logger.info("早停机制已初始化")
    
    def __call__(self, score: float) -> bool:
        """检查是否应该早停
        
        Args:
            score: 当前性能指标
            
        Returns:
            是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                # 性能有提升
                self.best_score = score
                self.counter = 0
            else:
                # 性能没有提升
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                # 性能有提升
                self.best_score = score
                self.counter = 0
            else:
                # 性能没有提升
                self.counter += 1
        
        # 检查是否应该早停
        if self.counter >= self.patience:
            self.logger.info(f"触发早停，{self.patience}轮内性能未提升")
            self.early_stop = True
            return True
            
        return False
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger.info("早停状态已重置")


def enhanced_error_handler(func):
    """装饰器：增强错误处理和通知"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            await self.notifier.send_status_update('错误发生', error_details)
            raise
    return wrapper


class SelfSupervisedTrainer:
    """自监督学习训练器
    
    该类负责训练自监督学习模型，管理数据加载、模型训练和评估等过程。
    """
    
    def __init__(self, model: SequenceModel, data_dir: str, save_dir: str, device: str = None):
        """初始化训练器
        
        Args:
            model: 序列模型
            data_dir: 数据目录
            save_dir: 保存目录
            device: 设备 ('cuda' 或 'cpu')
        """
        self.model = model
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证模型路径
        model_path = Path("models/llava-sbs")
        if not model_path.exists():
            raise ValueError(f"模型路径 {model_path} 不存在")
        
        # 设置设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设备检查
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.logger.error("CUDA设备不可用，自动回退到CPU。")
            self.device = 'cpu'
        self.model = self.model.to(self.device)
        
        # 创建信号跟踪器
        self.signal_tracker = SignalTracker(save_dir=str(self.save_dir / 'signals'))
        
        # 创建奖励机制
        self.reward_mechanism = RewardMechanism(self.signal_tracker)
        
        # 设置日志
        self.logger = logging.getLogger('self_supervised_trainer')
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {}
        }
        
        # 当前训练阶段
        self.current_stage = 1
        
        # 初始化优化器和损失函数
        self._init_optimizer_and_loss()
        
        # 添加内存跟踪器
        self.memory_tracker = MemoryTracker(log_interval=20)
        
        # 增量数据加载支持
        self.incremental_loader = None
        self.save_memory = False
        
        # 添加学习率调度器
        self.scheduler = None
        
        # 早停机制
        self.early_stopping = None
        
        # 添加通知器
        self.notifier = get_discord_notifier()
        
        # 初始化混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        self.use_amp = True if self.device == 'cuda' else False
        
        # 初始化梯度检查点
        self.use_gradient_checkpointing = False
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            self.use_gradient_checkpointing = True
            
        # 初始化Weights & Biases
        self.wandb_logger = None
        
        # 移动模型到设备
        self.model.to(self.device)
    
    def _init_optimizer_and_loss(self):
        """初始化优化器和损失函数"""
        # 基础优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # 使用奖励机制包装优化器
        self.optimizer = self.reward_mechanism.create_reward_based_optimizer(self.optimizer)
        
        # 损失函数
        self.sequence_loss = nn.CrossEntropyLoss()
        self.signal_loss = nn.CrossEntropyLoss()
        self.price_loss = nn.MSELoss()
        
        # 使用奖励机制包装损失函数
        self.weighted_sequence_loss = self.reward_mechanism.weighted_loss_function(self.sequence_loss)
        self.weighted_signal_loss = self.reward_mechanism.weighted_loss_function(self.signal_loss)
        self.weighted_price_loss = self.reward_mechanism.weighted_loss_function(self.price_loss)
    
    def _create_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        try:
            dataset = ChartDataset(self.data_dir)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        except Exception as e:
            self.logger.error(f"创建数据加载器时出错: {e}")
            raise
    
    def setup_lr_scheduler(self, scheduler_type: str = 'plateau', **kwargs):
        """设置学习率调度器
        
        Args:
            scheduler_type: 调度器类型
            **kwargs: 其他参数
        """
        if scheduler_type == 'plateau':
            # 根据验证损失调整学习率
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',  # 指标越小越好
                factor=kwargs.get('factor', 0.5),  # 学习率衰减因子
                patience=kwargs.get('patience', 5),  # 容忍轮次
                verbose=True
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    def setup_early_stopping(self, patience: int = 10, min_delta: float = 0.001, monitor: str = 'val_loss'):
        """设置早停机制
        
        Args:
            patience: 容忍多少轮性能不提升
            min_delta: 认为是提升的最小变化量
            monitor: 监控的指标，'val_loss'或'val_accuracy'
        """
        mode = 'min' if 'loss' in monitor else 'max'
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            mode=mode
        )
        self.logger.info(f"设置早停机制，监控指标: {monitor}, 容忍轮次: {patience}")

    @enhanced_error_handler
    async def train(self, num_epochs: int = 10, batch_size: int = 32, validate_every: int = 1,
              learning_rate: float = 0.001, accumulation_steps: int = 1):
        """训练自监督学习模型。
        
        Args:
            num_epochs (int): 训练的轮次。
            batch_size (int): 每个批次的样本数。
            validate_every (int): 每多少轮进行一次验证。
            learning_rate (float): 学习率。
            accumulation_steps (int): 梯度累积步数。
        
        Returns:
            dict: 包含最终指标、最佳指标和训练历史的字典。
        """
        self.logger.info(f"开始训练，轮次: {num_epochs}, 批处理大小: {batch_size}, 学习率: {learning_rate}")
        
        # 设置学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # 创建数据加载器
        try:
            dataloader = self._create_dataloader(batch_size=batch_size)
        except Exception as e:
            self.logger.error(f"创建数据加载器时出错: {e}")
            return
        
        # 数据验证
        if not self._validate_data():
            self.logger.error("数据验证失败，训练终止。")
            return
        
        # 记录开始时间
        start_time = time.time()
        best_val_loss = float('inf')
        best_metrics = {}
        
        # 创建进度条
        pbar = tqdm(total=num_epochs, desc="训练进度")
        
        # 使用梯度累积的等效批处理大小
        effective_batch_size = batch_size * accumulation_steps
        if accumulation_steps > 1:
            self.logger.info(f"使用梯度累积，累积步数: {accumulation_steps}, 等效批处理大小: {effective_batch_size}")
        
        for epoch in range(num_epochs):
            # 设置为训练模式
            self.model.train()
            
            # 在每个epoch开始时重置损失和指标
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_samples = 0
            
            # 梯度累积相关变量
            running_loss = 0.0
            samples_in_batch = 0
            
            # 清空梯度
            self.optimizer.zero_grad()
            
            # 启用自动混合精度
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                for batch_idx, batch in enumerate(dataloader):
                    try:
                        # 将数据移到设备
                        inputs = self._to_device(batch)
                        
                        # 前向传播
                        outputs = self.model(**inputs)
                        loss = self._compute_loss(outputs, inputs['labels'])
                        
                        # 缩放损失并反向传播
                        scaled_loss = loss / accumulation_steps
                        self.scaler.scale(scaled_loss).backward()
                        
                        # 梯度累积
                        if (batch_idx + 1) % accumulation_steps == 0:
                            # 梯度裁剪
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            
                            # 优化器步进
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            
                            if self.scheduler is not None:
                                self.scheduler.step()
                        
                        # 记录指标
                        if self.wandb_logger:
                            self.wandb_logger.log_batch({
                                'loss': loss.item(),
                                'lr': self.optimizer.param_groups[0]['lr']
                            }, batch_idx, epoch)
                        
                        # 更新进度
                        epoch_loss += loss.item()
                        num_samples += len(batch[0])
                        samples_in_batch += len(batch[0])
                        
                        # 跟踪内存使用
                        self.memory_tracker.track()
                    except Exception as e:
                        self.logger.error(f"训练过程中出错: {e}")
                        continue
            
            # 计算平均损失和准确率
            epoch_loss /= num_samples
            epoch_acc /= num_samples
            
            # 添加到历史记录
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['epoch_losses'].append(epoch_loss)
            self.training_history['epoch_accuracies'].append(epoch_acc)
            
            # 打印进度
            log_message = f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
            
            # 验证
            val_metrics = {}
            if validate_every > 0 and (epoch + 1) % validate_every == 0:
                try:
                    val_metrics = self.validate()
                    val_loss = val_metrics['val_loss']
                    val_acc = val_metrics['val_accuracy']
                    
                    # 更新日志信息
                    log_message += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    
                    # 记录验证指标
                    self.training_history['val_losses'].append(val_loss)
                    self.training_history['val_accuracies'].append(val_acc)
                    
                    # 检查是否为最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_metrics = val_metrics.copy()
                        
                        # 保存最佳模型
                        self.save_checkpoint(name="best_model", include_optimizer=True)
                        self.logger.info(f"最佳模型已保存，验证损失: {val_loss:.4f}")
                except Exception as e:
                    self.logger.error(f"验证过程中出错: {e}")
                    continue
            
            # 更新日志
            self.logger.info(log_message)
            
            # 更新进度条
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            
            # 在epoch结束时保存检查点
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(name=f"epoch_{epoch+1}", include_optimizer=True)
            
            # 动态内存管理
            if self.save_memory:
                self.logger.info("启用内存优化策略。")
                # 这里可以添加具体的内存管理逻辑，例如使用更小的批处理大小或清理不必要的变量。
                self.batch_size = max(1, self.batch_size // 2)  # 示例：减小批处理大小
            
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            if self.early_stopping and self.early_stopping(val_loss):
                self.logger.info("早停触发，训练终止")
                break
        
        # 保存最终模型
        self.save_checkpoint(name="final_model", include_optimizer=True)
        
        # 关闭进度条
        pbar.close()
        
        # 计算总训练时间
        total_time = time.time() - start_time
        self.logger.info(f"训练完成，总时间: {formatTimeDelta(total_time)}")
        
        # 绘制训练历史
        self._plot_history()
        
        # 返回训练结果
        final_metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'val_loss': val_metrics.get('val_loss', None),
            'val_accuracy': val_metrics.get('val_accuracy', None),
            'training_time': total_time
        }
        
        return {
            'final_metrics': final_metrics,
            'best_metrics': best_metrics,
            'training_history': self.training_history
        }
    
    def _compute_loss(self, outputs, labels):
        """计算损失
        
        根据当前训练阶段计算不同的损失
        """
        loss = 0.0
        sequence_points_pred = outputs['sequence_points']
        sequence_points_true = labels.get('sequence_points')
        
        if sequence_points_true is not None:
            sequence_points_true = sequence_points_true.to(self.device)
            loss += self.weighted_sequence_loss(sequence_points_pred, sequence_points_true)
        
        if self.current_stage == 2:
            market_structure = outputs['market_structure']
            market_structure_true = labels.get('market_structure')
            
            if market_structure_true is not None:
                market_structure_true = market_structure_true.to(self.device)
                loss += self.weighted_price_loss(market_structure, market_structure_true)
        
        elif self.current_stage == 3:
            signal_pred = outputs['signal']
            prices_pred = outputs['prices']
            signal_true = labels.get('signal')
            prices_true = labels.get('prices')
            
            if signal_true is not None:
                signal_true = signal_true.to(self.device)
                loss += 0.3 * self.weighted_signal_loss(signal_pred, signal_true)
            
            if prices_true is not None:
                prices_true = prices_true.to(self.device)
                loss += 0.4 * self.weighted_price_loss(prices_pred, prices_true)
        
        return loss
    
    @enhanced_error_handler
    async def validate(self):
        """验证模型
        
        Returns:
            val_loss: 验证损失
            metrics: 验证指标
        """
        self.logger.info("开始验证...")
        self.model.eval()
        
        # 创建验证数据加载器
        val_dataloader = self._create_dataloader(batch_size=32, shuffle=False)
        
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for chart_data, labels in val_dataloader:
                # 将数据移动到设备
                chart_data = chart_data.to(self.device)
                
                # 前向传播
                outputs = self.model(chart_data)
                
                # 计算损失
                loss = self._compute_loss(outputs, labels)
                val_loss += loss.item()
                
                # 收集预测和标签
                all_preds.append(outputs)
                all_labels.append(labels)
        
        # 计算平均损失
        val_loss /= len(val_dataloader)
        
        # 计算指标
        metrics = self._compute_metrics(all_preds, all_labels)
        
        self.logger.info(f"验证完成，损失: {val_loss:.4f}")
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")
        
        return val_loss, metrics
    
    def _compute_metrics(self, all_preds, all_labels):
        """计算评估指标"""
        metrics = {}
        
        # 根据当前阶段计算不同的指标
        if self.current_stage == 1:
            # 阶段1: 序列识别准确率
            correct = 0
            total = 0
            
            for preds, labels in zip(all_preds, all_labels):
                sequence_points_pred = preds['sequence_points']
                sequence_points_true = labels.get('sequence_points')
                
                if sequence_points_true is not None:
                    sequence_points_true = sequence_points_true.to(self.device)
                    _, predicted = torch.max(sequence_points_pred, 1)
                    _, true = torch.max(sequence_points_true, 1)
                    
                    correct += (predicted == true).sum().item()
                    total += true.size(0)
            
            if total > 0:
                metrics['sequence_accuracy'] = correct / total
        
        elif self.current_stage == 2:
            # 阶段2: 市场结构分析指标
            # 这里可以添加更多特定于市场结构分析的指标
            pass
            
        else:
            # 阶段3: 交易信号生成指标
            signal_correct = 0
            signal_total = 0
            
            for preds, labels in zip(all_preds, all_labels):
                signal_pred = preds['signal']
                signal_true = labels.get('signal')
                
                if signal_true is not None:
                    signal_true = signal_true.to(self.device)
                    _, predicted = torch.max(signal_pred, 1)
                    _, true = torch.max(signal_true, 1)
                    
                    signal_correct += (predicted == true).sum().item()
                    signal_total += true.size(0)
            
            if signal_total > 0:
                metrics['signal_accuracy'] = signal_correct / signal_total
        
        return metrics
    
    def _finetune_with_pseudo_labels(self, batch_size: int = 16):
        """使用伪标签进行微调"""
        # 获取伪标签
        pseudo_labels = self.reward_mechanism.generate_pseudo_labels(confidence_threshold=0.8)
        
        if not pseudo_labels:
            self.logger.info("没有足够的伪标签用于微调")
            return
        
        self.logger.info(f"使用 {len(pseudo_labels)} 个伪标签进行微调")
        
        # 微调模型
        if len(pseudo_labels) < 5:
            self.logger.warning("伪标签数量不足，无法进行微调。")
            return
        
        self.model.train()
        
        # 将伪标签分成批次
        num_batches = (len(pseudo_labels) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(pseudo_labels))
            batch = pseudo_labels[start_idx:end_idx]
            
            # 准备数据
            chart_data = [item['chart_data'] for item in batch]
            labels = [item['labels'] for item in batch]
            
            # 转换为张量
            chart_data = torch.tensor(chart_data).to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(chart_data)
            
            # 计算损失
            loss = self._compute_loss(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            self.logger.info(f"伪标签微调批次 {batch_idx+1}/{num_batches}, 损失: {loss.item():.4f}")
    
    def fine_tune_with_rewards(self, signals: List[Dict[str, Any]]):
        """使用奖励值微调模型
        
        Args:
            signals: 信号列表，包含奖励值
        """
        self.model.train()  # 设置为训练模式
        for signal in signals:
            try:
                reward_value = signal.get('reward_value', 0.0)
                chart_data = torch.tensor(signal['chart_data']).to(self.device)
                labels = signal['labels']
                self.optimizer.zero_grad()
                outputs = self.model(chart_data)
                loss = self._compute_loss(outputs, labels) * reward_value  # 根据奖励值调整损失
                loss.backward()
                self.optimizer.step()
                self.logger.info(f"伪标签微调，损失: {loss.item():.4f}, 奖励值: {reward_value}")
            except torch.cuda.OutOfMemoryError as e:
                self.logger.error(f"内存不足: {e}", exc_info=True)
                torch.cuda.empty_cache()
            except ValueError as e:
                self.logger.error(f"数据错误: {e}", exc_info=True)
            except Exception as e:
                self.logger.error(f"微调模型时出错: {e}", exc_info=True)
    
    def evaluate(self):
        """评估模型
        
        Returns:
            metrics: 评估指标
        """
        # 验证模型
        _, metrics = self.validate()
        
        # 保存评估结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.save_dir / f"evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"评估结果已保存到 {results_file}")
        
        return metrics
    
    def predict(self, chart_data):
        """使用模型进行预测
        
        Args:
            chart_data: 图表数据
            
        Returns:
            predictions: 预测结果
        """
        self.model.eval()
        
        # 转换为张量
        if not isinstance(chart_data, torch.Tensor):
            chart_data = torch.tensor(chart_data).to(self.device)
        else:
            chart_data = chart_data.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(chart_data)
        
        return outputs
    
    def save_checkpoint(self, name: str = None, include_optimizer: bool = True, notify: bool = True):
        """保存检查点
        
        Args:
            name: 检查点名称
            include_optimizer: 是否包含优化器状态
            notify: 是否发送通知
        
        Returns:
            检查点路径
        """
        if name is None:
            name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_dir = self.save_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        
        # 准备检查点数据
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'current_stage': self.current_stage,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': {
                'training_time': self.total_training_time,
                'epochs_completed': len(self.history.get('train_loss', [])),
                'best_metrics': self.best_metrics if hasattr(self, 'best_metrics') else {},
                'model_config': self.model.config if hasattr(self.model, 'config') else {}
            }
        }
        
        # 添加优化器状态（可选）
        if include_optimizer and self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            # 如果有调度器，也保存它
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存到 {checkpoint_path}")
        
        # 创建检查点信息文件
        info_path = checkpoint_dir / f"{name}_info.json"
        checkpoint_info = {
            'path': str(checkpoint_path),
            'name': name,
            'timestamp': checkpoint['timestamp'],
            'epochs_completed': checkpoint['metadata']['epochs_completed'],
            'training_time': checkpoint['metadata']['training_time'],
            'metrics': {
                'train_loss': self.history.get('train_loss', [])[-1] if self.history.get('train_loss', []) else None,
                'val_loss': self.history.get('val_loss', [])[-1] if self.history.get('val_loss', []) else None,
                'accuracy': self.history.get('accuracy', [])[-1] if self.history.get('accuracy', []) else None
            },
            'stage': self.current_stage
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_info, f, ensure_ascii=False, indent=4)
        
        # 更新检查点列表
        self._update_checkpoint_list(checkpoint_info)
        
        # 发送通知
        if notify and hasattr(self, 'notifier') and self.notifier is not None:
            try:
                self._send_checkpoint_notification(checkpoint_info)
            except Exception as e:
                self.logger.error(f"发送检查点通知时出错: {e}")
        
        # 清理旧检查点
        self._clean_old_checkpoints()
        
        return str(checkpoint_path)
    
    def _update_checkpoint_list(self, checkpoint_info: Dict):
        """更新检查点列表
        
        Args:
            checkpoint_info: 检查点信息
        """
        checkpoint_dir = self.save_dir / 'checkpoints'
        list_path = checkpoint_dir / 'checkpoint_list.json'
        
        # 读取现有列表
        checkpoint_list = []
        if list_path.exists():
            try:
                with open(list_path, 'r', encoding='utf-8') as f:
                    checkpoint_list = json.load(f)
            except Exception as e:
                self.logger.warning(f"读取检查点列表时出错: {e}")
        
        # 添加新检查点
        checkpoint_list.append(checkpoint_info)
        
        # 保存更新后的列表
        with open(list_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_list, f, ensure_ascii=False, indent=4)
    
    def _send_checkpoint_notification(self, checkpoint_info: Dict):
        """发送检查点通知
        
        Args:
            checkpoint_info: 检查点信息
        """
        if not hasattr(self, 'notifier') or self.notifier is None:
            return
        
        # 创建进度图表
        charts = self._create_progress_charts()
        
        # 构建通知消息
        message = f"**检查点已保存**\n"
        message += f"**名称:** {checkpoint_info['name']}\n"
        message += f"**时间:** {checkpoint_info['timestamp']}\n"
        message += f"**已完成轮次:** {checkpoint_info['epochs_completed']}\n"
        message += f"**训练时间:** {formatTimeDelta(checkpoint_info['training_time'])}\n"
        message += f"**当前阶段:** {checkpoint_info['stage']}\n\n"
        
        # 添加指标信息
        message += "**当前指标:**\n"
        for metric, value in checkpoint_info['metrics'].items():
            if value is not None:
                message += f"- {metric}: {value:.4f}\n"
        
        # 发送通知
        self.notifier.send_training_progress(
            message=message,
            charts=charts,
            checkpoint_path=checkpoint_info['path']
        )
    
    def _create_progress_charts(self) -> List[str]:
        """创建训练进度图表
        
        Returns:
            图表文件路径列表
        """
        charts = []
        plot_dir = os.path.join(self.save_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # 创建损失曲线图
        loss_plot_path = os.path.join(plot_dir, f"loss_plot_{int(time.time())}.png")
        plt.figure(figsize=(10, 6))
        
        # 绘制训练损失
        if self.training_history['epoch_losses']:
            plt.plot(self.training_history['epochs'], self.training_history['epoch_losses'], 'b-', label='训练损失')
        
        # 绘制验证损失
        if self.training_history['val_losses']:
            # 确保x坐标正确
            val_epochs = [e for i, e in enumerate(self.training_history['epochs']) if i % self.validate_every == 0]
            if len(val_epochs) > len(self.training_history['val_losses']):
                val_epochs = val_epochs[:len(self.training_history['val_losses'])]
            plt.plot(val_epochs, self.training_history['val_losses'], 'r-', label='验证损失')
        
        plt.title('训练和验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.savefig(loss_plot_path)
        plt.close()
        charts.append(loss_plot_path)
        
        # 创建准确率曲线图
        acc_plot_path = os.path.join(plot_dir, f"accuracy_plot_{int(time.time())}.png")
        plt.figure(figsize=(10, 6))
        
        # 绘制训练准确率
        if self.training_history['epoch_accuracies']:
            plt.plot(self.training_history['epochs'], self.training_history['epoch_accuracies'], 'b-', label='训练准确率')
        
        # 绘制验证准确率
        if self.training_history['val_accuracies']:
            # 确保x坐标正确
            val_epochs = [e for i, e in enumerate(self.training_history['epochs']) if i % self.validate_every == 0]
            if len(val_epochs) > len(self.training_history['val_accuracies']):
                val_epochs = val_epochs[:len(self.training_history['val_accuracies'])]
            plt.plot(val_epochs, self.training_history['val_accuracies'], 'r-', label='验证准确率')
        
        plt.title('训练和验证准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True)
        plt.savefig(acc_plot_path)
        plt.close()
        charts.append(acc_plot_path)
        
        # 创建学习率曲线图
        if self.training_history['learning_rates']:
            lr_plot_path = os.path.join(plot_dir, f"lr_plot_{int(time.time())}.png")
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.training_history['learning_rates'])+1), self.training_history['learning_rates'], 'g-')
            plt.title('学习率变化')
            plt.xlabel('更新次数')
            plt.ylabel('学习率')
            plt.yscale('log')  # 使用对数刻度
            plt.grid(True)
            plt.savefig(lr_plot_path)
            plt.close()
            charts.append(lr_plot_path)
            
        # 添加内存使用图表
        memory_plot_path = self.memory_tracker.plot_memory_usage(
            save_path=os.path.join(plot_dir, f'memory_usage_{int(time.time())}.png')
        )
        if memory_plot_path:
            charts.append(memory_plot_path)
            
        return charts
    
    def save_model(self, name: str = None, include_metadata: bool = True):
        """保存模型
        
        Args:
            name: 模型名称
            include_metadata: 是否包含元数据
            
        Returns:
            模型保存路径
        """
        if name is None:
            name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        models_dir = self.save_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / f"{name}.pt"
        
        if include_metadata:
            # 保存模型和元数据
            metadata = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'training_time': self.total_training_time,
                'epochs_completed': len(self.history.get('train_loss', [])),
                'metrics': {
                    'train_loss': self.history.get('train_loss', [])[-1] if self.history.get('train_loss', []) else None,
                    'val_loss': self.history.get('val_loss', [])[-1] if self.history.get('val_loss', []) else None,
                    'accuracy': self.history.get('accuracy', [])[-1] if self.history.get('accuracy', []) else None
                },
                'stage': self.current_stage,
                'model_config': self.model.config if hasattr(self.model, 'config') else {}
            }
            
            save_data = {
                'model_state_dict': self.model.state_dict(),
                'metadata': metadata
            }
            
            torch.save(save_data, model_path)
        else:
            # 只保存模型权重
            torch.save(self.model.state_dict(), model_path)
        
        self.logger.info(f"模型已保存到 {model_path}")
        
        # 创建模型信息文件
        info_path = models_dir / f"{name}_info.json"
        model_info = {
            'path': str(model_path),
            'name': name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epochs_completed': len(self.history.get('train_loss', [])) if hasattr(self, 'history') else 0,
            'metrics': {
                'train_loss': self.history.get('train_loss', [])[-1] if self.history.get('train_loss', []) else None,
                'val_loss': self.history.get('val_loss', [])[-1] if self.history.get('val_loss', []) else None,
                'accuracy': self.history.get('accuracy', [])[-1] if self.history.get('accuracy', []) else None
            } if hasattr(self, 'history') else {},
            'stage': self.current_stage
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)
        
        return str(model_path)
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True, strict: bool = True):
        """加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            load_optimizer: 是否加载优化器状态
            strict: 是否严格加载模型权重
            
        Returns:
            加载的检查点数据
        """
        self.logger.info(f"正在从 {checkpoint_path} 加载检查点")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型状态
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            else:
                self.logger.warning("检查点中没有模型状态字典")
            
            # 加载优化器状态 (如果有)
            if load_optimizer and 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    self.logger.warning(f"加载优化器状态时出错: {e}")
            
            # 加载调度器状态 (如果有)
            if load_optimizer and 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler') and self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    self.logger.warning(f"加载调度器状态时出错: {e}")
            
            # 加载历史记录
            if 'history' in checkpoint:
                self.history = checkpoint['history']
            
            # 加载当前阶段
            if 'current_stage' in checkpoint:
                self.current_stage = checkpoint['current_stage']
            
            # 加载元数据 (如果有)
            if 'metadata' in checkpoint:
                if 'training_time' in checkpoint['metadata']:
                    self.total_training_time = checkpoint['metadata']['training_time']
                
                if 'best_metrics' in checkpoint['metadata']:
                    self.best_metrics = checkpoint['metadata']['best_metrics']
            
            self.logger.info(f"检查点加载成功，已完成轮次: {len(self.history.get('train_loss', []))}")
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"加载检查点时出错: {e}")
            raise
    
    def load_model(self, model_path: str, strict: bool = True):
        """加载模型
        
        Args:
            model_path: 模型路径
            strict: 是否严格加载模型权重
            
        Returns:
            模型元数据 (如果有)
        """
        self.logger.info(f"正在从 {model_path} 加载模型")
        
        try:
            # 尝试加载模型和元数据
            loaded_data = torch.load(model_path, map_location=self.device)
            
            metadata = None
            
            # 检查是否包含元数据
            if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                # 加载模型状态
                self.model.load_state_dict(loaded_data['model_state_dict'], strict=strict)
                
                # 提取元数据 (如果有)
                if 'metadata' in loaded_data:
                    metadata = loaded_data['metadata']
            else:
                # 直接加载为模型状态字典
                self.model.load_state_dict(loaded_data, strict=strict)
            
            self.logger.info(f"模型加载成功")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"加载模型时出错: {e}")
            raise
    
    def list_checkpoints(self) -> List[Dict]:
        """列出所有可用的检查点
        
        Returns:
            检查点信息列表
        """
        checkpoint_dir = self.save_dir / 'checkpoints'
        
        if not checkpoint_dir.exists():
            return []
        
        # 检查是否有检查点列表文件
        list_path = checkpoint_dir / 'checkpoint_list.json'
        if list_path.exists():
            try:
                with open(list_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"读取检查点列表时出错: {e}")
        
        # 如果没有列表文件或读取失败，搜索目录
        checkpoint_files = list(checkpoint_dir.glob('*.pt'))
        checkpoint_list = []
        
        for cp_file in checkpoint_files:
            # 检查是否有对应的信息文件
            info_file = checkpoint_dir / f"{cp_file.stem}_info.json"
            
            if info_file.exists():
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        checkpoint_info = json.load(f)
                        checkpoint_list.append(checkpoint_info)
                except Exception as e:
                    self.logger.warning(f"读取检查点信息文件 {info_file} 时出错: {e}")
            else:
                # 创建基本信息
                checkpoint_list.append({
                    'path': str(cp_file),
                    'name': cp_file.stem,
                    'timestamp': datetime.fromtimestamp(cp_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return checkpoint_list
    
    def export_model(self, export_path: str, format: str = 'pt', include_config: bool = True):
        """导出模型为不同格式
        
        Args:
            export_path: 导出路径
            format: 导出格式 ('pt', 'onnx')
            include_config: 是否包含配置
            
        Returns:
            导出的模型路径
        """
        self.logger.info(f"正在导出模型为 {format} 格式")
        
        # 确保目录存在
        export_dir = os.path.dirname(export_path)
        os.makedirs(export_dir, exist_ok=True)
        
        if format.lower() == 'pt':
            # 导出为PyTorch格式
            if include_config and hasattr(self.model, 'config'):
                save_data = {
                    'model_state_dict': self.model.state_dict(),
                    'config': self.model.config
                }
                torch.save(save_data, export_path)
            else:
                torch.save(self.model.state_dict(), export_path)
                
        elif format.lower() == 'onnx':
            # 导出为ONNX格式
            try:
                # 获取输入样本
                dummy_input = self._get_dummy_input()
                
                # 导出模型
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    export_path,
                    verbose=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
            except Exception as e:
                self.logger.error(f"导出ONNX模型时出错: {e}")
                raise
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        self.logger.info(f"模型已导出到 {export_path}")
        
        return export_path
    
    def _get_dummy_input(self):
        """获取模型输入示例，用于ONNX导出
        
        Returns:
            张量或张量元组
        """
        # 如果模型有一个方法来提供输入示例，使用它
        if hasattr(self.model, 'get_dummy_input'):
            return self.model.get_dummy_input()
        
        # 否则，创建一个简单的输入张量
        # 注意: 这需要根据实际模型输入要求进行调整
        return torch.randn(1, 3, 224, 224, device=self.device)

    def _plot_history(self):
        """绘制训练历史"""
        history_dir = self.save_dir / 'history'
        history_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 绘制损失
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.savefig(history_dir / f"loss_history_{timestamp}.png")
        plt.close()
        
        # 绘制指标
        for metric_name, values in self.history['metrics'].items():
            plt.figure(figsize=(10, 6))
            plt.plot(values)
            plt.title(f'{metric_name} 历史')
            plt.xlabel('验证轮次')
            plt.ylabel(metric_name)
            plt.grid(True)
            plt.savefig(history_dir / f"{metric_name}_history_{timestamp}.png")
            plt.close()
        
        # 保存历史数据
        with open(history_dir / f"training_history_{timestamp}.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"训练历史已保存到 {history_dir}")
    
    def set_stage(self, stage: int):
        """设置训练阶段
        
        Args:
            stage: 训练阶段 (1-3)
        """
        if stage not in [1, 2, 3]:
            raise ValueError("训练阶段必须是 1, 2 或 3")
        
        self.current_stage = stage
        
        # 设置模型阶段
        if stage == 1:
            self.model.train_stage1()
        elif stage == 2:
            self.model.train_stage2()
        else:
            self.model.train_stage3()
        
        self.logger.info(f"设置训练阶段为 {stage}")
    
    def record_signal(self, chart_data, prediction, confidence):
        """记录交易信号
        
        Args:
            chart_data: 图表数据
            prediction: 预测结果
            confidence: 置信度
            
        Returns:
            signal_id: 信号ID
        """
        return self.signal_tracker.record_signal(chart_data, prediction, confidence)
    
    def update_signal(self, signal_id, price_data):
        """更新信号跟踪
        
        Args:
            signal_id: 信号ID
            price_data: 价格数据
        """
        self.signal_tracker.update_signal_tracking(signal_id, price_data)
    
    def get_signal_stats(self):
        """获取信号统计数据"""
        return self.signal_tracker.get_stats()

    def enable_memory_optimization(self, save_memory: bool = True):
        """启用内存优化
        
        Args:
            save_memory: 是否启用内存优化
        """
        self.save_memory = save_memory
        self.logger.info(f"内存优化模式: {'启用' if save_memory else '禁用'}")
        
        if save_memory:
            # 启用梯度检查点
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                self.use_gradient_checkpointing = True
            
            # 启用内存高效的注意力
            if hasattr(self.model, 'enable_memory_efficient_attention'):
                self.model.enable_memory_efficient_attention()
            
            # 启用Flash Attention
            if hasattr(self.model, 'enable_flash_attention'):
                self.model.enable_flash_attention()
        else:
            # 禁用所有优化
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
            self.use_gradient_checkpointing = False
    
    def setup_incremental_loader(self, data_path: str, batch_size: int = 1000, 
                            window_size: int = 100, stride: int = 20):
        """设置增量数据加载器
        
        Args:
            data_path: 数据文件路径
            batch_size: 每次加载的行数
            window_size: 图表窗口大小
            stride: 窗口滑动步长
        """
        self.incremental_loader = DataIncrementalLoader(
            data_path=data_path,
            batch_size=batch_size,
            window_size=window_size,
            stride=stride
        )
        self.logger.info(f"已设置增量数据加载器，批处理大小: {batch_size}")
    
    def process_data_incrementally(self, processor_func, progress_callback=None):
        """增量处理数据
        
        Args:
            processor_func: 处理函数
            progress_callback: 进度回调函数
            
        Returns:
            处理结果列表
        """
        if not self.incremental_loader:
            self.logger.error("未设置增量数据加载器，无法进行增量处理")
            return []
        
        self.logger.info("开始增量处理数据...")
        results = self.incremental_loader.process_all_data(processor_func, progress_callback)
        self.logger.info(f"增量处理完成，生成 {len(results)} 个结果")
        
        # 清理资源
        self.incremental_loader.stop()
        
        # 返回处理结果
        return results

    def _to_device(self, data):
        """递归检查和移动所有张量到正确设备"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._to_device(item) for item in data]
        return data

    def _validate_data(self):
        """验证数据完整性和格式
        
        检查数据目录是否存在，数据格式是否正确，必要的列是否存在等。
        """
        if not self.data_dir.exists():
            raise ValueError(f"数据目录不存在: {self.data_dir}")
        
        # 检查数据文件
        data_files = list(self.data_dir.glob('*.json'))
        if not data_files:
            raise ValueError(f"数据目录中没有找到JSON文件: {self.data_dir}")
        
        # 验证数据格式
        sample_file = data_files[0]
        try:
            with open(sample_file, 'r') as f:
                sample_data = json.load(f)
                
            # 检查必要字段
            required_fields = ['chart_data', 'labels']
            for field in required_fields:
                if field not in sample_data:
                    raise ValueError(f"数据缺少必要字段: {field}")
                
            # 检查图表数据格式
            if 'chart_data' in sample_data:
                chart_data = sample_data['chart_data']
                required_columns = ['date', 'price', 'volume']
                for col in required_columns:
                    if col not in chart_data:
                        raise ValueError(f"图表数据缺少必要列: {col}")
                    
            # 检查标签格式
            if 'labels' in sample_data:
                labels = sample_data['labels']
                if not isinstance(labels, dict):
                    raise ValueError("标签必须是字典格式")
                
            self.logger.info("数据验证通过")
            
        except json.JSONDecodeError:
            raise ValueError(f"JSON文件格式错误: {sample_file}")
        except Exception as e:
            raise ValueError(f"数据验证失败: {str(e)}")

    def _clean_old_checkpoints(self, max_to_keep=5):
        """清理旧检查点，保持最新的检查点数量
        
        Args:
            max_to_keep: 保留的最大检查点数量
        """
        checkpoints = sorted(self.list_checkpoints(), key=lambda x: x['timestamp'])
        if len(checkpoints) > max_to_keep:
            for cp in checkpoints[:-max_to_keep]:
                os.remove(cp['path'])
                os.remove(f"{cp['path'][:-3]}_info.json")
                self.logger.info(f"已删除旧检查点: {cp['path']}")

    def auto_save_state(self):
        """自动保存训练状态"""
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_metric': self.best_metric
        }
        torch.save(state, f"{self.checkpoint_path}/latest.pt")
        self.logger.info("训练状态已自动保存")

    def resume_training(self, checkpoint_path: str = None):
        """从检查点恢复训练"""
        if checkpoint_path is None:
            checkpoint_path = f"{self.checkpoint_path}/latest.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.current_epoch = checkpoint['epoch']
            self.best_metric = checkpoint['best_metric']
            self.logger.info(f"从检查点恢复训练，当前轮次: {self.current_epoch}")
        else:
            self.logger.warning("检查点文件不存在，无法恢复训练")

    def get_gpu_utilization(self) -> float:
        """获取GPU使用率"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
            return 0.0
        except Exception as e:
            self.logger.warning(f"获取GPU使用率失败: {e}")
            return 0.0

    def calculate_batch_speed(self) -> float:
        """计算批处理速度(样本/秒)"""
        if not hasattr(self, 'batch_times'):
            return 0.0
        if len(self.batch_times) < 2:
            return 0.0
        avg_time = sum(self.batch_times[-10:]) / len(self.batch_times[-10:])
        return self.batch_size / avg_time if avg_time > 0 else 0

    def get_stage_description(self, stage: int) -> str:
        """获取训练阶段描述"""
        stage_descriptions = {
            1: "初始训练阶段 - 基础特征学习",
            2: "微调阶段 - 市场特征适应",
            3: "强化学习阶段 - 策略优化",
            4: "验证阶段 - 性能评估"
        }
        return stage_descriptions.get(stage, "未知阶段")

    def send_stage_metrics(self):
        """发送阶段性指标"""
        try:
            metrics = {
                "gpu_utilization": self.get_gpu_utilization(),
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "batch_processing_speed": self.calculate_batch_speed(),
                "memory_efficiency": self.memory_tracker.get_memory_stats()['efficiency']
            }
            
            # 添加当前阶段信息
            metrics["current_stage"] = self.current_stage
            metrics["stage_description"] = self.get_stage_description(self.current_stage)
            
            # 添加训练进度
            if hasattr(self, 'total_epochs'):
                current_epoch = len(self.history.get('train_loss', []))
                metrics["training_progress"] = f"{current_epoch}/{self.total_epochs}"
                metrics["progress_percentage"] = (current_epoch / self.total_epochs) * 100
            
            if hasattr(self, 'notifier'):
                self.notifier.send_monitor_message(metrics)
                self.logger.info(f"已发送阶段性指标: {metrics}")
        except Exception as e:
            self.logger.error(f"发送阶段性指标失败: {e}")

    def auto_resume_training(self):
        """自动恢复训练"""
        try:
            checkpoints = self.list_checkpoints()
            if not checkpoints:
                self.logger.info("未找到可用的检查点")
                return False
                
            # 获取最新的检查点
            latest_checkpoint = checkpoints[-1]
            self.logger.info(f"找到最新检查点: {latest_checkpoint['name']}")
            
            # 加载检查点
            self.load_checkpoint(latest_checkpoint['path'])
            
            # 发送恢复通知
            if hasattr(self, 'notifier'):
                resume_message = (
                    f"🔄 **自动恢复训练**\n\n"
                    f"**检查点信息:**\n"
                    f"- 名称: {latest_checkpoint['name']}\n"
                    f"- 时间: {latest_checkpoint['timestamp']}\n"
                    f"- 已完成轮次: {latest_checkpoint['epochs_completed']}\n"
                    f"- 阶段: {self.get_stage_description(latest_checkpoint.get('stage', 1))}"
                )
                self.notifier.send_message_sync(resume_message)
            
            return True
        except Exception as e:
            self.logger.error(f"自动恢复训练失败: {e}")
            return False

    def notify_stage_change(self, stage: int):
        """通知阶段变更"""
        try:
            # 更新当前阶段
            self.current_stage = stage
            
            # 准备阶段变更消息
            message = (
                f"📊 **训练阶段变更**\n\n"
                f"**当前阶段:** 阶段 {stage}\n"
                f"**阶段描述:** {self.get_stage_description(stage)}\n\n"
                f"**训练状态:**\n"
                f"- 已完成轮次: {len(self.history.get('train_loss', []))}\n"
                f"- 当前学习率: {self.optimizer.param_groups[0]['lr']:.6f}\n"
                f"- GPU使用率: {self.get_gpu_utilization():.1f}%\n"
                f"- 批处理速度: {self.calculate_batch_speed():.1f} 样本/秒\n"
            )
            
            # 添加最近的指标
            if self.history.get('train_loss'):
                message += f"- 训练损失: {self.history['train_loss'][-1]:.4f}\n"
            if self.history.get('val_loss'):
                message += f"- 验证损失: {self.history['val_loss'][-1]:.4f}\n"
                
            # 添加内存使用信息
            memory_stats = self.memory_tracker.get_memory_stats()
            message += (
                f"\n**内存状态:**\n"
                f"- 当前使用: {memory_stats['current_usage']:.1f}GB\n"
                f"- 峰值使用: {memory_stats['peak_usage']:.1f}GB\n"
                f"- 内存效率: {memory_stats['efficiency']:.1f}%"
            )
            
            # 发送通知
            if hasattr(self, 'notifier'):
                self.notifier.send_message_sync(message)
                self.logger.info(f"已发送阶段变更通知: 阶段 {stage}")
        except Exception as e:
            self.logger.error(f"发送阶段变更通知失败: {e}")

    def track_gpu(self):
        """跟踪GPU使用情况"""
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # 转换为MB
            self.logger.info(f"GPU内存使用: {gpu_memory_allocated:.2f} MB")
            self.gpu_stats.append(gpu_memory_allocated)

    def get_gpu_stats(self) -> Dict:
        """获取GPU使用统计信息"""
        return {
            'avg_gpu_memory_mb': np.mean(self.gpu_stats) if self.gpu_stats else 0,
            'max_gpu_memory_mb': max(self.gpu_stats) if self.gpu_stats else 0,
            'current_gpu_memory_mb': torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        }


# 时间格式化工具函数
def formatTimeDelta(seconds: float) -> str:
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

# 异步测试
async def test_async_notification():
    # 创建训练器实例
    trainer = SelfSupervisedTrainer(...)
    await trainer.notifier.send_monitor_message({"status": "training", "progress": "50%"})

# 同步测试
def test_sync_notification():
    # 创建训练器实例
    trainer = SelfSupervisedTrainer(...)
    trainer.notifier.send_message_sync("训练开始", webhook_type='monitor') 