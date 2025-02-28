import os
import logging
import torch
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import shutil
import sys
import matplotlib.pyplot as plt

from ..model.llava_model_wrapper import LLaVAModelWrapper
from ..utils.signal_tracker import SignalTracker
from ..utils.reward_mechanism import RewardMechanism
from ..utils.progress_notifier import get_notifier, ProgressNotifier
from ..utils.validation_set_creator import ValidationSetCreator
from .self_supervised_trainer import SelfSupervisedTrainer
from ..utils.discord_notifier import DiscordNotifier
from ..data.chart_generator import ChartGenerator


class SelfSupervisedManager:
    """
    自监督学习管理器
    
    协调自监督学习的整个流程，包括数据生成、模型预测、信号跟踪和奖励机制
    """
    
    def __init__(self, csv_path: str, model_path: str, output_dir: str,
             window_size: int = 100, stride: int = 20, batch_size: int = 16,
             effective_batch_size: int = 64, epochs: int = 20, learning_rate: float = 0.001,
             validate_every: int = 1, save_every: int = 2, use_lr_scheduler: bool = False,
             scheduler_type: str = 'plateau', 
             scheduler_t_max: int = 10,
             scheduler_eta_min: float = 1e-6,  # 添加默认值
             notifier: ProgressNotifier = None, reward_config: Dict = None,
             use_early_stopping: bool = False,
             early_stopping_patience: int = 10,
             early_stopping_monitor: str = 'val_loss',
             checkpoint_dir: str = None,
             save_memory: bool = False,
             incremental_processing: bool = False,
             incremental_batch_size: int = 1000,
             gc_interval: int = 500,
             discord_webhook: str = None):
        """初始化自监督学习管理器

        Args:
            csv_path: CSV数据路径
            model_path: 模型路径
            output_dir: 输出目录
            window_size: 图表窗口大小
            stride: 图表滑动步长
            batch_size: 批处理大小
            effective_batch_size: 有效批处理大小
            epochs: 训练轮次
            learning_rate: 学习率
            validate_every: 验证间隔
            save_every: 保存间隔
            use_lr_scheduler: 是否使用学习率调度器
            notifier: 进度通知器
            reward_config: 奖励机制配置
            use_early_stopping: 是否使用早停策略
            early_stopping_patience: 早停策略的耐心值
            early_stopping_monitor: 早停策略的监控指标
            checkpoint_dir: 检查点目录
            save_memory: 是否启用内存优化
            incremental_processing: 是否启用增量处理
            incremental_batch_size: 增量批处理大小
            gc_interval: 垃圾回收间隔
            discord_webhook: Discord Webhook URL
        """
        # 初始化代码
        self.use_lr_scheduler = use_lr_scheduler
        self.scheduler_type = scheduler_type
        self.scheduler_t_max = scheduler_t_max 
        self.scheduler_eta_min = scheduler_eta_min
        # ... 其他初始化代码 ...
        # 设置基本路径和参数
        self.csv_path = csv_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.effective_batch_size = effective_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.validate_every = validate_every
        self.save_every = save_every
        self.use_lr_scheduler = use_lr_scheduler
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.checkpoint_dir = checkpoint_dir
        self.save_memory = save_memory
        self.incremental_processing = incremental_processing
        self.incremental_batch_size = incremental_batch_size
        self.gc_interval = gc_interval
        
        # 设置日志记录器
        self.logger = logging.getLogger("self_supervised_manager")
        
        # 进度通知器
        self.notifier = notifier
        
        # 奖励机制配置
        self.reward_config = reward_config or {}
        
        # 创建必要的目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化状态
        self.charts_processed = 0
        self.sbs_sequences_found = 0
        self.training_metrics = {}
        self.best_metrics = {}
        
        # 增量处理和内存优化设置
        self.use_incremental_processing = False
        self.incremental_batch_size = 1000  # 默认增量批处理大小
        
        # 初始化组件
        self.logger.info("初始化自监督学习组件")
        
        # 初始化数据处理器
        # self.chart_processor = CSVToChartProcessor(
        #     self.csv_path, 
        #     {
        #         'output_dir': self.output_dir,
        #         'window_size': self.window_size
        #     }
        # )
        
        # 初始化信号跟踪器
        self.signal_tracker = SignalTracker(
            save_dir=self.output_dir,
            tracking_window=5
        )
        
        # 信号跟踪配置
        self.signal_tracker_config = {
            'profit_target': 0.01,
            'stop_loss': 0.005,
            'max_holding_periods': 20
        }
        
        # 初始化奖励机制
        self.reward_mechanism = RewardMechanism(
            signal_tracker=self.signal_tracker,
            config={
                'min_confidence': 0.5,
                'profit_factor': 2.0,
                'loss_factor': 1.0
            }
        )
        
        # 初始化进度通知器
        if self.notifier:
            self.progress_notifier = get_notifier(
                webhook_url=self.notifier.webhook_url,
                interval_minutes=self.notifier.interval_minutes,
                save_dir='logs/notifications'
            )
        else:
            self.progress_notifier = None
            
        # 初始化验证集创建器
        self.validation_creator = ValidationSetCreator(
            base_data_path=self.csv_path,
            output_dir=self.output_dir
        )
        
        # 初始化LLaVA模型
        self.model = None  # 延迟加载模型
        
        # 处理进度
        self.processing_stats = {
            'start_time': time.time(),
            'total_charts': 0,
            'processed_charts': 0,
            'sbs_sequences': 0
        }
        
        # 在 __init__ 方法中添加 DiscordNotifier 的初始化
        self.discord_notifier = DiscordNotifier(os.getenv('DISCORD_SIGNAL_WEBHOOK'), self.logger)
        self.discord_webhook = discord_webhook
    
    def load_model(self):
        """
        加载LLaVA模型
        """
        if self.model is None:
            self.logger.info("加载LLaVA模型")
            self.model = LLaVAModelWrapper(
                model_path=self.model_path,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
    
    def generate_charts(self) -> List[str]:
        """生成图表
        
        从CSV数据生成图表
        
        Returns:
            生成的图表路径列表
        """
        self.logger.info(f"从CSV生成图表，窗口大小: {self.window_size}，步长: {self.stride}")
        
        # 创建图表处理器
        # processor = CSVToChartProcessor(
        #     output_dir=self.output_dir,
        #     window_size=self.window_size
        # )
        
        chart_paths = []
        
        # 增量处理模式
        if self.use_incremental_processing:
            self.logger.info(f"使用增量处理模式，批处理大小: {self.incremental_batch_size}")
            
            # 创建自监督训练器
            trainer = SelfSupervisedTrainer(
                model=None,  # 不需要模型
                data_dir=self.output_dir,
                save_dir="models/self_supervised"
            )
            
            # 设置增量加载器
            trainer.setup_incremental_loader(
                data_path=self.csv_path,
                batch_size=self.incremental_batch_size,
                window_size=self.window_size,
                stride=self.stride
            )
            
            # 启用内存优化
            if self.save_memory:
                trainer.enable_memory_optimization(True)
            
            # 定义窗口处理函数
            def process_window(window_df):
                try:
                    chart_path = processor.create_chart_from_dataframe(window_df)
                    if chart_path:
                        self.charts_processed += 1
                        # 每处理100个图表，发送一次进度通知
                        if self.progress_notifier and self.charts_processed % 100 == 0:
                            self._send_processing_progress("图表生成")
                        return chart_path
                except Exception as e:
                    self.logger.error(f"处理窗口时出错: {e}")
                return None
            
            # 增量处理所有数据
            chart_paths = trainer.process_data_incrementally(process_window, self._update_progress)
            
        else:
            # 传统处理模式
            self.logger.info("使用传统处理模式")
            chart_paths = processor.create_charts_from_csv(
                csv_path=self.csv_path, 
                stride=self.stride,
                callback=self._update_progress
            )
            self.charts_processed = len(chart_paths)
        
        self.logger.info(f"已生成 {len(chart_paths)} 个图表")
        
        return chart_paths
    
    def process_charts(self, chart_paths: List[str]) -> List[str]:
        """处理图表
        
        使用模型处理图表，识别出SBS序列
        
        Args:
            chart_paths: 图表路径列表
            
        Returns:
            SBS序列路径列表
        """
        from src.self_supervised.model.llava_processor import LLaVAProcessor
        
        # 初始化处理器
        processor = LLaVAProcessor(
            model_path=self.model_path
        )
        
        self.logger.info(f"使用LLaVA模型处理图表，模型路径: {self.model_path}")
        
        sequences = []
        processed_count = 0
        
        # 批处理图表
        if self.save_memory:
            # 启用内存优化，使用小批量处理
            batch_size = min(self.batch_size, 8)  # 使用更小的批量
            self.logger.info(f"启用内存优化，使用批处理大小: {batch_size}")
            
            # 分批处理
            for i in range(0, len(chart_paths), batch_size):
                batch = chart_paths[i:i+batch_size]
                
                # 处理批次
                batch_sequences = processor.process_charts(batch)
                sequences.extend(batch_sequences)
                
                processed_count += len(batch)
                self._update_progress(processed_count, len(chart_paths))
                
                # 主动进行垃圾回收
                if processed_count % 100 == 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 发送进度通知
                if self.progress_notifier and processed_count % 50 == 0:
                    self._send_processing_progress("图表处理")
        else:
            # 正常批处理
            for i in range(0, len(chart_paths), self.batch_size):
                batch = chart_paths[i:i+self.batch_size]
                
                # 处理批次
                batch_sequences = processor.process_charts(batch)
                sequences.extend(batch_sequences)
                
                processed_count += len(batch)
                self._update_progress(processed_count, len(chart_paths))
                
                # 发送进度通知
                if self.progress_notifier and processed_count % 100 == 0:
                    self._send_processing_progress("图表处理")
        
        # 过滤掉空序列
        valid_sequences = [seq for seq in sequences if seq]
        self.sbs_sequences_found = len(valid_sequences)
        
        self.logger.info(f"已处理 {processed_count} 个图表，找到 {len(valid_sequences)} 个SBS序列")
        
        return valid_sequences
    
    def _split_list(self, lst, batch_size):
        """
        将列表分割成批次
        
        Args:
            lst: 要分割的列表
            batch_size: 批次大小
            
        Returns:
            批次列表的生成器
        """
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]
    
    def prepare_training_data(self) -> Tuple[List[Dict], List[float]]:
        """
        准备训练数据
        
        Returns:
            包含训练样本和权重的元组
        """
        # 获取训练数据
        return self.reward_mechanism.get_training_data()
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """
        获取信号统计数据
        
        Returns:
            信号统计数据字典
        """
        # 从信号跟踪器获取信号统计
        return self.signal_tracker.get_stats()
    
    def save_sbs_sequences(self, paths: List[str]):
        """
        保存SBS序列图片，用于未来的微调
        
        Args:
            paths: SBS序列图片路径列表
        """
        if not paths:
            self.logger.warning("没有SBS序列图片需要保存")
            return
            
        self.logger.info(f"保存 {len(paths)} 个SBS序列图片用于未来微调")
        
        # 确保目标目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 获取信号统计
        signal_stats = self.get_signal_stats()
        
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 为每个序列创建单独的目录
        sbs_sequences_saved = 0
        for path in paths:
            # 获取文件名和基本名
            filename = os.path.basename(path)
            basename = os.path.splitext(filename)[0]
            
            # 创建序列目录
            seq_dir = Path(self.output_dir) / f"seq_{timestamp}_{basename}"
            seq_dir.mkdir(exist_ok=True)
            
            # 复制文件到序列目录
            try:
                # 目标路径
                target_path = seq_dir / filename
                
                # 复制文件
                shutil.copy(path, target_path)
                
                # 获取该图表的信号信息
                signal_info = self.signal_tracker.get_signal_for_chart(path)
                
                # 如果有信号信息，创建信息文件
                if signal_info:
                    # 添加额外信息
                    info = {
                        'chart_path': path,
                        'created_at': timestamp,
                        'label': signal_info.get('label', 'neutral'),
                        'confidence': signal_info.get('confidence', 0.0),
                        'signal_result': signal_info.get('result', 'unknown'),
                        'price_change': signal_info.get('price_change', 0.0),
                        'holding_periods': signal_info.get('holding_periods', 0)
                    }
                    
                    # 保存信息文件
                    with open(seq_dir / 'info.json', 'w', encoding='utf-8') as f:
                        json.dump(info, f, ensure_ascii=False, indent=2)
                        
                    sbs_sequences_saved += 1
                    self.logger.debug(f"已保存序列及其信息: {seq_dir}")
                else:
                    self.logger.warning(f"未找到图表的信号信息: {path}")
            except Exception as e:
                self.logger.error(f"保存序列图片失败: {e}")
        
        self.logger.info(f"成功保存 {sbs_sequences_saved} 个完整的SBS序列")
        
        # 更新进度通知
        if self.progress_notifier:
            self.progress_notifier.update_status({
                'sbs_sequences': sbs_sequences_saved,
                'total_signals': signal_stats['total_signals'],
                'successful_signals': signal_stats['successful_signals'],
                'failed_signals': signal_stats['failed_signals'],
                'win_rate': signal_stats['win_rate']
            })
            
        # 创建验证集
        if self.validation_creator.create_annotated_images:
            try:
                validation_count = self.validation_creator.create_validation_set()
                self.logger.info(f"已创建验证集，共 {validation_count} 个序列")
                
                # 创建验证集分布图表
                chart_path = self.validation_creator.create_distribution_chart()
                self.logger.info(f"已创建验证集分布图表: {chart_path}")
            except Exception as e:
                self.logger.error(f"创建验证集时出错: {e}")
    
    def generate_profit_chart(self, profits: List[float]):
        """生成盈利情况图表"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(profits)), profits, marker='o')
        plt.title('每个Epoch的盈利情况')
        plt.xlabel('Epoch')
        plt.ylabel('盈利')
        plt.grid(True)
        
        chart_path = os.path.join(self.output_dir, 'profit_chart.png')
        plt.savefig(chart_path)
        plt.close()
        return chart_path

    def train_model(self, model_config: Dict = None):
        """训练自监督学习模型"""
        from src.self_supervised.model.sequence_model import SequenceModel
        from src.self_supervised.utils.validation_set_creator import ValidationSetCreator
        
        self.logger.info("开始训练序列模型")
        
        # 默认配置
        config = {
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32,
            'validation': True,
            'validation_set_info': {'name': '2024年最后一周'},
            'resume_checkpoint': None,
            'save_memory': False
        }
        
        # 更新配置
        if model_config:
            config.update(model_config)
        
        # 创建模型
        model = SequenceModel(
            input_dim=512,  # 特征维度
            hidden_dim=256,
            output_dim=3,   # bullish, bearish, neutral
            num_layers=2
        )
        
        # 创建训练器
        trainer = SelfSupervisedTrainer(
            model=model,
            data_dir=os.path.join(self.output_dir, 'sbs_sequences'),
            save_dir='models/self_supervised'
        )
        
        # 设置内存优化
        if config['save_memory']:
            trainer.enable_memory_optimization(True)
        
        # 从检查点恢复
        if config['resume_checkpoint']:
            self.logger.info(f"从检查点恢复训练: {config['resume_checkpoint']}")
            trainer.load_checkpoint(config['resume_checkpoint'])
        
        # 设置奖励机制
        if self.reward_config:
            trainer.set_reward_config(self.reward_config)
        
        # 设置进度通知器
        if self.progress_notifier:
            trainer.set_notifier(self.progress_notifier)
        
        # 创建验证集
        validation_creator = ValidationSetCreator(
            base_data_path='data/NQ_full_1min_continuous.csv',
            output_dir=self.output_dir
        )
        validation_info = validation_creator.create_last_week_of_year_validation(year=2024)
        
        # 训练模型
        self.logger.info(f"开始训练，轮次: {config['epochs']}, 学习率: {config['learning_rate']}")
        total_epochs = config['epochs']
        self.progress_notifier.set_total_epochs(total_epochs)
        profits = []  # 用于记录每个epoch的盈利情况
        
        for epoch in range(total_epochs):
            # 训练逻辑...
            metrics = self.run_training_epoch(epoch)
            
            # 计算当前盈利情况
            profit = self.calculate_profit(metrics)
            profits.append(profit)  # 记录盈利情况
            
            # 进行验证
            validation_metrics = self.validate_model(validation_info)
            
            # 发送训练进度通知
            if self.progress_notifier:
                self.progress_notifier.send_training_progress(epoch, metrics, profit=profit)
                self.progress_notifier.send_training_progress(epoch, validation_metrics, profit=None)
            
            # 其他逻辑...
            
            # 在需要发送消息的地方调用 send_message
            if self.discord_notifier:
                self.discord_notifier.send_message("训练开始通知", {"epoch": epoch})
        
        # 训练结束后生成盈利情况图表
        profit_chart_path = self.generate_profit_chart(profits)
        self.logger.info(f'盈利情况图表已生成: {profit_chart_path}')
        
        # 训练结束后的逻辑...
        
        # 保存训练指标
        self.training_metrics = metrics
        self.best_metrics = metrics
        
        self.logger.info(f"训练完成，最终准确率: {self.training_metrics.get('accuracy', 0):.4f}")
        
        return metrics
    
    def _update_progress(self, current: int, total: int):
        """更新进度
        
        Args:
            current: 当前处理数量
            total: 总数量
        """
        progress = min(current / total if total > 0 else 0, 1.0)
        sys.stdout.write(f"\r处理进度: [{current}/{total}] {progress:.1%}")
        sys.stdout.flush()

    def _send_processing_progress(self, stage: str):
        """发送处理进度通知
        
        Args:
            stage: 当前处理阶段
        """
        if not self.progress_notifier:
            return
            
        # 发送进度通知
        message = f"**{stage}进度更新**\n\n"
        
        if stage == "图表生成":
            message += f"- 已生成图表: {self.charts_processed}\n"
        elif stage == "图表处理":
            message += f"- 已处理图表: {self.charts_processed}\n"
            message += f"- 已找到SBS序列: {self.sbs_sequences_found}\n"
        
        # 添加内存使用信息
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        message += f"\n**资源使用:**\n"
        message += f"- 内存使用: {memory_mb:.2f} MB\n"
        message += f"- CPU使用: {psutil.cpu_percent()}%\n"
        
        # 发送通知
        self.progress_notifier.send_training_progress(
            message=message,
            metrics={
                "charts_processed": self.charts_processed,
                "sbs_sequences": self.sbs_sequences_found,
                "memory_mb": memory_mb
            }
        )
    
    def run_self_supervised_learning(self, skip_chart_gen: bool = False, save_memory: bool = False, resume_from_checkpoint: str = None, training_config: Dict = None, save_every: int = 2, validate_every: int = 1, checkpoint_dir: str = None, accumulation_steps: int = 1):
        """运行自监督学习流程
        Args:
            skip_chart_gen: 是否跳过图表生成
            save_memory: 是否启用内存优化
            resume_from_checkpoint: 从检查点恢复训练
            save_every: 每多少轮保存一次模型检查点
            validate_every: 每多少轮进行一次验证
            checkpoint_dir: 检查点保存目录
            accumulation_steps: 梯度累积步数
        """
        # 处理save_every和validate_every参数的逻辑
        self.logger.info(f"每 {save_every} 轮保存一次模型检查点")
        self.logger.info(f"每 {validate_every} 轮进行一次验证")
        # 继续其他训练逻辑
        ...
    
    def calculate_a100_estimation(self, num_charts: int) -> Dict[str, float]:
        """
        计算使用A100处理所有图表的估计时间和内存需求
        
        Args:
            num_charts: 图表数量
            
        Returns:
            包含估计时间和内存需求的字典
        """
        time_per_chart = 0.15  # 假设的A100处理时间
        memory_per_chart = 0.02  # 假设的A100内存需求
        
        # 批处理时间
        batch_size = self.batch_size
        batch_processing_time = (num_charts / batch_size) * time_per_chart
        
        # 总时间（考虑其他操作）
        total_time = batch_processing_time * 1.25  # 增加25%时间作为其他操作的估计
        
        # 转换为小时和天
        total_time_hours = total_time / 3600
        total_time_days = total_time_hours / 24
        
        # 内存需求
        total_memory_gb = num_charts * memory_per_chart
        
        return {
            'total_charts': num_charts,
            'time_per_chart': time_per_chart,
            'memory_per_chart': memory_per_chart,
            'batch_processing_time': batch_processing_time / 3600,  # 转换为小时
            'total_time_hours': total_time_hours,
            'total_time_days': total_time_days,
            'total_memory_gb': total_memory_gb
        } 

    def get_trainer(self) -> SelfSupervisedTrainer:
        """获取自监督训练器实例"""
        from .self_supervised_trainer import SelfSupervisedTrainer
        model = LLaVAModelWrapper(model_path=self.model_path)  # 确保使用正确的模型路径
        trainer = SelfSupervisedTrainer(
            model=model,
            data_dir=os.path.join(self.output_dir, 'sbs_sequences'),
            save_dir='models/self_supervised'
        )
        return trainer 