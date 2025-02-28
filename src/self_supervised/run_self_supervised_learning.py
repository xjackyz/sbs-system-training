import os
import argparse
import json
import logging
import time
import sys
import traceback
from datetime import datetime
from pathlib import Path
import asyncio

# 添加src到Python路径
sys.path.append('/home/easyai/桌面/sbs_system/src')

from src.self_supervised.trainer.self_supervised_manager import SelfSupervisedManager
from src.self_supervised.utils.validation_set_creator import ValidationSetCreator
from src.self_supervised.utils.progress_notifier import ProgressNotifier
from src.notification.discord_notifier import DiscordNotifier
from src.self_supervised.utils.error_handler import ErrorLogger, ErrorHandler, catch_and_log_errors

os.chdir('/home/easyai/桌面/sbs_system')  # 设置当前工作目录

def main():
    """主函数"""
    log_timestamp = setup_logging()
    
    # 解析命令行参数
    args = parse_arguments()  # 确保在这里解析参数
    
    # 创建 SelfSupervisedManager 实例
    manager = SelfSupervisedManager(
        csv_path=args.csv_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validate_every=args.validate_every,
        save_every=args.save_every,
        use_lr_scheduler=args.use_lr_scheduler,
        scheduler_type=args.scheduler_type,
        scheduler_t_max=args.scheduler_t_max,
        scheduler_eta_min=args.scheduler_min_lr,
        use_early_stopping=args.use_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_monitor=args.early_stopping_monitor,
        checkpoint_dir=args.checkpoint_dir,
        discord_webhook=args.discord_webhook,
        save_memory=args.save_memory,
        incremental_processing=args.incremental_processing,
        incremental_batch_size=args.incremental_batch_size,
        gc_interval=args.gc_interval
    )

    

def setup_logging():
    """设置日志"""
    log_dir = os.path.join('logs', 'self_supervised')
    os.makedirs(log_dir, exist_ok=True)
    
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'run_{log_timestamp}.log')
    
    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 设置其他库的日志级别
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logging.info(f"日志文件路径: {log_file}")
    return log_timestamp


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='自监督学习训练脚本')
    
    # 数据参数组
    data_group = parser.add_argument_group('数据参数')
    data_group.add_argument('--csv_path', type=str, required=True, help='CSV数据文件路径')
    data_group.add_argument('--window_size', type=int, default=100, help='窗口大小（K线数量）')
    data_group.add_argument('--stride', type=int, default=60, help='步长（生成图表时的滑动窗口步长）')
    data_group.add_argument('--normalize', type=bool, default=True, help='是否归一化数据')
    data_group.add_argument('--output_dir', type=str, default='output', help='输出目录')
    
    # 验证集参数组
    validation_group = parser.add_argument_group('验证集参数')
    validation_group.add_argument('--create_validation_set', action='store_true', help='是否创建验证集')
    validation_group.add_argument('--validation_mode', type=str, default='time', choices=['time', 'random'], help='验证集创建模式: time (基于时间) 或 random (随机抽样)')
    validation_group.add_argument('--validation_ratio', type=float, default=0.1, help='验证集比例 (当mode=random时使用)')
    validation_group.add_argument('--validation_from_date', type=str, help='验证集起始日期 (当mode=time时使用，格式: YYYY-MM-DD)')
    validation_group.add_argument('--validation_to_date', type=str, help='验证集结束日期 (当mode=time时使用，格式: YYYY-MM-DD)')
    
    # 训练参数组
    training_group = parser.add_argument_group('训练参数')
    training_group.add_argument('--batch_size', type=int, default=32, help='批量大小')
    training_group.add_argument('--epochs', type=int, default=10, help='训练轮数')
    training_group.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    training_group.add_argument('--validate_every', type=int, default=1, help='每多少轮验证一次')
    training_group.add_argument('--accumulation_steps', type=int, default=1, help='梯度累积步数，用于增大有效批处理大小')
    training_group.add_argument('--effective_batch_size', type=int, help='有效批处理大小，如果指定，将自动计算所需的梯度累积步数')
    
    # 添加学习率调度器参数
    lr_scheduler_group = parser.add_argument_group('学习率调度器参数')
    lr_scheduler_group.add_argument('--use_lr_scheduler', action='store_true', help='是否使用学习率调度器')
    lr_scheduler_group.add_argument('--scheduler_type', type=str, default='plateau', 
                                   choices=['plateau', 'cosine', 'step', 'cyclic'], 
                                   help='学习率调度器类型')
    lr_scheduler_group.add_argument('--scheduler_patience', type=int, default=5, 
                                   help='ReduceLROnPlateau的耐心值，即经过多少个验证周期性能未提升才降低学习率')
    lr_scheduler_group.add_argument('--scheduler_factor', type=float, default=0.5, 
                                   help='ReduceLROnPlateau的衰减因子，学习率减少的比例')
    lr_scheduler_group.add_argument('--scheduler_min_lr', type=float, default=1e-6, 
                                   help='最小学习率，学习率不会衰减到低于此值')
    lr_scheduler_group.add_argument('--scheduler_t_max', type=int, default=10, 
                                   help='CosineAnnealingLR的T_max参数，半周期长度')
    lr_scheduler_group.add_argument('--scheduler_step_size', type=int, default=10, 
                                   help='StepLR的step_size参数，每多少轮衰减一次')
    lr_scheduler_group.add_argument('--scheduler_gamma', type=float, default=0.1, 
                                   help='StepLR的gamma参数，衰减因子')
    
    # 添加早停参数
    early_stopping_group = parser.add_argument_group('早停参数')
    early_stopping_group.add_argument('--use_early_stopping', action='store_true', 
                                     help='是否使用早停机制')
    early_stopping_group.add_argument('--early_stopping_patience', type=int, default=10, 
                                     help='早停的耐心值，即经过多少个验证周期性能未提升才停止训练')
    early_stopping_group.add_argument('--early_stopping_delta', type=float, default=0.001, 
                                     help='早停的最小改进量，低于此值的改进被视为没有提升')
    early_stopping_group.add_argument('--early_stopping_monitor', type=str, default='val_loss', 
                                     choices=['val_loss', 'val_accuracy'], 
                                     help='早停监控的指标')
    
    # 模型参数组
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--model_path', type=str, help='预训练模型路径 (如果要从检查点继续训练)')
    
    # 奖励机制参数组
    reward_group = parser.add_argument_group('奖励机制参数')
    reward_group.add_argument('--use_nq_contract', action='store_true', help='是否使用NQ合约作为奖励信号')
    reward_group.add_argument('--reward_config', type=str, help='奖励配置文件路径 (JSON格式)')
    
    # 保存参数组
    save_group = parser.add_argument_group('保存参数')
    save_group.add_argument('--save_every', type=int, default=5, help='每多少轮保存一次模型检查点')
    save_group.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')
    
    # Discord通知参数组
    discord_group = parser.add_argument_group('Discord通知参数')
    discord_group.add_argument('--discord_webhook', type=str, help='Discord Webhook URL')
    discord_group.add_argument('--notify_every', type=int, default=5, help='每多少轮通知一次进度')
    
    # 内存优化参数组
    memory_group = parser.add_argument_group('内存优化参数')
    memory_group.add_argument('--save_memory', action='store_true', help='启用内存优化模式，减少内存使用')
    memory_group.add_argument('--incremental_processing', action='store_true', help='启用增量数据处理，适用于大型数据集')
    memory_group.add_argument('--incremental_batch_size', type=int, default=5000, help='增量处理的批处理大小')
    memory_group.add_argument('--cache_process_results', action='store_true', help='将处理结果缓存到磁盘，减少内存使用')
    memory_group.add_argument('--gc_interval', type=int, default=1000, help='垃圾回收间隔(以处理窗口数计)')
    
    # 其他参数
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--skip_chart_gen', action='store_true', help='跳过图表生成步骤')
    parser.add_argument('--skip_train', action='store_true', help='跳过训练步骤，只生成图表')
    parser.add_argument('--estimate_only', action='store_true', help='只估算处理时间和资源使用，不执行实际训练')
    
    # 添加错误处理参数组
    error_group = parser.add_argument_group('错误处理参数')
    error_group.add_argument('--auto_restart', action='store_true', help='训练出错时自动重启')
    error_group.add_argument('--max_retries', type=int, default=3, help='最大重试次数')
    error_group.add_argument('--retry_delay', type=int, default=60, help='重试延迟（秒）')
    error_group.add_argument('--performance_monitoring', action='store_true', help='启用性能监控')
    error_group.add_argument('--performance_threshold', type=float, default=0.05, help='性能下降阈值')
    error_group.add_argument('--monitoring_window', type=int, default=5, help='性能监控窗口大小')
    
    return parser.parse_args()


def estimate_processing(args):
    """估计处理时间和资源需求
    
    Args:
        args: 命令行参数
        
    Returns:
        估计信息字典
    """
    # 计算每个图表的内存占用
    chart_size_mb = 0.02  # 单个图表约20KB
    # 每个SBS序列的内存占用
    sbs_sequence_size_mb = 0.1  # 约100KB/序列
    
    # 加载CSV文件并计算总行数
    with open(args.csv_path, 'r') as f:
        next(f)  # 跳过标题行
        total_rows = sum(1 for _ in f)
    
    logging.info(f"数据文件: {args.csv_path}, 总行数: {total_rows}")
    
    # 计算图表数量
    total_charts = (total_rows - args.window_size) // args.stride + 1
    if total_charts <= 0:
        total_charts = 1
    
    logging.info(f"窗口大小: {args.window_size}, 步长: {args.stride}")
    logging.info(f"估计图表数量: {total_charts}")
    
    # 估计处理时间
    chart_processing_time = 0.15  # 每个图表平均处理时间（秒）
    total_processing_time = total_charts * chart_processing_time  # 计算总处理时间
    total_hours = total_processing_time / 3600
    
    logging.info(f"平均每张图表处理时间: {chart_processing_time:.4f} 秒")
    logging.info(f"估计总处理时间: {total_hours:.2f} 小时 ({total_processing_time:.2f} 秒)")
    
    # 估计内存需求
    estimated_memory_mb = total_charts * chart_size_mb
    estimated_sbs_memory_mb = total_charts * 0.05 * sbs_sequence_size_mb  # 假设5%的图表形成SBS序列
    total_memory_mb = estimated_memory_mb + estimated_sbs_memory_mb
    
    # 考虑内存优化
    if args.save_memory or args.incremental_processing:
        # 增量处理模式下内存占用降低
        memory_reduction_factor = 0.3 if args.incremental_processing else 0.7
        total_memory_mb = total_memory_mb * memory_reduction_factor
        logging.info(f"启用内存优化，预计内存需求减少至原始需求的 {int(memory_reduction_factor*100)}%")
    
    logging.info(f"估计内存需求: {total_memory_mb:.2f} MB ({total_memory_mb/1024:.2f} GB)")
    
    # 返回估计信息
    return {
        'total_rows': total_rows,
        'total_charts': total_charts,
        'processing_time_seconds': total_processing_time,
        'processing_time_hours': total_hours,
        'memory_mb': total_memory_mb
    }


def setup_progress_notifier(args, log_timestamp):
    """设置进度通知器
    
    Args:
        args: 命令行参数
        log_timestamp: 日志时间戳
        
    Returns:
        进度通知器实例或None
    """
    if not args.discord_webhook:
        return None
    
    # 创建进度通知器
    notifier = ProgressNotifier(
        interval_minutes=args.notify_every
    )
    
    # 发送启动通知
    try:
        config_summary = {
            'csv_file': os.path.basename(args.csv_path),
            'window_size': args.window_size,
            'stride': args.stride,
            'batch_size': args.batch_size,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'log_file': f'logs/self_supervised/run_{log_timestamp}.log',
            'estimated_time': f'预计训练时间: {args.epochs * (total_processing_time / args.batch_size):.2f}小时'
        }
        notifier.send_training_started(config_summary)
        logger.info('已发送启动通知')

        # 设置阶段性推送
        asyncio.create_task(notifier.on_periodic_notification())
    except Exception as e:
        logger.error(f'发送启动通知时出错: {e}')
    
    return notifier


def create_validation_set(args):
    """创建验证集
    
    Args:
        args: 命令行参数
        
    Returns:
        验证集信息
    """
    if not args.create_validation_set:
        logging.info("未启用验证集创建")
        return None
    
    logging.info("开始创建验证集...")
    
    # 创建验证集创建器
    validation_creator = ValidationSetCreator(
        base_data_path=args.csv_path,
        output_dir="data/validation"
    )
    
    validation_info = None
    
    # 根据验证集类型创建验证集
    if args.validation_mode == 'time':
        validation_info = validation_creator.create_date_range_validation(
            start_date=args.validation_from_date,
            end_date=args.validation_to_date
        )
        logging.info(f"已创建日期范围验证集: {args.validation_from_date} 到 {args.validation_to_date}")
    elif args.validation_mode == 'random':
        validation_info = validation_creator.create_random_validation(
            ratio=args.validation_ratio
        )
        logging.info(f"已创建随机验证集，比例: {args.validation_ratio}")
    
    if validation_info:
        logging.info(f"验证集创建完成: {validation_info.get('name')}")
        logging.info(f"包含 {validation_info.get('data_points')} 个数据点")
    else:
        logging.warning("验证集创建失败或返回空信息")
    
    return validation_info


def main():
    """主函数"""
    log_timestamp = setup_logging()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # 如果只是估算，则调用估算函数并退出
    if args.estimate_only:
        estimate_processing(args)
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 创建日志目录
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # 计算有效批处理大小和累积步数
    if args.effective_batch_size:
        if args.effective_batch_size < args.batch_size:
            logger.warning(f"有效批处理大小({args.effective_batch_size})小于实际批处理大小({args.batch_size})，将使用实际批处理大小")
            args.accumulation_steps = 1
        else:
            # 计算所需的梯度累积步数
            args.accumulation_steps = max(1, args.effective_batch_size // args.batch_size)
            logger.info(f"有效批处理大小: {args.effective_batch_size}, 实际批处理大小: {args.batch_size}, 梯度累积步数: {args.accumulation_steps}")
    
    # 初始化Discord通知器
    discord_notifier = None
    if args.discord_webhook:
        try:
            discord_notifier = DiscordNotifier()
            logger.info("已初始化Discord通知器")
        except Exception as e:
            logger.error(f"初始化Discord通知器失败: {e}")
    
    # 初始化错误日志记录器和错误处理器
    error_logger = ErrorLogger(
        log_dir=logs_dir,
        notifier=discord_notifier,
        save_system_info=True
    )
    
    error_handler = ErrorHandler(
        error_logger=error_logger,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        auto_restart=args.auto_restart
    )
    
    # 读取奖励配置
    reward_config = None
    if args.reward_config:
        try:
            with open(args.reward_config, 'r') as f:
                reward_config = json.load(f)
        except Exception as e:
            error_logger.log_error(e, {'file': args.reward_config}, 
                                 recovery_suggestion="请检查奖励配置文件路径是否正确，确保文件格式为有效的JSON。")
            logger.error(f"无法加载奖励配置: {e}")
            reward_config = None
    
    # 创建进度通知器
    progress_notifier = ProgressNotifier(
        discord_notifier=discord_notifier,
        notify_every=args.notify_every,
        charts_dir=os.path.join(args.output_dir, 'charts')
    )
    
    # 记录开始时间
    start_time = time.time()
    
    # 发送开始训练通知
    if discord_notifier:
        training_config = {
            'batch_size': args.batch_size,
            'effective_batch_size': args.effective_batch_size or args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'csv_path': os.path.basename(args.csv_path),
            'window_size': args.window_size,
            'stride': args.stride,
            'memory_optimization': args.save_memory,
            'incremental_processing': args.incremental_processing
        }
        
        progress_notifier.send_training_started(training_config)
    
    try:
        # 创建SelfSupervisedManager实例
        manager = SelfSupervisedManager(
            csv_path=args.csv_path,
            model_path=args.model_path,
            output_dir=args.output_dir,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            effective_batch_size=args.effective_batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            validate_every=args.validate_every,
            save_every=args.save_every,
            use_lr_scheduler=args.use_lr_scheduler,
            scheduler_type=args.scheduler_type,
            scheduler_t_max=args.scheduler_t_max,
            scheduler_eta_min=args.scheduler_min_lr,
            use_early_stopping=args.use_early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_monitor=args.early_stopping_monitor,
            checkpoint_dir=args.checkpoint_dir,
            discord_webhook=args.discord_webhook,
            save_memory=args.save_memory,
            incremental_processing=args.incremental_processing,
            incremental_batch_size=args.incremental_batch_size,
            gc_interval=args.gc_interval
        )
        
        # 设置内存优化选项
        if args.save_memory or args.incremental_processing:
            manager.save_memory = args.save_memory
            manager.use_incremental_processing = args.incremental_processing
            manager.incremental_batch_size = args.incremental_batch_size
            logger.info(f"启用内存优化: {'是' if args.save_memory else '否'}, 增量处理: {'是' if args.incremental_processing else '否'}")
        
        # 设置验证集创建选项
        if args.create_validation_set:
            validation_params = {
                'mode': args.validation_mode,
                'ratio': args.validation_ratio,
            }
            
            if args.validation_mode == 'time' and args.validation_from_date and args.validation_to_date:
                validation_params['from_date'] = args.validation_from_date
                validation_params['to_date'] = args.validation_to_date
            
            manager.set_validation_params(**validation_params)
        
        # 设置总轮次以便进度通知
        if progress_notifier:
            progress_notifier.set_total_epochs(args.epochs)
        
        # 运行自监督学习
        manager.run_self_supervised_learning(
            skip_chart_gen=args.skip_chart_gen,
            save_every=args.save_every,
            validate_every=args.validate_every,
            checkpoint_dir=args.checkpoint_dir,
            accumulation_steps=args.accumulation_steps
        )
        
        # 如果使用学习率调度器，设置调度器
        if args.use_lr_scheduler:
            scheduler_params = {
                'factor': args.scheduler_factor,
                'patience': args.scheduler_patience,
                'min_lr': args.scheduler_min_lr,
                'T_max': args.scheduler_t_max,
                'step_size': args.scheduler_step_size,
                'gamma': args.scheduler_gamma
            }
            trainer = manager.get_trainer()
            if trainer:
                trainer.setup_lr_scheduler(
                    scheduler_type=args.scheduler_type,
                    **scheduler_params
                )
                logger.info(f"设置学习率调度器: {args.scheduler_type}")
        
        # 如果使用早停机制，设置早停
        if args.use_early_stopping:
            trainer = manager.get_trainer()
            if trainer:
                trainer.setup_early_stopping(
                    patience=args.early_stopping_patience,
                    min_delta=args.early_stopping_delta,
                    monitor=args.early_stopping_monitor
                )
                logger.info(f"设置早停机制: {args.early_stopping_monitor}, 耐心度: {args.early_stopping_patience}")
        
        # 设置性能监控
        if args.performance_monitoring:
            if trainer:
                trainer.setup_performance_monitoring(
                    threshold=args.performance_threshold,
                    window_size=args.monitoring_window
                )
                logger.info(f"设置性能监控: 阈值={args.performance_threshold}, 窗口大小={args.monitoring_window}")
        
        # 计算总训练时间
        total_time = time.time() - start_time
        hours, remainder = divmod(int(total_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours}小时{minutes}分钟{seconds}秒" if hours > 0 else f"{minutes}分钟{seconds}秒"
        
        logger.info(f"自监督学习完成! 总耗时: {time_str}")
        
        # 创建并发送训练总结
        if discord_notifier:
            trainer = manager.get_trainer()
            if trainer:
                # 获取最终指标
                final_metrics = trainer.get_final_metrics() if hasattr(trainer, 'get_final_metrics') else {}
                best_metrics = trainer.get_best_metrics() if hasattr(trainer, 'get_best_metrics') else {}
                
                # 创建总结图表
                summary_chart = progress_notifier.create_training_summary_chart()
                charts = [summary_chart] if summary_chart else []
                
                # 发送完成通知
                discord_notifier.send_message(
                    f"🎉 **自监督学习训练完成!** 🎉\n\n"
                    f"**总训练时间:** {time_str}\n"
                    f"**总轮次:** {args.epochs}\n"
                    f"**CSV数据:** {os.path.basename(args.csv_path)}\n"
                    f"**窗口大小:** {args.window_size}\n"
                    f"**步长:** {args.stride}\n\n"
                    f"**最终指标:**\n" + 
                    "\n".join([f"- {k}: {v:.4f}" for k, v in final_metrics.items()]) +
                    (f"\n\n**最佳指标:**\n" + "\n".join([f"- {k}: {v:.4f}" for k, v in best_metrics.items()]) if best_metrics else ""),
                    files=charts
                )
    
    except Exception as e:
        # 获取异常信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # 记录错误
        logger.exception("训练过程中发生错误")
        
        # 记录到错误日志
        error_context = {
            'phase': 'main',
            'args': {k: v for k, v in vars(args).items() if not k.startswith('_')}
        }
        error_logger.log_error(e, error_context, notify=True)
        
        # 发送错误通知
        if discord_notifier:
            try:
                discord_notifier.send_message(
                    f"❌ **训练过程中出现错误** ❌\n\n"
                    f"**错误类型:** {type(e).__name__}\n"
                    f"**错误消息:** {str(e)}\n\n"
                    f"详细信息请查看日志文件。\n\n"
                    f"**发生时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            except Exception as notify_error:
                logger.error(f"发送错误通知时出错: {notify_error}")
        
        # 抛出异常，确保进程以非零状态退出
        raise


if __name__ == "__main__":
        main()