import os
import sys
import traceback
import logging
import time
import json
import psutil
import platform
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

from src.notification.discord_notifier import DiscordNotifier

class ErrorLogger:
    """错误日志记录器
    
    记录错误详情到日志文件，并可选择发送通知
    """
    
    def __init__(self, log_dir: str = "logs", 
                notifier: Optional[DiscordNotifier] = None,
                save_system_info: bool = True):
        """初始化错误日志记录器
        
        Args:
            log_dir: 日志保存目录
            notifier: Discord通知器
            save_system_info: 是否保存系统信息
        """
        self.log_dir = log_dir
        self.notifier = notifier
        self.save_system_info = save_system_info
        self.logger = logging.getLogger('error_logger')
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置文件处理器
        self.log_file = os.path.join(log_dir, f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 错误计数
        self.error_count = 0
        self.warning_count = 0
        
        # 状态信息
        self.status = {
            'last_error': None,
            'last_error_time': None,
            'system_info': self._get_system_info() if save_system_info else None
        }
    
    def _get_system_info(self) -> Dict:
        """获取系统信息
        
        Returns:
            系统信息字典
        """
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'hostname': platform.node(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
            'memory_available': psutil.virtual_memory().available / (1024 * 1024 * 1024),  # GB
            'disk_usage': {
                'total': psutil.disk_usage('/').total / (1024 * 1024 * 1024),  # GB
                'used': psutil.disk_usage('/').used / (1024 * 1024 * 1024),  # GB
                'free': psutil.disk_usage('/').free / (1024 * 1024 * 1024)  # GB
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return info
    
    def log_error(self, error: Exception, context: Dict = None, notify: bool = True,
                recovery_suggestion: str = None) -> str:
        """记录错误
        
        Args:
            error: 错误异常
            context: 错误发生时的上下文信息
            notify: 是否发送通知
            recovery_suggestion: 恢复建议
            
        Returns:
            错误日志文件路径
        """
        # 获取异常信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # 更新状态
        self.error_count += 1
        self.status['last_error'] = str(error)
        self.status['last_error_time'] = datetime.now().isoformat()
        
        # 构建错误消息
        error_msg = f"错误({self.error_count}): {str(error)}\n"
        error_msg += f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if context:
            error_msg += "\n上下文信息:\n"
            for key, value in context.items():
                error_msg += f"- {key}: {value}\n"
        
        error_msg += f"\n详细堆栈:\n{tb_str}"
        
        # 记录到日志文件
        self.logger.error(error_msg)
        
        # 创建单独的错误日志文件
        error_log_file = os.path.join(self.log_dir, f"error_{self.error_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        try:
            with open(error_log_file, 'w', encoding='utf-8') as f:
                # 写入错误信息
                f.write(error_msg)
                
                # 写入系统信息
                if self.save_system_info and self.status['system_info']:
                    f.write("\n\n系统信息:\n")
                    f.write(json.dumps(self.status['system_info'], indent=2, ensure_ascii=False))
                
                # 写入恢复建议
                if recovery_suggestion:
                    f.write(f"\n\n恢复建议:\n{recovery_suggestion}")
        except Exception as e:
            self.logger.error(f"写入错误日志文件时发生错误: {e}")
        
        # 发送通知
        if notify and self.notifier:
            self._send_error_notification(error, tb_str, context, recovery_suggestion)
        
        return error_log_file
    
    def _send_error_notification(self, error: Exception, traceback_str: str, 
                               context: Dict = None, recovery_suggestion: str = None):
        """发送错误通知
        
        Args:
            error: 错误异常
            traceback_str: 异常堆栈
            context: 错误上下文
            recovery_suggestion: 恢复建议
        """
        if not self.notifier:
            return
            
        # 构建消息
        message = f"❌ **训练中发生错误 #{self.error_count}** ❌\n\n"
        message += f"**错误类型:** {type(error).__name__}\n"
        message += f"**错误信息:** {str(error)}\n"
        message += f"**时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # 添加内存信息
        memory = psutil.virtual_memory()
        message += f"\n**内存状态:**\n"
        message += f"- 总内存: {memory.total / (1024*1024*1024):.2f} GB\n"
        message += f"- 可用内存: {memory.available / (1024*1024*1024):.2f} GB\n"
        message += f"- 内存使用率: {memory.percent}%\n"
        
        # 添加上下文信息
        if context:
            message += "\n**上下文信息:**\n"
            for key, value in context.items():
                message += f"- {key}: {value}\n"
        
        # 添加恢复建议
        if recovery_suggestion:
            message += f"\n**恢复建议:**\n{recovery_suggestion}\n"
        
        # 添加堆栈信息（截断以适应Discord消息限制）
        message += "\n**异常堆栈:**\n```\n"
        if len(traceback_str) > 1000:
            message += traceback_str[:997] + "...\n```"
        else:
            message += traceback_str + "\n```"
        
        # 发送消息
        try:
            self.notifier.send_message(message)
        except Exception as e:
            self.logger.error(f"发送错误通知时发生错误: {e}")
    
    def log_warning(self, message: str, context: Dict = None, notify: bool = True):
        """记录警告
        
        Args:
            message: 警告消息
            context: 上下文信息
            notify: 是否发送通知
        """
        self.warning_count += 1
        
        # 构建警告消息
        warning_msg = f"警告({self.warning_count}): {message}\n"
        warning_msg += f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if context:
            warning_msg += "\n上下文信息:\n"
            for key, value in context.items():
                warning_msg += f"- {key}: {value}\n"
        
        # 记录到日志
        self.logger.warning(warning_msg)
        
        # 发送通知
        if notify and self.notifier:
            self._send_warning_notification(message, context)
    
    def _send_warning_notification(self, message: str, context: Dict = None):
        """发送警告通知
        
        Args:
            message: 警告消息
            context: 上下文信息
        """
        if not self.notifier:
            return
            
        # 构建消息
        discord_message = f"⚠️ **训练警告 #{self.warning_count}** ⚠️\n\n"
        discord_message += f"**警告:** {message}\n"
        discord_message += f"**时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # 添加上下文信息
        if context:
            discord_message += "\n**上下文信息:**\n"
            for key, value in context.items():
                discord_message += f"- {key}: {value}\n"
        
        # 发送消息
        try:
            self.notifier.send_message(discord_message)
        except Exception as e:
            self.logger.error(f"发送警告通知时发生错误: {e}")

class ErrorHandler:
    """错误处理器
    
    捕获并处理训练过程中的错误
    """
    
    def __init__(self, error_logger: ErrorLogger, max_retries: int = 3, 
                retry_delay: int = 60, auto_restart: bool = False,
                critical_errors: List[str] = None):
        """初始化错误处理器
        
        Args:
            error_logger: 错误日志记录器
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            auto_restart: 是否自动重启
            critical_errors: 不重试的严重错误类型列表
        """
        self.error_logger = error_logger
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_restart = auto_restart
        self.critical_errors = critical_errors or ['MemoryError', 'KeyboardInterrupt', 'SystemExit']
        
        self.retries = 0
        self.logger = logging.getLogger('error_handler')
    
    def handle_error(self, error: Exception, context: Dict = None, 
                   recovery_func: Optional[Callable] = None) -> bool:
        """处理错误
        
        Args:
            error: 错误异常
            context: 错误上下文
            recovery_func: 恢复函数
            
        Returns:
            是否成功恢复
        """
        error_type = type(error).__name__
        
        # 记录错误
        recovery_suggestion = self._get_recovery_suggestion(error)
        self.error_logger.log_error(error, context, notify=True, recovery_suggestion=recovery_suggestion)
        
        # 检查是否为严重错误
        if error_type in self.critical_errors:
            self.logger.error(f"严重错误，不会尝试恢复: {error_type}")
            return False
        
        # 重试逻辑
        if self.retries < self.max_retries and self.auto_restart:
            self.retries += 1
            retry_message = f"尝试恢复（第{self.retries}次）..."
            self.logger.info(retry_message)
            
            # 等待一段时间后重试
            time.sleep(self.retry_delay)
            
            # 执行恢复函数
            if recovery_func:
                try:
                    recovery_func()
                    self.logger.info(f"恢复成功（第{self.retries}次尝试）")
                    return True
                except Exception as e:
                    self.logger.error(f"恢复失败: {e}")
            
            return False
        else:
            if self.retries >= self.max_retries:
                self.logger.error(f"达到最大重试次数 ({self.max_retries})，不再尝试恢复")
            return False
    
    def _get_recovery_suggestion(self, error: Exception) -> str:
        """获取恢复建议
        
        Args:
            error: 错误异常
            
        Returns:
            恢复建议
        """
        error_type = type(error).__name__
        
        suggestions = {
            'MemoryError': "内存不足。尝试减小批处理大小、启用梯度累积、增加内存优化或使用更小的模型。",
            'RuntimeError': "运行时错误，可能是CUDA相关。尝试重启训练，减小批处理大小，或检查CUDA版本兼容性。",
            'ValueError': "参数错误。检查数据格式和模型配置是否匹配。",
            'FileNotFoundError': "找不到文件。检查文件路径是否正确，确保数据文件存在。",
            'KeyError': "字典键错误。检查配置文件或代码中的键名是否正确。",
            'IndexError': "索引错误。检查数据批次大小或者数组索引是否超出范围。",
            'AttributeError': "属性错误。检查对象属性是否存在或拼写是否正确。",
        }
        
        if error_type in suggestions:
            return suggestions[error_type]
        
        # 默认建议
        return "请检查日志文件查找详细错误信息，并考虑调整训练参数或数据处理流程。"
    
    def reset_retries(self):
        """重置重试计数"""
        self.retries = 0

# 装饰器函数，用于自动捕获和处理函数调用中的错误
def catch_and_log_errors(error_handler: ErrorHandler, context_provider: Optional[Callable[[], Dict]] = None):
    """错误捕获和日志记录装饰器
    
    Args:
        error_handler: 错误处理器
        context_provider: 提供上下文信息的函数
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取上下文信息
                context = {}
                if context_provider:
                    try:
                        context = context_provider()
                    except Exception as ce:
                        logging.error(f"获取错误上下文时失败: {ce}")
                
                # 添加函数信息到上下文
                context.update({
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                })
                
                # 处理错误
                error_handler.handle_error(e, context)
                
                # 重新抛出异常
                raise
                
        return wrapper
    return decorator 