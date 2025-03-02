#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SBS系统日志工具模块
配置和管理SBS系统的日志记录
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(config=None, level=logging.INFO, log_to_file=True):
    """
    设置日志记录配置
    
    参数:
        config: 配置字典，包含日志配置
        level: 日志级别
        log_to_file: 是否将日志写入文件
    """
    # 重置之前的日志配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 创建格式器
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # 设置根日志器级别
    logging.root.setLevel(level)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(console_handler)
    
    # 如果配置中指定了日志级别，则使用它
    if config and 'logging' in config:
        level_name = config['logging'].get('level', 'INFO')
        level = getattr(logging, level_name.upper())
        logging.root.setLevel(level)
    
    # 根据配置添加文件处理器
    if log_to_file:
        log_dir = None
        
        # 从配置中获取日志目录
        if config and 'paths' in config and 'log_dir' in config['paths']:
            log_dir = config['paths']['log_dir']
        
        # 如果未指定，使用默认日志目录
        if not log_dir:
            log_dir = 'logs'
        
        # 确保日志目录存在
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{log_dir}/sbs_{timestamp}.log"
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
        
        logging.info(f"日志将被保存到 {log_filename}")
    
    # 设置第三方库的日志级别
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    logging.info("日志系统初始化完成")
    

def get_logger(name, level=None):
    """
    获取具有指定名称的日志器
    
    参数:
        name: 日志器名称
        level: 日志级别（可选）
        
    返回:
        logger: 日志器实例
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
    
    return logger


class LogCapture:
    """
    用于捕获和重定向日志的上下文管理器
    """
    
    def __init__(self, logger_name=None):
        """
        初始化日志捕获器
        
        参数:
            logger_name: 要捕获的日志器名称，如果为None则捕获根日志器
        """
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name) if logger_name else logging.root
        self.log_records = []
        self.handler = None
        
    def __enter__(self):
        """
        进入上下文，开始捕获日志
        """
        class MemoryHandler(logging.Handler):
            def __init__(self, record_list):
                super().__init__()
                self.record_list = record_list
                
            def emit(self, record):
                self.record_list.append(record)
        
        self.handler = MemoryHandler(self.log_records)
        self.logger.addHandler(self.handler)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文，停止捕获日志
        """
        self.logger.removeHandler(self.handler)
        
    def get_logs(self, level=None, formatter=None):
        """
        获取捕获的日志记录
        
        参数:
            level: 过滤指定级别的日志
            formatter: 日志格式化器
            
        返回:
            logs: 日志记录列表
        """
        if formatter is None:
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            
        logs = []
        for record in self.log_records:
            if level is None or record.levelno >= level:
                logs.append(formatter.format(record))
                
        return logs 