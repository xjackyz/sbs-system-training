"""
配置工具模块
提供配置文件的加载和管理功能
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from .logger import setup_logger

logger = setup_logger('config')

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
        
    Returns:
        配置字典
    """
    try:
        # 加载环境变量
        load_dotenv()
        
        # 如果未指定配置文件路径，使用默认路径
        if not config_path:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
            config_path = os.path.join(config_dir, 'config.yaml')
            
        # 检查文件是否存在
        if not os.path.exists(config_path):
            logger.warning(f'配置文件不存在: {config_path}，将使用默认配置')
            return get_default_config()
            
        # 根据文件扩展名选择加载方式
        _, ext = os.path.splitext(config_path)
        if ext.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        elif ext.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            logger.error(f'不支持的配置文件格式: {ext}')
            return get_default_config()
            
        # 合并环境变量
        config = merge_env_vars(config)
        
        logger.info(f'成功加载配置文件: {config_path}')
        return config
        
    except Exception as e:
        logger.error(f'加载配置文件失败: {str(e)}')
        return get_default_config()
        
def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    return {
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'debug': os.getenv('DEBUG', 'True').lower() == 'true',
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        
        'device': {
            'use_gpu': os.getenv('USE_GPU', 'True').lower() == 'true',
            'num_workers': int(os.getenv('NUM_WORKERS', '4'))
        },
        
        'model': {
            'path': os.getenv('MODEL_PATH', 'models/latest'),
            'batch_size': int(os.getenv('BATCH_SIZE', '32'))
        },
        
        'api': {
            'tradingview_key': os.getenv('TRADINGVIEW_API_KEY'),
            'discord_webhook': os.getenv('DISCORD_WEBHOOK_URL')
        },
        
        'database': {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'name': os.getenv('DB_NAME', 'sbs_system'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        },
        
        'cache': {
            'redis_host': os.getenv('REDIS_HOST', 'localhost'),
            'redis_port': int(os.getenv('REDIS_PORT', '6379')),
            'redis_password': os.getenv('REDIS_PASSWORD')
        },
        
        'network': {
            'mirror_url': os.getenv('MIRROR_URL', 'https://api.example.com'),
            'verify_ssl': os.getenv('VERIFY_SSL', 'True').lower() == 'true',
            'timeout': int(os.getenv('API_TIMEOUT', '30'))
        },
        
        'security': {
            'secret_key': os.getenv('SECRET_KEY'),
            'jwt_secret': os.getenv('JWT_SECRET'),
            'encryption_key': os.getenv('ENCRYPTION_KEY')
        },
        
        'monitoring': {
            'prometheus_port': int(os.getenv('PROMETHEUS_PORT', '9090')),
            'grafana_port': int(os.getenv('GRAFANA_PORT', '3000'))
        },
        
        'backup': {
            'path': os.getenv('BACKUP_PATH', '/path/to/backup'),
            'retention_days': int(os.getenv('BACKUP_RETENTION_DAYS', '7'))
        },
        
        'notification': {
            'enabled': os.getenv('NOTIFICATION_ENABLED', 'True').lower() == 'true',
            'level': os.getenv('NOTIFICATION_LEVEL', 'INFO')
        }
    }
    
def merge_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    将环境变量合并到配置中
    
    Args:
        config: 原始配置字典
        
    Returns:
        合并后的配置字典
    """
    try:
        # 环境变量优先级高于配置文件
        env_config = {
            'environment': os.getenv('ENVIRONMENT'),
            'debug': os.getenv('DEBUG'),
            'log_level': os.getenv('LOG_LEVEL'),
            'device': {
                'use_gpu': os.getenv('USE_GPU'),
                'num_workers': os.getenv('NUM_WORKERS')
            },
            # ... 其他环境变量
        }
        
        # 递归更新配置
        def update_dict(d1: Dict, d2: Dict) -> Dict:
            for k, v in d2.items():
                if isinstance(v, dict) and k in d1:
                    d1[k] = update_dict(d1[k], v)
                elif v is not None:  # 只更新非None值
                    d1[k] = v
            return d1
            
        return update_dict(config, env_config)
        
    except Exception as e:
        logger.error(f'合并环境变量失败: {str(e)}')
        return config
        
def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        filepath: 保存路径
    """
    try:
        # 创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 根据文件扩展名选择保存方式
        _, ext = os.path.splitext(filepath)
        if ext.lower() == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        elif ext.lower() in ['.yml', '.yaml']:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, allow_unicode=True)
        else:
            logger.error(f'不支持的配置文件格式: {ext}')
            return
            
        logger.info(f'配置已保存到: {filepath}')
        
    except Exception as e:
        logger.error(f'保存配置文件失败: {str(e)}')
        raise 