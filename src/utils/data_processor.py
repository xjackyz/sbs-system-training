import os
import logging
from dotenv import load_dotenv

def load_config(config_path=None):
    """
    加载环境变量配置和YAML配置文件
    
    Args:
        config_path: YAML配置文件路径（可选）
    """
    # 加载环境变量
    load_dotenv()
    
    config = {
        # 系统配置
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'debug': os.getenv('DEBUG', 'true').lower() == 'true',
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        
        # 设备配置
        'use_gpu': os.getenv('USE_GPU', 'true').lower() == 'true',
        'num_workers': int(os.getenv('NUM_WORKERS', '4')),
        
        # 模型配置
        'model_path': os.getenv('MODEL_PATH', 'models/llava-sbs'),
        'vision_model_path': os.getenv('VISION_MODEL_PATH', 'openai/clip-vit-large-patch14-336'),
        'max_length': int(os.getenv('MAX_LENGTH', '4096')),
        'batch_size': int(os.getenv('BATCH_SIZE', '4')),
        
        # 缓存配置
        'cache_dir': os.getenv('CACHE_DIR', '.cache'),
        'use_cache': os.getenv('USE_CACHE', 'true').lower() == 'true',
        
        # 网络配置
        'use_mirror': os.getenv('USE_MIRROR', 'true').lower() == 'true',
        'mirror_url': os.getenv('MIRROR_URL', 'https://hf-mirror.com'),
        'verify_ssl': os.getenv('VERIFY_SSL', 'true').lower() == 'true',
        
        # Discord配置
        'discord_bot_token': os.getenv('DISCORD_BOT_TOKEN'),
        'discord_client_id': os.getenv('DISCORD_CLIENT_ID'),
        'discord_permissions': os.getenv('DISCORD_PERMISSIONS'),
        
        # Discord Webhooks
        'discord_monitor_webhook': os.getenv('DISCORD_MONITOR_WEBHOOK'),
        'discord_signal_webhook': os.getenv('DISCORD_SIGNAL_WEBHOOK'),
        'discord_upload_webhook': os.getenv('DISCORD_UPLOAD_WEBHOOK'),
        
        # 代理配置
        'http_proxy': os.getenv('HTTP_PROXY'),
        'https_proxy': os.getenv('HTTPS_PROXY'),
        
        # 数据路径
        'data_path': os.getenv('DATA_PATH', '/home/easyai/桌面/sbs_system/data'),
        
        # 安全配置
        'api_key': os.getenv('API_KEY'),
        'enable_ssl': os.getenv('ENABLE_SSL', 'false').lower() == 'true',
        
        # 自监督学习配置
        'self_supervised_enable': os.getenv('SELF_SUPERVISED_ENABLE', 'false').lower() == 'true',
    }
    
    # 如果提供了配置文件路径，加载YAML配置
    if config_path:
        import yaml
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                config.update(yaml_config)
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            raise
    
    # 验证必要的配置项
    if not config['discord_bot_token']:
        logging.warning("Discord Bot Token未设置，Discord Bot功能将无法使用")
        
    if not (config['discord_monitor_webhook'] and config['discord_signal_webhook'] and config['discord_upload_webhook']):
        logging.warning("Discord Webhook未完全设置，部分通知功能可能无法使用")
    
    return config 