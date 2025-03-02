"""
配置管理器
用于统一管理系统配置，支持层次化配置和环境特定配置
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv

class ConfigManager:
    """统一配置管理类，用于加载、合并和访问系统配置"""
    
    def __init__(self, config_dir: str = None, config_name: str = "config.yaml", env_file: str = ".env"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的config/
            config_name: 主配置文件名，默认为config.yaml
            env_file: 环境变量文件名，默认为.env
        """
        # 设置配置目录
        self.config_dir = config_dir or os.path.join(os.getcwd(), 'config')
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
            
        # 设置日志
        self.logger = logging.getLogger('config_manager')
        
        # 加载环境变量
        self._load_env_vars(env_file)
        
        # 初始化配置字典
        self.config = {}
        
        # 加载主配置文件
        self._load_main_config(config_name)
        
        # 加载其他配置文件
        self._load_additional_configs()
        
        # 使用环境变量覆盖配置
        self._override_with_env_vars()
        
    def _load_env_vars(self, env_file: str) -> None:
        """
        加载环境变量文件
        
        Args:
            env_file: 环境变量文件名
        """
        env_path = os.path.join(os.getcwd(), env_file)
        if os.path.exists(env_path):
            load_dotenv(env_path)
            self.logger.info(f"加载环境变量文件: {env_path}")
        else:
            self.logger.warning(f"环境变量文件不存在: {env_path}")
            
    def _load_main_config(self, config_name: str) -> None:
        """
        加载主配置文件
        
        Args:
            config_name: 配置文件名
        """
        config_path = os.path.join(self.config_dir, config_name)
        if os.path.exists(config_path):
            self.config = self._load_yaml(config_path)
            self.logger.info(f"加载主配置文件: {config_path}")
        else:
            self.logger.warning(f"主配置文件不存在: {config_path}")
            
    def _load_additional_configs(self) -> None:
        """加载其他配置文件并合并到主配置中"""
        # 加载系统配置
        self._load_and_merge("system_config.yaml", "system")
        
        # 加载训练配置
        self._load_and_merge("training_config.yaml", "training")
        
        # 加载优化配置
        self._load_and_merge("optimization_config.yaml", "optimization")
        
        # 加载自监督学习配置
        self._load_and_merge("self_supervised_config.yaml", "self_supervised")
        
    def _load_and_merge(self, config_name: str, config_key: str) -> None:
        """
        加载配置文件并合并到主配置中
        
        Args:
            config_name: 配置文件名
            config_key: 配置在主配置中的键名
        """
        config_path = os.path.join(self.config_dir, config_name)
        if os.path.exists(config_path):
            config_data = self._load_yaml(config_path)
            # 如果配置键不存在，创建一个空字典
            if config_key not in self.config:
                self.config[config_key] = {}
            # 合并配置
            self.config[config_key].update(config_data)
            self.logger.info(f"加载并合并配置文件: {config_path}")
        else:
            self.logger.warning(f"配置文件不存在: {config_path}")
            
    def _load_yaml(self, file_path: str) -> Dict:
        """
        加载YAML文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"加载YAML文件失败: {file_path}, 错误: {str(e)}")
            return {}
            
    def _override_with_env_vars(self) -> None:
        """使用环境变量覆盖配置"""
        for key, value in os.environ.items():
            # 只处理SBS_开头的环境变量
            if key.startswith("SBS_"):
                # 去掉SBS_前缀，转为小写
                config_key = key[4:].lower()
                # 将点分隔符号转换为嵌套字典访问
                parts = config_key.split('_')
                
                # 逐层访问配置字典
                current = self.config
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:
                        # 最后一个部分，设置值
                        current[part] = self._convert_value(value)
                    else:
                        # 非最后部分，确保存在子字典
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                        
    def _convert_value(self, value: str) -> Any:
        """
        转换字符串值为合适的类型
        
        Args:
            value: 字符串值
            
        Returns:
            转换后的值
        """
        # 尝试转换为整数
        try:
            return int(value)
        except ValueError:
            pass
            
        # 尝试转换为浮点数
        try:
            return float(value)
        except ValueError:
            pass
            
        # 处理布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
            
        # 返回原始字符串
        return value
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，使用点分隔表示嵌套结构
            default: 默认值
            
        Returns:
            配置值
        """
        parts = key.split('.')
        current = self.config
        
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键，使用点分隔表示嵌套结构
            value: 配置值
        """
        parts = key.split('.')
        current = self.config
        
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # 最后一个部分，设置值
                current[part] = value
            else:
                # 确保存在子字典
                if part not in current:
                    current[part] = {}
                current = current[part]
                
    def save(self, file_path: str = None) -> None:
        """
        保存配置到文件
        
        Args:
            file_path: 文件路径，默认为主配置文件
        """
        file_path = file_path or os.path.join(self.config_dir, "config.yaml")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"配置已保存到: {file_path}")
        except Exception as e:
            self.logger.error(f"保存配置失败: {file_path}, 错误: {str(e)}")
            
    def export_as_json(self, file_path: str = None) -> None:
        """
        将配置导出为JSON文件
        
        Args:
            file_path: 文件路径，默认为主配置文件同名JSON
        """
        file_path = file_path or os.path.join(self.config_dir, "config.json")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"配置已导出为JSON: {file_path}")
        except Exception as e:
            self.logger.error(f"导出JSON配置失败: {file_path}, 错误: {str(e)}")
            
    def get_full_config(self) -> Dict:
        """
        获取完整配置字典
        
        Returns:
            完整配置字典
        """
        return self.config
        
    def get_section(self, section: str) -> Dict:
        """
        获取特定部分的配置
        
        Args:
            section: 配置部分名称
            
        Returns:
            部分配置字典
        """
        return self.config.get(section, {}) 