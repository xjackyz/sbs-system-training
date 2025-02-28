from typing import Dict, Any, Optional
import yaml
import json
import os
from pathlib import Path
import logging
from pydantic import BaseModel, validator
from datetime import datetime

class TrainingConfig(BaseModel):
    """训练配置模型"""
    batch_size: int
    learning_rate: float
    num_epochs: int
    gradient_accumulation_steps: int
    mixed_precision: str
    max_seq_length: int
    model_parallel: bool
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError('batch_size必须大于0')
        return v
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if v <= 0:
            raise ValueError('learning_rate必须大于0')
        return v

class SystemConfig(BaseModel):
    """系统配置模型"""
    log_level: str
    data_dir: str
    model_dir: str
    checkpoint_dir: str
    max_gpu_memory: float
    enable_tensorboard: bool
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level必须是以下之一: {valid_levels}')
        return v.upper()

class MonitorConfig(BaseModel):
    """监控配置模型"""
    metrics_interval: int
    history_size: int
    alert_thresholds: Dict[str, float]
    
    @validator('metrics_interval')
    def validate_metrics_interval(cls, v):
        if v < 1:
            raise ValueError('metrics_interval必须大于等于1')
        return v

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 加载配置
        self.training_config = self._load_config('training_config.yaml', TrainingConfig)
        self.system_config = self._load_config('system_config.yaml', SystemConfig)
        self.monitor_config = self._load_config('monitor_config.yaml', MonitorConfig)
        
        # 配置历史记录
        self.config_history = []
        
    def validate_config(self, config: Dict) -> bool:
        """验证配置文件的完整性和正确性"""
        required_fields = ['model', 'training', 'data', 'notification']
        return all(field in config for field in required_fields)

    def _load_config(self, filename: str, model_class) -> BaseModel:
        """加载并验证配置文件"""
        file_path = self.config_dir / filename
        
        # 如果文件不存在，创建默认配置
        if not file_path.exists():
            default_config = self._create_default_config(model_class)
            self._save_config(filename, default_config)
            return default_config
        
        try:
            with open(file_path) as f:
                config_dict = yaml.safe_load(f)
            if not self.validate_config(config_dict):
                raise ValueError("配置文件缺少必需字段")
            return model_class(**config_dict)
        except Exception as e:
            logging.error(f"加载配置文件 {filename} 失败: {e}")
            default_config = self._create_default_config(model_class)
            self._save_config(filename, default_config)
            return default_config
    
    def _create_default_config(self, model_class) -> BaseModel:
        """创建默认配置"""
        if model_class == TrainingConfig:
            return TrainingConfig(
                batch_size=4,
                learning_rate=2e-5,
                num_epochs=10,
                gradient_accumulation_steps=16,
                mixed_precision='bf16',
                max_seq_length=512,
                model_parallel=False
            )
        elif model_class == SystemConfig:
            return SystemConfig(
                log_level='INFO',
                data_dir='data',
                model_dir='models',
                checkpoint_dir='checkpoints',
                max_gpu_memory=0.9,
                enable_tensorboard=True
            )
        elif model_class == MonitorConfig:
            return MonitorConfig(
                metrics_interval=60,
                history_size=3600,
                alert_thresholds={
                    'gpu_temperature': 80,
                    'gpu_memory_percent': 90,
                    'cpu_usage': 90,
                    'memory_usage': 90
                }
            )
    
    def _save_config(self, filename: str, config: BaseModel):
        """保存配置到文件"""
        file_path = self.config_dir / filename
        with open(file_path, 'w') as f:
            yaml.dump(config.dict(), f)
    
    def update_training_config(self, updates: Dict[str, Any]):
        """更新训练配置"""
        current_dict = self.training_config.dict()
        current_dict.update(updates)
        
        # 验证新配置
        new_config = TrainingConfig(**current_dict)
        
        # 保存历史记录
        self.config_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'training_config',
            'changes': updates
        })
        
        # 更新配置
        self.training_config = new_config
        self._save_config('training_config.yaml', new_config)
    
    def update_system_config(self, updates: Dict[str, Any]):
        """更新系统配置"""
        current_dict = self.system_config.dict()
        current_dict.update(updates)
        
        new_config = SystemConfig(**current_dict)
        
        self.config_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'system_config',
            'changes': updates
        })
        
        self.system_config = new_config
        self._save_config('system_config.yaml', new_config)
    
    def update_monitor_config(self, updates: Dict[str, Any]):
        """更新监控配置"""
        current_dict = self.monitor_config.dict()
        current_dict.update(updates)
        
        new_config = MonitorConfig(**current_dict)
        
        self.config_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'monitor_config',
            'changes': updates
        })
        
        self.monitor_config = new_config
        self._save_config('monitor_config.yaml', new_config)
    
    def get_config_history(self) -> list:
        """获取配置更新历史"""
        return self.config_history
    
    def export_configs(self, output_dir: str):
        """导出所有配置"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 导出当前配置
        configs = {
            'training_config': self.training_config.dict(),
            'system_config': self.system_config.dict(),
            'monitor_config': self.monitor_config.dict()
        }
        
        with open(output_path / 'configs.json', 'w') as f:
            json.dump(configs, f, indent=2)
        
        # 导出历史记录
        with open(output_path / 'config_history.json', 'w') as f:
            json.dump(self.config_history, f, indent=2)
    
    def import_configs(self, config_file: str):
        """导入配置"""
        try:
            with open(config_file) as f:
                configs = json.load(f)
            
            self.training_config = TrainingConfig(**configs['training_config'])
            self.system_config = SystemConfig(**configs['system_config'])
            self.monitor_config = MonitorConfig(**configs['monitor_config'])
            
            # 保存导入的配置
            self._save_config('training_config.yaml', self.training_config)
            self._save_config('system_config.yaml', self.system_config)
            self._save_config('monitor_config.yaml', self.monitor_config)
            
        except Exception as e:
            logging.error(f"导入配置失败: {e}")
            raise