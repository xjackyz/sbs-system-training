"""
配置验证模块
使用 Pydantic 进行配置验证
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator, EmailStr
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = 'development'
    PRODUCTION = 'production'
    TESTING = 'testing'

class DeviceConfig(BaseModel):
    use_gpu: bool = Field(True, description="是否使用GPU")
    num_workers: int = Field(4, description="工作线程数")

class ModelConfig(BaseModel):
    path: str = Field(..., description="模型路径")
    batch_size: int = Field(32, description="批处理大小")
    type: str = Field('transformer', description="模型类型")
    hidden_size: int = Field(768, description="隐藏层大小")
    num_hidden_layers: int = Field(12, description="隐藏层数量")
    num_attention_heads: int = Field(12, description="注意力头数量")
    intermediate_size: int = Field(3072, description="中间层大小")
    hidden_dropout_prob: float = Field(0.1, description="隐藏层dropout概率")
    attention_probs_dropout_prob: float = Field(0.1, description="注意力dropout概率")

class ApiConfig(BaseModel):
    tradingview_key: str = Field(..., description="TradingView API密钥")
    discord_webhook: str = Field(..., description="Discord Webhook URL")

class DatabaseConfig(BaseModel):
    host: str = Field('localhost', description="数据库主机")
    port: int = Field(5432, description="数据库端口")
    name: str = Field('sbs_system', description="数据库名称")
    user: str = Field(..., description="数据库用户名")
    password: str = Field(..., description="数据库密码")

class CacheConfig(BaseModel):
    redis_host: str = Field('localhost', description="Redis主机")
    redis_port: int = Field(6379, description="Redis端口")
    redis_password: Optional[str] = Field(None, description="Redis密码")

class NetworkConfig(BaseModel):
    mirror_url: str = Field('https://api.example.com', description="镜像URL")
    verify_ssl: bool = Field(True, description="是否验证SSL")
    timeout: int = Field(30, description="超时时间（秒）")

class SecurityConfig(BaseModel):
    secret_key: str = Field(..., description="密钥")
    jwt_secret: str = Field(..., description="JWT密钥")
    encryption_key: str = Field(..., description="加密密钥")

class MonitoringConfig(BaseModel):
    prometheus_port: int = Field(9090, description="Prometheus端口")
    grafana_port: int = Field(3000, description="Grafana端口")

class BackupConfig(BaseModel):
    path: str = Field('/path/to/backup', description="备份路径")
    retention_days: int = Field(7, description="备份保留天数")

class NotificationConfig(BaseModel):
    enabled: bool = Field(True, description="是否启用通知")
    level: str = Field('INFO', description="通知级别")
    email_host: str = Field('smtp.qq.com', description="SMTP服务器地址")
    email_port: int = Field(587, description="SMTP服务器端口")
    email_use_tls: bool = Field(True, description="是否使用TLS")
    email_host_user: EmailStr = Field(..., description="发件人邮箱")
    email_host_password: str = Field(..., description="邮箱授权码")
    email_from: EmailStr = Field(..., description="发件人显示名称")
    email_to: List[EmailStr] = Field(..., description="收件人邮箱列表")
    email_subject_prefix: str = Field("[SBS系统]", description="邮件主题前缀")

class TrainingConfig(BaseModel):
    gpu_settings: Dict = Field(..., description="GPU相关设置")
    baseline_validation: Dict = Field(..., description="基准验证配置")
    initial_training: Dict = Field(..., description="初始训练配置")
    validation: Dict = Field(..., description="验证配置")
    data: Dict = Field(..., description="数据配置")
    optimizer: Dict = Field(..., description="优化器配置")
    process: Dict = Field(..., description="训练过程配置")

class SignalConfig(BaseModel):
    thresholds: Dict = Field(..., description="信号阈值配置")
    min_interval: int = Field(24, description="最小信号间隔（小时）")

class SystemConfig(BaseModel):
    environment: Environment = Field(Environment.DEVELOPMENT, description="环境类型")
    debug: bool = Field(True, description="是否开启调试模式")
    log_level: str = Field('INFO', description="日志级别")
    device: DeviceConfig
    model: ModelConfig
    api: ApiConfig
    database: DatabaseConfig
    cache: CacheConfig
    network: NetworkConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    backup: BackupConfig
    notification: NotificationConfig
    training: TrainingConfig
    signal: SignalConfig

    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'日志级别必须是以下之一: {valid_levels}')
        return v.upper()

    @validator('level')
    def validate_notification_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'通知级别必须是以下之一: {valid_levels}')
        return v.upper()

    class Config:
        validate_assignment = True 