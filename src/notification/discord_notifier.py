"""
Discord通知模块
"""
import discord
import aiohttp
from discord import Webhook
from datetime import datetime
import pytz
from typing import Dict, Optional, List, Union, Any
from dataclasses import dataclass
import logging
import os
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
import requests
import time

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('discord_notifier')

# 设置北京时区
beijing_tz = pytz.timezone('Asia/Shanghai')

@dataclass
class DiscordConfig:
    """Discord配置"""
    webhooks: Dict[str, str]  # 不同类型通知的webhook URLs
    username: str = "SBS Training System"  # 机器人用户名
    avatar_url: Optional[str] = None  # 机器人头像URL
    embed_color: int = 0x00ff00  # 嵌入消息颜色
    max_retries: int = 3  # 最大重试次数
    retry_delay: int = 5  # 重试延迟（秒）


class DiscordNotifier:
    """Discord通知器"""
    
    def __init__(self, discord_config: Optional[DiscordConfig] = None):
        """
        初始化Discord通知器
        
        Args:
            discord_config: Discord配置
        """
        self.config = discord_config or self._load_default_config()
        self.logger = logging.getLogger(__name__)
        
    def _load_default_config(self) -> DiscordConfig:
        """加载默认配置"""
        return DiscordConfig(
            webhooks={
                'monitor': os.getenv('DISCORD_MONITOR_WEBHOOK', ''),
                'signal': os.getenv('DISCORD_SIGNAL_WEBHOOK', ''),
                'debug': 'https://discord.com/api/webhooks/1344358842548621393/DHU6CvcChMDyC0qBqTwUbW7zs-kKw65GeNY2qxCBLCLbMfCg-At53wuKjec8yLPKt21D'
            },
            username=os.getenv('DISCORD_BOT_USERNAME', 'SBS Training System'),
            avatar_url=os.getenv('DISCORD_BOT_AVATAR', None)
        )
    
    async def send_signal(self, data: Dict) -> bool:
        """
        发送信号消息
        
        Args:
            data: 信号数据
        
        Returns:
            bool: 是否发送成功
        """
        webhook_url = self.config.webhooks.get('signal')
        if not webhook_url:
            self.logger.error("未配置信号webhook URL")
            return False
            
        title = "🎯 新交易信号"
        return await self._send_message(webhook_url, title, data)
    
    async def send_monitor_message(self, data: Dict) -> bool:
        """
        发送监控消息
        
        Args:
            data: 监控数据
        
        Returns:
            bool: 是否发送成功
        """
        webhook_url = self.config.webhooks.get('monitor')
        if not webhook_url:
            self.logger.error("未配置监控webhook URL")
            return False
            
        title = "📊 训练监控"
        return await self._send_message(webhook_url, title, data)
        
    async def send_status_update(self, status: str, details: Dict = None) -> bool:
        """
        发送状态更新
        
        Args:
            status: 状态消息
            details: 详细信息
        
        Returns:
            bool: 是否发送成功
        """
        webhook_url = self.config.webhooks.get('monitor')
        if not webhook_url:
            self.logger.error("未配置监控webhook URL")
            return False
            
        data = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            **(details or {})
        }
        
        title = "ℹ️ 状态更新"
        return await self._send_message(webhook_url, title, data)
        
    async def _send_message(self, webhook_url: str, title: str, data: Dict) -> bool:
        """
        发送消息
        
        Args:
            webhook_url: webhook URL
            title: 消息标题
            data: 消息数据
        
        Returns:
            bool: 是否发送成功
        """
        embed = {
            'title': title,
            'color': self.config.embed_color,
            'timestamp': datetime.now().isoformat(),
            'fields': []
        }
        
        # 将数据转换为Discord嵌入字段
        for key, value in data.items():
            embed['fields'].append({
                'name': key.replace('_', ' ').title(),
                'value': str(value),
                'inline': True
            })
            
        payload = {
            'username': self.config.username,
            'avatar_url': self.config.avatar_url,
            'embeds': [embed]
        }
        
        # 尝试发送消息
        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(webhook_url, json=payload) as response:
                        if response.status == 204:
                            return True
                        self.logger.error(f"Discord API响应错误: {response.status}")
                        return False
            except Exception as e:
                self.logger.error(f"发送Discord消息失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    
        return False
            
    # 同步方法
    def send_message_sync(self, message: str, webhook_type: str = 'monitor') -> bool:
        """
        同步发送消息
        
        Args:
            message: 消息内容
            webhook_type: webhook类型
            
        Returns:
            bool: 是否发送成功
        """
        webhook_url = self.config.webhooks.get(webhook_type)
        if not webhook_url:
            self.logger.error(f"未配置 {webhook_type} webhook URL")
            return False
            
        payload = {
            'username': self.config.username,
            'avatar_url': self.config.avatar_url,
            'content': message
        }
        
        # 尝试发送消息
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(webhook_url, json=payload)
                if response.status_code == 204:
                    return True
                self.logger.error(f"Discord API响应错误: {response.status_code}")
                return False
            except Exception as e:
                self.logger.error(f"发送Discord消息失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    
        return False


def get_discord_notifier() -> DiscordNotifier:
    """
    获取Discord通知器实例
    
    Returns:
        DiscordNotifier: 通知器实例
    """
    config = DiscordConfig(
        webhooks={
            'monitor': os.getenv('DISCORD_MONITOR_WEBHOOK'),
            'signal': os.getenv('DISCORD_SIGNAL_WEBHOOK'),
            'debug': 'https://discord.com/api/webhooks/1344358842548621393/DHU6CvcChMDyC0qBqTwUbW7zs-kKw65GeNY2qxCBLCLbMfCg-At53wuKjec8yLPKt21D'
        },
        username=os.getenv('DISCORD_BOT_USERNAME', 'SBS Training System'),
        avatar_url=os.getenv('DISCORD_BOT_AVATAR')
    )
    
    return DiscordNotifier(config) 