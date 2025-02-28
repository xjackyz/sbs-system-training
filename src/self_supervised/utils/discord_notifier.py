from typing import List, Dict, Optional
import requests
import os
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from dataclasses import dataclass

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
    def __init__(self, webhook_url: str, logger):
        self.webhook_url = webhook_url
        self.logger = logger

    def send_message(self, message: str, files: List[str] = None):
        """发送消息到 Discord
        Args:
            message: 消息内容
            files: 附加文件列表
        """
        if not self.webhook_url:
            self.logger.warning("未设置 webhook_url，无法发送消息")
            return
        payload = {'content': message}
        if files:
            for file_path in files:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        file_obj = MultipartEncoder(fields={'file': (os.path.basename(file_path), f, 'application/octet-stream')})
                        payload['files'] = [file_obj]
                else:
                    self.logger.warning(f"文件不存在: {file_path}")
        try:
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()  # 检查请求是否成功
            self.logger.info("消息发送成功")
        except Exception as e:
            self.logger.error(f"发送消息失败: {str(e)}")