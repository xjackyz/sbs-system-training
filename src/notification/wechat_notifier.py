"""
微信通知处理模块
用于发送微信小程序订阅消息通知
"""

import json
import logging
import requests
from typing import Dict, Optional
from datetime import datetime
from src.utils.exceptions import NotificationError

logger = logging.getLogger(__name__)

class WeChatNotifier:
    """微信通知处理器"""
    
    def __init__(self, config: Dict):
        """
        初始化微信通知处理器
        
        Args:
            config: 配置字典，包含微信相关配置
        """
        self.app_id = config['wechat_app_id']
        self.app_secret = config['wechat_app_secret']
        self.template_id = config['wechat_template_id']
        self.access_token = None
        self.token_expires = 0
        
    def _get_access_token(self) -> str:
        """
        获取微信接口调用凭证
        
        Returns:
            str: 访问令牌
        """
        now = datetime.now().timestamp()
        if self.access_token and now < self.token_expires:
            return self.access_token
            
        url = f"https://api.weixin.qq.com/cgi-bin/token"
        params = {
            "grant_type": "client_credential",
            "appid": self.app_id,
            "secret": self.app_secret
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            result = response.json()
            
            if "access_token" not in result:
                raise NotificationError(f"获取微信access_token失败: {result.get('errmsg', '未知错误')}")
                
            self.access_token = result["access_token"]
            self.token_expires = now + result["expires_in"] - 300  # 提前5分钟过期
            return self.access_token
            
        except Exception as e:
            logger.error(f"获取微信access_token时出错: {str(e)}")
            raise NotificationError(f"获取微信access_token失败: {str(e)}")
    
    def send_notification(self, 
                        openid: str,
                        title: str,
                        content: str,
                        level: str = "INFO",
                        **kwargs) -> bool:
        """
        发送微信通知
        
        Args:
            openid: 接收者的微信openid
            title: 通知标题
            content: 通知内容
            level: 通知级别
            **kwargs: 其他模板数据
            
        Returns:
            bool: 发送是否成功
        """
        try:
            access_token = self._get_access_token()
            url = f"https://api.weixin.qq.com/cgi-bin/message/subscribe/send?access_token={access_token}"
            
            # 构建模板数据
            template_data = {
                "thing1": {"value": title[:20]},  # 标题限制20字符
                "thing2": {"value": content[:20]},  # 内容限制20字符
                "date3": {"value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                "thing4": {"value": level}
            }
            
            # 添加其他自定义数据
            for key, value in kwargs.items():
                if key not in template_data:
                    template_data[key] = {"value": str(value)}
            
            data = {
                "touser": openid,
                "template_id": self.template_id,
                "data": template_data
            }
            
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            
            if result.get("errcode", 0) != 0:
                raise NotificationError(f"发送微信通知失败: {result.get('errmsg', '未知错误')}")
                
            logger.info(f"成功发送微信通知给用户 {openid}")
            return True
            
        except Exception as e:
            logger.error(f"发送微信通知时出错: {str(e)}")
            raise NotificationError(f"发送微信通知失败: {str(e)}")
            
    def send_batch_notification(self, 
                              openids: list,
                              title: str,
                              content: str,
                              level: str = "INFO",
                              **kwargs) -> Dict[str, bool]:
        """
        批量发送微信通知
        
        Args:
            openids: 接收者的微信openid列表
            title: 通知标题
            content: 通知内容
            level: 通知级别
            **kwargs: 其他模板数据
            
        Returns:
            Dict[str, bool]: 每个用户的发送结果
        """
        results = {}
        for openid in openids:
            try:
                success = self.send_notification(openid, title, content, level, **kwargs)
                results[openid] = success
            except Exception as e:
                logger.error(f"向用户 {openid} 发送通知失败: {str(e)}")
                results[openid] = False
        return results 