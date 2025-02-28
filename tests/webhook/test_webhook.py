import requests
import json
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Webhook URLs
SIGNAL_WEBHOOK = "https://discord.com/api/webhooks/1343455498187571312/NNHdDTLdTE1Lg5PVKojiMM4CT2_8lNcdnpGlBdIHTTQHAfQ-BeZFAHJtlaNErPZkXBDA"
MONITOR_WEBHOOK = "https://discord.com/api/webhooks/1343455788697518133/eMO_2hFoerAliK6eBct00rD5U8k-IXGEeD6-Jg0k30_54A7Uchi-IPdbL3LHPYUnPAkA"
UPLOAD_WEBHOOK = "https://discord.com/api/webhooks/1343455502352388179/G_Vkp50OqNErkWgXAMKlKEECBQ5qOj-g3lkArCiofkdnUN9456uANEHEOEoY_qaFJx-4"

def send_webhook_message(webhook_url, content=None, embed=None, username="SBS系统测试"):
    """发送webhook消息"""
    payload = {
        "username": username
    }
    
    if content:
        payload["content"] = content
        
    if embed:
        payload["embeds"] = [embed]
    
    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        logger.info(f"消息发送成功: {response.status_code}")
        return True
    except Exception as e:
        logger.error(f"发送webhook消息失败: {e}")
        return False

def test_signal_webhook():
    """测试信号webhook"""
    print("测试交易信号webhook...")
    embed = {
        "title": "SBS交易信号分析 - 测试",
        "description": "这是一个测试信号，用于验证webhook功能",
        "color": 0x00ff00,  # 绿色
        "fields": [
            {
                "name": "📊 序列评估",
                "value": "有效性: ✅\n完整度: 95%\n可信度: 87%",
                "inline": False
            },
            {
                "name": "📈 交易信号",
                "value": "方向: ⬆️ 多\n入场区域: 1.0850-1.0870\n止损位: 1.0820\n目标位: 1.0950",
                "inline": False
            },
            {
                "name": "🎯 关键点位",
                "value": "突破点: 1.0840\nPoint 1: 1.0825\nPoint 2: 1.0950\nPoint 3: 1.0830\nPoint 4: 1.0850",
                "inline": False
            }
        ],
        "footer": {
            "text": "SBS系统 - 信号测试"
        }
    }
    
    return send_webhook_message(SIGNAL_WEBHOOK, embed=embed)

def test_monitor_webhook():
    """测试监控webhook"""
    print("测试系统监控webhook...")
    embed = {
        "title": "系统监控状态 - 测试",
        "description": "这是一个测试消息，用于验证监控webhook功能",
        "color": 0xffff00,  # 黄色
        "fields": [
            {
                "name": "系统状态",
                "value": "运行正常",
                "inline": True
            },
            {
                "name": "资源使用",
                "value": "CPU: 45%\n内存: 2.8GB\n磁盘: 60%",
                "inline": True
            },
            {
                "name": "响应时间",
                "value": "平均: 1.2秒",
                "inline": True
            }
        ],
        "footer": {
            "text": "SBS系统 - 监控测试"
        }
    }
    
    return send_webhook_message(MONITOR_WEBHOOK, embed=embed)

def test_upload_webhook():
    """测试上传webhook"""
    print("测试上传通知webhook...")
    content = "✅ 图片上传成功，正在进行分析..."
    
    return send_webhook_message(UPLOAD_WEBHOOK, content=content)

if __name__ == "__main__":
    print("开始测试Discord webhooks...")
    
    # 测试信号webhook
    if test_signal_webhook():
        print("✅ 信号webhook测试成功")
    else:
        print("❌ 信号webhook测试失败")
    
    # 测试监控webhook
    if test_monitor_webhook():
        print("✅ 监控webhook测试成功")
    else:
        print("❌ 监控webhook测试失败")
    
    # 测试上传webhook
    if test_upload_webhook():
        print("✅ 上传webhook测试成功")
    else:
        print("❌ 上传webhook测试失败")
    
    print("测试完成！") 