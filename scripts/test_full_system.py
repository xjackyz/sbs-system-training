import requests
import logging
import sys
import os
from pathlib import Path
from src.self_supervised.trainer.self_supervised_trainer import SelfSupervisedTrainer
from src.self_supervised.model.sequence_model import SequenceModel
from src.utils.config import load_config
from src.notification.discord_notifier import get_discord_notifier

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_full_system')

# Discord Webhook URLs
webhooks = {
    '交易信号': 'https://discord.com/api/webhooks/1343455498187571312/NNHdDTLdTE1Lg5PVKojiMM4CT2_8lNcdnpGlBdIHTTQHAfQ-BeZFAHJtlaNErPZkXBDA',
    '系统监控': 'https://discord.com/api/webhooks/1343455788697518133/eMO_2hFoerAliK6eBct00rD5U8k-IXGEeD6-Jg0k30_54A7Uchi-IPdbL3LHPYUnPAkA',
    '上传信号': 'https://discord.com/api/webhooks/1343455502352388179/G_Vkp50OqNErkWgXAMKlKEECBQ5qOj-g3lkArCiofkdnUN9456uANEHEOEoY_qaFJx-4',
    'bug': 'https://discord.com/api/webhooks/1344358842548621393/DHU6CvcChMDyC0qBqTwUbW7zs-kKw65GeNY2qxCBLCLbMfCg-At53wuKjec8yLPKt21D'
}

# 测试消息内容
test_message = "这是一个测试消息，确保Webhook正常工作。"

def send_test_message(webhook_url, title):
    payload = {
        'content': test_message,
        'embeds': [{
            'title': title,
            'description': test_message,
            'color': 0x00ff00
        }]
    }
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 204:
            logger.info(f"{title} - 消息发送成功！")
        else:
            logger.error(f"{title} - 消息发送失败: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"{title} - 发送消息时发生错误: {e}")

def test_self_supervised_model(config_path: str):
    """测试自监督模型"""
    try:
        # 加载配置
        config = load_config(config_path)
        notifier = get_discord_notifier()
        
        # 初始化训练器
        trainer = SelfSupervisedTrainer(config)
        trainer.notifier = notifier
        
        # 发送测试开始通知
        start_message = "🚀 开始自监督模型测试..."
        notifier.send_message_sync(start_message)
        
        # 进行模型训练和验证
        trainer.train()
        validation_metrics = trainer.validate()
        
        # 发送测试完成通知
        completion_message = f"✅ 测试完成！\n验证指标: {validation_metrics}"
        notifier.send_message_sync(completion_message)
        logger.info(completion_message)
    except Exception as e:
        logger.error(f"自监督模型测试过程中发生错误: {e}")
        error_message = f"❌ 测试错误: {str(e)}"
        notifier.send_message_sync(error_message)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        logger.error("请提供配置文件路径作为参数.")
        sys.exit(1)
    config_file_path = sys.argv[1]
    test_self_supervised_model(config_file_path) 