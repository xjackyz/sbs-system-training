import requests
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_system')

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

if __name__ == '__main__':
    for title, url in webhooks.items():
        send_test_message(url, title) 