import discord
import logging
import os
import sys

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Discord配置
TOKEN = os.getenv("DISCORD_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")  # 从环境变量加载Token

class TestBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        
    async def on_ready(self):
        logger.info(f"机器人 {self.user} 已准备就绪!")
        print(f"机器人 {self.user} 已成功连接到Discord!")
        
    async def on_message(self, message):
        if message.author == self.user:
            return
            
        logger.info(f"收到消息: {message.content}")
        await message.channel.send(f"收到消息: {message.content}")

if __name__ == "__main__":
    client = TestBot()
    try:
        client.run(TOKEN)
    except Exception as e:
        logger.error(f"运行机器人时出错: {e}")
        sys.exit(1) 