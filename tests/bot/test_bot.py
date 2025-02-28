import discord
import logging
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Discord配置
TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # 从环境变量获取

class MinimalBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        
    async def on_ready(self):
        logger.info(f"机器人 {self.user} 已准备就绪!")
        logger.info(f"机器人ID: {self.user.id}")
        logger.info("在以下服务器中:")
        for guild in self.guilds:
            logger.info(f"- {guild.name} (ID: {guild.id})")
            
    async def on_message(self, message):
        if message.author == self.user:
            return
            
        logger.info(f"收到消息: {message.content}")
        logger.info(f"来自: {message.author}")
        logger.info(f"在频道: {message.channel.name}")
        
        # 检查是否有图片附件
        if message.attachments:
            for attachment in message.attachments:
                if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    logger.info(f"收到图片: {attachment.filename}")
                    logger.info(f"URL: {attachment.url}")
                    await message.channel.send(f"收到图片: {attachment.filename}")

if __name__ == "__main__":
    if not TOKEN:
        logger.error("未设置DISCORD_BOT_TOKEN环境变量")
        exit(1)
        
    client = MinimalBot()
    logger.info("正在启动机器人...")
    client.run(TOKEN) 