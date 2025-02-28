import os
import sys
from pathlib import Path
import json
import logging
from dotenv import load_dotenv
import discord
from discord.ext import commands
import tempfile

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
load_dotenv()

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_full_bot')

# 机器人配置
TOKEN = os.getenv("DISCORD_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")  # 从环境变量加载Token
UPLOAD_CHANNEL_ID = None  # 将在机器人启动后从用户输入获取

class ConfigHelper:
    @staticmethod
    def load_config():
        """加载配置"""
        config_path = Path('config/config.json')
        if not config_path.exists():
            logger.warning("配置文件不存在，使用默认配置")
            return {
                "discord": {
                    "token": TOKEN,
                    "webhooks": {
                        "signal": "YOUR_WEBHOOK_URL_HERE",
                        "monitor": "YOUR_WEBHOOK_URL_HERE",
                        "upload": "YOUR_WEBHOOK_URL_HERE"
                    }
                }
            }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 替换Token为我们的测试token
            if "discord" in config:
                config["discord"]["token"] = TOKEN
                
            return config
        except Exception as e:
            logger.error(f"加载配置文件出错: {e}")
            sys.exit(1)

class TestFullBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        # 加载配置
        self.config = ConfigHelper.load_config()
        
        # 创建临时目录
        self.temp_dir = Path('temp')
        self.temp_dir.mkdir(exist_ok=True)
        
    async def on_ready(self):
        """机器人就绪事件"""
        logger.info(f"机器人 {self.user} 已准备就绪!")
        print(f"\n机器人 {self.user} 已成功连接到Discord!")
        print("请在Discord中输入上传频道的ID...")
        
    async def on_message(self, message):
        """消息处理"""
        global UPLOAD_CHANNEL_ID
        
        # 忽略自己的消息
        if message.author == self.user:
            return
            
        # 如果我们还没有设置上传频道ID，任何消息都认为是设置上传频道ID
        if not UPLOAD_CHANNEL_ID and message.content.isdigit():
            UPLOAD_CHANNEL_ID = int(message.content)
            await message.channel.send(f"上传频道ID已设置为: {UPLOAD_CHANNEL_ID}")
            logger.info(f"上传频道ID已设置为: {UPLOAD_CHANNEL_ID}")
            return
            
        # 处理命令
        await self.process_commands(message)
        
        # 处理图片附件（如果上传频道ID已设置）
        if UPLOAD_CHANNEL_ID and message.channel.id == UPLOAD_CHANNEL_ID:
            if message.attachments:
                for attachment in message.attachments:
                    if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        await self.process_image(message, attachment)
                        
    async def process_image(self, message, attachment):
        """处理图片附件"""
        try:
            # 发送确认消息
            await message.channel.send(f"收到图片: {attachment.filename}，开始处理...")
            
            # 下载图片
            temp_path = self.temp_dir / f"temp_{attachment.filename}"
            await attachment.save(temp_path)
            
            # 发送分析中消息
            analysis_msg = await message.channel.send("图像分析中，请稍候...")
            
            # 模拟分析过程
            await self.simulate_image_analysis(message, temp_path, analysis_msg)
            
        except Exception as e:
            logger.error(f"处理图片时出错: {e}")
            await message.channel.send(f"处理图片时出错: {e}")
            
    async def simulate_image_analysis(self, message, image_path, analysis_msg):
        """模拟图像分析过程"""
        import asyncio
        
        # 步骤1: 图像质量检查
        await analysis_msg.edit(content="步骤1/4: 图像质量检查中...")
        await asyncio.sleep(1)
        
        # 步骤2: 图像预处理
        await analysis_msg.edit(content="步骤2/4: 图像预处理中...")
        await asyncio.sleep(1.5)
        
        # 步骤3: 图表区域裁剪
        await analysis_msg.edit(content="步骤3/4: 图表区域裁剪中...")
        await asyncio.sleep(1.5)
        
        # 步骤4: LLaVA模型分析
        await analysis_msg.edit(content="步骤4/4: LLaVA模型分析中...")
        await asyncio.sleep(2)
        
        # 分析完成，发送结果
        await analysis_msg.edit(content="✅ 图像分析完成!")
        
        # 发送模拟结果
        embed = discord.Embed(
            title="SBS交易信号分析",
            description=f"分析来自 {message.author.mention} 的图表",
            color=0x00ff00
        )
        
        # 添加序列评估
        embed.add_field(
            name="📊 序列评估",
            value="有效性: ✅\n完整度: 92%\n可信度: 85%",
            inline=False
        )
        
        # 添加交易信号
        embed.add_field(
            name="📈 交易信号",
            value="方向: ⬆️ 多\n入场区域: 1.0850-1.0870\n止损位: 1.0820\n目标位: 1.0950",
            inline=False
        )
        
        # 添加关键点位
        embed.add_field(
            name="🎯 关键点位",
            value="突破点: 1.0840\nPoint 1: 1.0825\nPoint 2: 1.0950\nPoint 3: 1.0830\nPoint 4: 1.0850",
            inline=False
        )
        
        # 添加趋势分析
        embed.add_field(
            name="📊 趋势分析",
            value="SMA20: 上升\nSMA200: 上升\n整体趋势: 上升",
            inline=False
        )
        
        # 添加风险评估
        embed.add_field(
            name="⚠️ 风险评估",
            value="风险等级: 中\n主要风险: 市场波动性较高，建议谨慎入场",
            inline=False
        )
        
        # 设置页脚
        embed.set_footer(text=f"SBS系统 | {message.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 发送分析结果
        await message.channel.send(embed=embed)
        
        # 清理临时文件
        if os.path.exists(image_path):
            os.remove(image_path)

def main():
    """主函数"""
    try:
        bot = TestFullBot()
        bot.run(TOKEN)
    except Exception as e:
        logger.error(f"运行机器人失败: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 