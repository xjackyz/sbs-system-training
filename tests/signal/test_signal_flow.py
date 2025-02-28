import os
import sys
import logging
import requests
import json
from dotenv import load_dotenv
from pathlib import Path
import base64
import io
from PIL import Image
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_signal_flow")

# 从环境变量获取Discord配置
DISCORD_SIGNAL_WEBHOOK = os.getenv('DISCORD_SIGNAL_WEBHOOK')
DISCORD_MONITOR_WEBHOOK = os.getenv('DISCORD_MONITOR_WEBHOOK')
DISCORD_UPLOAD_WEBHOOK = os.getenv('DISCORD_UPLOAD_WEBHOOK')

def send_webhook_message(webhook_url, content=None, embeds=None, file_path=None):
    """发送webhook消息"""
    if not webhook_url:
        logger.error("未提供webhook URL")
        return False
        
    # 准备multipart/form-data数据
    payload = {}
    if content:
        payload['content'] = content
    if embeds:
        payload['embeds'] = embeds
        
    # 文件部分
    files = None
    if file_path and os.path.exists(file_path):
        files = {
            'file': (os.path.basename(file_path), open(file_path, 'rb'))
        }
        
    try:
        # 如果只有JSON数据
        if not files:
            response = requests.post(
                webhook_url, 
                json=payload,
                timeout=10
            )
        # 如果有文件
        else:
            # 将JSON转为字符串并添加到表单数据中
            form_data = {
                'payload_json': json.dumps(payload)
            }
            response = requests.post(
                webhook_url, 
                data=form_data,
                files=files,
                timeout=10
            )
            
        # 关闭文件
        if files:
            files['file'][1].close()
            
        if response.status_code == 204:
            logger.info(f"消息发送成功: {webhook_url}")
            return True
        else:
            logger.error(f"发送消息失败，状态码: {response.status_code}, 响应: {response.text}")
            return False
    except Exception as e:
        logger.error(f"发送webhook消息时出错: {e}")
        return False

def test_signal_webhook():
    """测试信号webhooks"""
    if not DISCORD_SIGNAL_WEBHOOK:
        logger.error("缺少DISCORD_SIGNAL_WEBHOOK环境变量")
        return False
        
    # 创建测试嵌入消息
    embeds = [{
        "title": "测试交易信号",
        "description": "这是一个测试交易信号",
        "color": 0x00ff00,
        "fields": [
            {
                "name": "📊 序列评估",
                "value": "有效性: ✅\n完整度: 85%\n可信度: 75%",
                "inline": False
            },
            {
                "name": "📈 交易信号",
                "value": "方向: ⬆️ 多\n入场区域: 1.2345-1.2350\n止损位: 1.2300\n目标位: 1.2400",
                "inline": False
            },
            {
                "name": "🎯 关键点位",
                "value": "突破点: 1.2340\nPoint 1: 1.2320\nPoint 2: 1.2380\nPoint 3: 1.2315\nPoint 4: 1.2345",
                "inline": False
            }
        ],
        "timestamp": "2024-02-25T10:00:00.000Z"
    }]
    
    return send_webhook_message(DISCORD_SIGNAL_WEBHOOK, embeds=embeds)

def test_upload_webhook():
    """测试上传图像到Discord的功能"""
    logger.info("测试上传webhook...")
    try:
        webhook_url = os.getenv('DISCORD_UPLOAD_WEBHOOK')
        if not webhook_url:
            logger.error("环境变量DISCORD_UPLOAD_WEBHOOK未设置")
            return False
        
        with open('temp/test_chart.png', 'rb') as f:
            files = {'file': ('test_chart.png', f, 'image/png')}
            payload = {'content': '正在测试上传图表功能'}
            response = requests.post(webhook_url, data=payload, files=files)
            
        if response.status_code == 200 or response.status_code == 204:
            logger.info(f"消息发送成功: {webhook_url}")
            return True
        else:
            logger.error(f"发送消息失败，状态码: {response.status_code}, 响应: {response.text}")
            return False
    except Exception as e:
        logger.error(f"测试上传webhook时出错: {str(e)}")
        return False

def test_monitor_webhook():
    """测试监控webhook"""
    if not DISCORD_MONITOR_WEBHOOK:
        logger.error("缺少DISCORD_MONITOR_WEBHOOK环境变量")
        return False
        
    # 创建测试嵌入消息
    embeds = [{
        "title": "系统监控测试",
        "description": "这是一个系统监控测试消息",
        "color": 0xffaa00,
        "fields": [
            {
                "name": "系统状态",
                "value": "✅ 运行正常",
                "inline": True
            },
            {
                "name": "资源使用",
                "value": "🖥️ CPU: 25%\n💾 内存: 2.5GB",
                "inline": True
            }
        ],
        "timestamp": "2024-02-25T10:00:00.000Z"
    }]
    
    return send_webhook_message(DISCORD_MONITOR_WEBHOOK, embeds=embeds)

def test_image_processing():
    """测试图像处理功能"""
    try:
        from src.image.processor import ImageProcessor
        
        # 测试图表路径
        test_chart = "temp/test_chart.png"
        if not os.path.exists(test_chart):
            logger.error(f"测试图表不存在: {test_chart}")
            return False
            
        # 创建图像处理器
        processor = ImageProcessor()
        
        # 检查图像质量
        quality_ok = processor.check_image_quality(test_chart)
        logger.info(f"图像质量检查: {'通过' if quality_ok else '未通过'}")
        
        # 预处理图像
        processed_image = processor.preprocess_image(test_chart)
        if not processed_image:
            logger.error("预处理图像失败")
            return False
        logger.info(f"预处理图像成功: {processed_image}")
        
        # 裁剪图表区域
        cropped_image = processor.crop_chart_area(processed_image)
        if not cropped_image:
            logger.error("裁剪图表区域失败")
            return False
        logger.info(f"裁剪图表区域成功: {cropped_image}")
        
        return True
    except Exception as e:
        logger.error(f"测试图像处理时出错: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始测试Discord交易信号流程")
    
    # 测试监控webhook
    logger.info("测试监控webhook...")
    monitor_result = test_monitor_webhook()
    logger.info(f"监控webhook测试: {'成功' if monitor_result else '失败'}")
    
    # 测试图像处理
    logger.info("测试图像处理功能...")
    image_result = test_image_processing()
    logger.info(f"图像处理测试: {'成功' if image_result else '失败'}")
    
    # 测试上传webhook
    logger.info("测试上传webhook...")
    upload_result = test_upload_webhook()
    logger.info(f"上传webhook测试: {'成功' if upload_result else '失败'}")
    
    # 测试信号webhook
    logger.info("测试信号webhook...")
    signal_result = test_signal_webhook()
    logger.info(f"信号webhook测试: {'成功' if signal_result else '失败'}")
    
    # 汇总结果
    overall_result = all([monitor_result, image_result, upload_result, signal_result])
    logger.info(f"测试完成，总体结果: {'成功' if overall_result else '失败'}")
    
    return 0 if overall_result else 1

if __name__ == "__main__":
    sys.exit(main()) 