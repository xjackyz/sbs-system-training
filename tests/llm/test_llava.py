"""LLaVA测试脚本"""
import os
import json
from pathlib import Path
import asyncio
from app.core.llava.analyzer import LLaVAAnalyzer
from config.sbs_prompt import SBS_PROMPT

async def test_llava():
    """测试LLaVA模型"""
    try:
        # 加载配置
        with open('config/config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 初始化分析器
        analyzer = LLaVAAnalyzer(config)
        
        # 获取测试图片列表
        image_dir = Path('训练内容/SBS_nolabel')
        image_files = [f for f in image_dir.glob('*.png') if not f.name.endswith('(1).png')]
        
        # 测试每张图片
        for image_file in image_files:
            print(f"\n正在分析图片: {image_file.name}")
            
            # 分析图片
            result = await analyzer.analyze_image(
                str(image_file),
                SBS_PROMPT
            )
            
            if result['success']:
                print("\n分析结果:")
                print(result['response'])
            else:
                print(f"分析失败: {result['error']}")
                
            # 等待一下，避免GPU过载
            await asyncio.sleep(2)
            
    except Exception as e:
        print(f"测试过程出错: {e}")

if __name__ == "__main__":
    asyncio.run(test_llava())
