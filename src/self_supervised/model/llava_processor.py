import os
import torch
import logging
from PIL import Image
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from tqdm import tqdm
import pandas as pd
import time
import random

from transformers import AutoProcessor, AutoModelForCausalLM

class LlavaProcessor:
    """
    LLaVA模型处理类，用于处理图像并生成分析结果
    """
    
    def __init__(self, 
                 model_path: str = "models/llava-sbs", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 config: Dict = None):
        """
        初始化LLaVA模型处理器
        
        Args:
            model_path: 模型路径
            device: 运行设备
            config: 配置参数
        """
        self.logger = logging.getLogger('llava_processor')
        self.logger.info(f"初始化LLaVA处理器，使用设备: {device}")
        
        # 默认配置
        default_config = {
            'max_new_tokens': 512,
            'temperature': 0.2,
            'top_p': 0.9,
            'top_k': 40,
            'num_beams': 1,
            'prompt_template': """
作为专业的金融图表分析专家，请分析这张K线图，识别和描述SBS（Structure Breakout Sequence）交易序列。

请识别以下关键点位和特征：

1. 市场结构：
   - 是否出现明显的支撑/阻力位突破
   - 突破前后的市场结构变化
   - K线形态和走势特征

2. 关键点位：
   - 突破点：市场结构被实体K线突破的位置
   - Point 1：突破后第一次回调形成的高点（做多）或低点（做空）
   - Point 2：由点1创造的最高高点（做多）或最低低点（做空）
   - Point 3：点1附近的流动性获取点
   - Point 4：点3附近的确认点

3. 形态特征：
   - 回调的深度和形态
   - 是否形成双顶/双底
   - 是否出现流动性获取
   - 是否有SCE（Single Candle Entry）信号

请以JSON格式返回以下信息：
{
    "market_structure": {
        "has_breakout": true/false,           // 是否出现有效突破
        "breakout_description": str,          // 突破形态描述
        "structure_change": str              // 市场结构变化描述
    },
    "key_points": {
        "breakout": {"price": float, "time": str},    // 突破点
        "point1": {"price": float, "time": str},      // Point 1
        "point2": {"price": float, "time": str},      // Point 2
        "point3": {"price": float, "time": str},      // Point 3
        "point4": {"price": float, "time": str}       // Point 4
    },
    "pattern_features": {
        "pullback_depth": float,             // 回调深度
        "double_pattern": str,               // 双顶/双底形态
        "liquidation": str,                  // 流动性获取描述
        "sce_signal": str                    // SCE信号描述
    },
    "trading_signal": {
        "direction": "long"/"short",         // 交易方向
        "entry_zone": {                      // 入场区域
            "upper": float,
            "lower": float
        },
        "stop_loss": float,                 // 止损位
        "target": float                     // 目标位
    }
}

注意事项：
1. 所有价格水平需精确到小数点后4位
2. 时间格式使用ISO 8601标准
3. 如果某些点位无法确定，将其设置为null
4. 只返回JSON格式数据，不要包含其他内容
""",
            'save_path': 'output/llava_results'
        }
        
        # 更新配置
        self.config = default_config
        if config:
            self.config.update(config)
        
        # 确保保存路径存在
        os.makedirs(self.config['save_path'], exist_ok=True)
        
        # 初始化设备
        self.device = device
        
        # 加载模型和处理器
        try:
            self.logger.info(f"加载LLaVA模型: {model_path}")
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)
            
            # 设置为评估模式
            self.model.eval()
            self.logger.info("LLaVA模型加载完成")
            
        except Exception as e:
            self.logger.error(f"加载LLaVA模型失败: {str(e)}")
            raise
    
    def process_image(self, image_path: str, custom_prompt: str = None) -> Dict[str, Any]:
        """
        处理单张图像并生成分析结果
        
        Args:
            image_path: 图像文件路径
            custom_prompt: 自定义提示，如果不提供则使用默认提示
            
        Returns:
            包含分析结果的字典
        """
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 准备提示
            prompt = custom_prompt if custom_prompt else self.config['prompt_template']
            
            # 准备输入
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成输出
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config['max_new_tokens'],
                    temperature=self.config['temperature'],
                    top_p=self.config['top_p'],
                    top_k=self.config['top_k'],
                    num_beams=self.config['num_beams'],
                    do_sample=self.config['temperature'] > 0
                )
            
            # 解码输出
            generated_text = self.processor.decode(output[0], skip_special_tokens=True)
            
            # 提取提示部分和回答部分
            response = generated_text.split(prompt)[-1].strip()
            
            # 提取SBS信号信息
            signal_info = self._extract_signal_info(response)
            
            # 准备结果
            result = {
                'image_path': image_path,
                'prompt': prompt,
                'full_response': response,
                'signal_info': signal_info,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理图像失败: {str(e)}")
            raise
    
    def _extract_signal_info(self, response: str) -> Dict[str, Any]:
        """
        从模型响应中提取SBS信号信息
        
        Args:
            response: 模型响应文本
            
        Returns:
            包含信号信息的字典
        """
        # 初始化信号信息
        signal_info = {
            'has_signal': False,
            'is_complete': False,
            'points': {
                'point1': None,
                'point2': None,
                'point3': None,
                'point4': None,
                'point5': None
            },
            'entry_point': None,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 0.0
        }
        
        # 检查是否存在信号
        if "SBS" in response and ("信号" in response or "序列" in response):
            signal_info['has_signal'] = True
            
            # 检查是否完整
            if all(f"点{i}" in response or f"点 {i}" in response for i in range(1, 6)):
                signal_info['is_complete'] = True
            
            # 提取置信度
            confidence_keywords = ["置信度", "确信", "可信度", "确定性", "概率"]
            for keyword in confidence_keywords:
                if keyword in response:
                    # 查找关键词附近的数字
                    pos = response.find(keyword)
                    sub_text = response[max(0, pos-20):pos+20]
                    confidence_values = [float(n)/100 for n in sub_text.split() if n.replace('.', '').isdigit() and float(n) <= 100]
                    if confidence_values:
                        signal_info['confidence'] = max(confidence_values)
                        break
            
            # 如果没有找到具体置信度，但确认有信号，给一个默认值
            if signal_info['confidence'] == 0.0 and signal_info['has_signal']:
                signal_info['confidence'] = 0.7
        
        return signal_info
    
    def process_batch(self, image_paths: List[str], batch_size: int = 1, custom_prompt: str = None) -> List[Dict[str, Any]]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像文件路径列表
            batch_size: 批处理大小
            custom_prompt: 自定义提示
            
        Returns:
            结果列表
        """
        results = []
        
        # 使用tqdm显示进度
        for i in tqdm(range(0, len(image_paths), batch_size), desc="处理图像批次"):
            batch_paths = image_paths[i:i+batch_size]
            
            # 处理每个图像
            for image_path in batch_paths:
                result = self.process_image(image_path, custom_prompt)
                results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = None) -> str:
        """
        保存处理结果
        
        Args:
            results: 结果列表
            output_path: 输出文件路径，如果不提供则自动生成
            
        Returns:
            保存的文件路径
        """
        if output_path is None:
            timestamp = int(time.time())
            output_path = os.path.join(self.config['save_path'], f"llava_results_{timestamp}.json")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存至: {output_path}")
        return output_path
    
    def filter_complete_signals(self, results: List[Dict[str, Any]], save_dir: str = None) -> List[Dict[str, Any]]:
        """
        过滤出包含完整SBS信号的结果
        
        Args:
            results: 结果列表
            save_dir: 如果提供，则将完整信号的图像保存到指定目录
            
        Returns:
            包含完整信号的结果列表
        """
        complete_signals = [r for r in results if r['signal_info']['is_complete']]
        
        if save_dir and complete_signals:
            os.makedirs(save_dir, exist_ok=True)
            
            for result in complete_signals:
                # 复制图像
                image_path = result['image_path']
                img_name = os.path.basename(image_path)
                dest_path = os.path.join(save_dir, img_name)
                
                # 使用PIL打开并保存，而不是简单复制，确保格式兼容
                img = Image.open(image_path)
                img.save(dest_path)
                
                # 保存标注信息
                annotation_path = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_annotation.json")
                with open(annotation_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
        
        return complete_signals
    
    def estimate_processing_time(self, total_images: int) -> Dict[str, float]:
        """
        估计处理所有图像所需的时间
        
        Args:
            total_images: 总图像数量
            
        Returns:
            包含时间估计的字典
        """
        # 测试单个图像的处理时间
        test_image = np.random.rand(224, 224, 3) * 255
        test_image = Image.fromarray(test_image.astype('uint8')).convert('RGB')
        test_image_path = os.path.join(self.config['save_path'], "test_image.jpg")
        test_image.save(test_image_path)
        
        # 预热
        _ = self.process_image(test_image_path)
        
        # 计时
        start_time = time.time()
        _ = self.process_image(test_image_path)
        end_time = time.time()
        
        # 清理
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        # 计算时间
        single_image_time = end_time - start_time
        total_time = single_image_time * total_images
        
        # 转换为天和小时
        days = total_time / (24 * 3600)
        hours = total_time / 3600
        
        return {
            'single_image_time': single_image_time,
            'total_time_seconds': total_time,
            'total_time_hours': hours,
            'total_time_days': days
        } 