import os
import torch
import logging
from typing import Dict, List, Union, Any
from PIL import Image
import json
import sys

sys.path.append('/home/easyai/桌面/sbs_system/src')

os.chdir('/home/easyai/桌面/sbs_system')  # 设置当前工作目录

class LLaVAModelWrapper:
    """
    LLaVA模型包装器
    
    封装LLaVA模型的加载和预测功能
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化LLaVA模型
        
        Args:
            model_path: 模型路径
            device: 设备，如 'cuda:0', 'cpu'
        """
        self.logger = logging.getLogger('llava_model')
        self.model_path = model_path
        
        # 如果未指定设备，自动选择
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model()
        
        # 模型提示词
        self.system_prompt = """
你是一个交易图表分析专家，专注于识别SBS交易序列。请分析以下交易图表并识别SBS序列的关键点位。

SBS序列的定义:
- 点1: 结构突破后的第一次回调形成的点
- 点2: 由点1创造出的最高/最低点
- 点3: 在点1附近获取流动性的点
- 点4: 在点3附近形成双底/双顶的确认点
- 点5: 趋势延续点，价格回到点2位置

请标识：
1. 是否存在完整的SBS序列
2. 各个点位的位置
3. 建议的交易方向、入场价、目标价和止损价
4. 风险评估

格式要求JSON格式:
```
{
  "has_sbs_sequence": true/false,
  "point1": {"price": 价格, "time_index": 时间索引},
  "point2": {"price": 价格, "time_index": 时间索引},
  "point3": {"price": 价格, "time_index": 时间索引},
  "point4": {"price": 价格, "time_index": 时间索引},
  "point5": {"price": 价格, "time_index": 时间索引} (如果存在),
  "direction": "long"/"short",
  "entry_price": 建议入场价,
  "target_price": 目标价,
  "stop_loss": 止损价,
  "risk_reward_ratio": 风险回报比,
  "confidence": 0到1之间的置信度,
  "analysis": "市场分析"
}
```
"""
        
    def _load_model(self):
        """
        加载LLaVA模型
        
        Returns:
            加载的模型
        """
        from ..model.llava_sbs.model.builder import load_pretrained_model
        try:
            self.logger.info(f"加载LLaVA模型: {self.model_path}")
            
            # 获取模型名称
            model_name = get_model_name_from_path(self.model_path)
            
            # 加载tokenizer和模型
            tokenizer, model, processor, _ = load_pretrained_model(
                model_path=self.model_path,
                model_name=model_name,
                device=self.device
            )
            
            # 将模型组件打包
            model_components = {
                'tokenizer': tokenizer,
                'model': model,
                'processor': processor
            }
            
            self.logger.info("模型加载成功")
            return model_components
            
        except ImportError as e:
            self.logger.error(f"加载LLaVA相关库失败: {str(e)}")
            self.logger.error("请确保已安装LLaVA相关依赖")
            raise
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def predict_chart(self, image: Union[str, Image.Image]) -> Dict:
        """
        预测图表中的SBS序列
        
        Args:
            image: 图表图像路径或PIL图像对象
            
        Returns:
            预测结果字典
        """
        try:
            # 如果是路径，加载图像
            if isinstance(image, str):
                image = Image.open(image)
            
            # 准备输入
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            
            # 准备对话模板
            conv = conv_templates["llava_v1"].copy()
            conv.system = self.system_prompt
            
            # 构建提示词
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n分析这张交易图表，识别SBS交易序列的关键点位，并提供详细分析。仅返回JSON格式结果，不要包含其他内容。"
            
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            
            # 准备模型输入
            prompt = conv.get_prompt()
            input_ids = self.model['tokenizer'](prompt).input_ids
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
            
            # 处理图像
            image_tensor = self.model['processor'](image).unsqueeze(0).to(self.device)
            
            # 设置stop_token_ids
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stop_token_ids = self.model['tokenizer'](stop_str).input_ids
            
            # 运行模型推理
            with torch.no_grad():
                outputs = self.model['model'].generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=1024,
                    stop_token_ids=stop_token_ids
                )
            
            # 解码输出
            output_text = self.model['tokenizer'].decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            output_text = output_text.strip()
            
            # 解析JSON
            try:
                # 提取JSON部分
                if '```json' in output_text:
                    json_str = output_text.split('```json')[1].split('```')[0].strip()
                elif '```' in output_text:
                    json_str = output_text.split('```')[1].strip()
                else:
                    json_str = output_text
                
                prediction = json.loads(json_str)
                return prediction
                
            except json.JSONDecodeError:
                self.logger.warning(f"无法解析JSON: {output_text}")
                # 返回一个基本结构
                return {
                    "has_sbs_sequence": False,
                    "confidence": 0.0,
                    "analysis": "无法解析模型输出",
                    "raw_output": output_text
                }
                
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            return {
                "has_sbs_sequence": False,
                "confidence": 0.0,
                "analysis": f"预测错误: {str(e)}"
            }
    
    def batch_predict(self, images: List[Union[str, Image.Image]], batch_size: int = 4) -> List[Dict]:
        """
        批量预测图表
        
        Args:
            images: 图表图像路径或PIL图像对象列表
            batch_size: 批处理大小
            
        Returns:
            预测结果列表
        """
        results = []
        total = len(images)
        
        self.logger.info(f"开始批量预测，共 {total} 张图片，批大小 {batch_size}")
        
        for i in range(0, total, batch_size):
            batch = images[i:i+batch_size]
            batch_results = []
            
            # 处理每个批次
            for image in batch:
                result = self.predict_chart(image)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if i % 20 == 0:
                self.logger.info(f"处理进度: {min(i+batch_size, total)}/{total}")
        
        self.logger.info("批量预测完成")
        return results 