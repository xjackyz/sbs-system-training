"""
LLaVA 模型实现模块
"""

import os
from typing import Dict, List, Optional, Union, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

from .base_model import BaseModel
from .llava_config import LLaVAConfig
from ..utils.logger import setup_logger
from ..utils.exceptions import ModelError

logger = setup_logger('llava_model')

class LLaVAModel(BaseModel):
    """LLaVA模型实现"""
    
    def __init__(self, config: Union[Dict[str, Any], LLaVAConfig]):
        """
        初始化LLaVA模型
        
        Args:
            config: 模型配置，可以是字典或LLaVAConfig实例
        """
        if isinstance(config, dict):
            config = LLaVAConfig(**config)
        super().__init__(config.dict())
        
        self.config = config
        self._init_model()
        
    def _init_model(self) -> None:
        """初始化模型和分词器"""
        try:
            # 量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map=self.config.device_map,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
                attn_implementation=self.config.attn_implementation,
                use_flash_attention=self.config.use_flash_attention,
                use_bettertransformer=self.config.use_bettertransformer
            )
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            # 检查是否有LoRA权重
            adapter_path = os.path.join(self.config.model_path, "adapter_model")
            if os.path.exists(adapter_path):
                logger.info(f"加载LoRA权重: {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_path,
                    torch_dtype=torch.float16
                )
            
            logger.info(f"模型已加载: {self.config.model_path}")
            
        except Exception as e:
            logger.error(f"初始化模型失败: {str(e)}")
            raise ModelError(f"初始化模型失败: {str(e)}")
            
    def forward(self, 
               input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               **kwargs) -> Dict[str, torch.Tensor]:
        """
        模型前向传播
        
        Args:
            input_ids: 输入词元ID
            attention_mask: 注意力掩码
            **kwargs: 其他参数
            
        Returns:
            模型输出
        """
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            return outputs
            
        except Exception as e:
            logger.error(f"模型前向传播失败: {str(e)}")
            raise ModelError(f"模型前向传播失败: {str(e)}")
            
    def generate(self, 
                prompt: Union[str, List[str]],
                max_length: Optional[int] = None,
                **kwargs) -> List[str]:
        """
        生成文本
        
        Args:
            prompt: 输入提示文本
            max_length: 最大生成长度
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本列表
        """
        try:
            # 处理输入
            if isinstance(prompt, str):
                prompt = [prompt]
                
            # 设置默认参数
            generation_config = {
                'max_length': max_length or self.config.max_length,
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'top_k': self.config.top_k,
                'repetition_penalty': self.config.repetition_penalty,
                'pad_token_id': self.config.pad_token_id,
                'eos_token_id': self.config.eos_token_id,
                'bos_token_id': self.config.bos_token_id,
                **kwargs
            }
            
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
                
            # 解码输出
            generated_texts = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"生成文本失败: {str(e)}")
            raise ModelError(f"生成文本失败: {str(e)}")
            
    def analyze_market(self, 
                      market_data: List[Dict],
                      **kwargs) -> Dict[str, Any]:
        """
        分析市场数据
        
        Args:
            market_data: 市场数据列表
            **kwargs: 其他参数
            
        Returns:
            分析结果
        """
        try:
            # 构建提示文本
            prompt = self._build_market_prompt(market_data)
            
            # 生成分析结果
            analysis = self.generate(prompt, **kwargs)[0]
            
            # 解析结果
            result = self._parse_analysis(analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"分析市场数据失败: {str(e)}")
            raise ModelError(f"分析市场数据失败: {str(e)}")
            
    def _build_market_prompt(self, market_data: List[Dict]) -> str:
        """
        构建市场分析提示文本
        
        Args:
            market_data: 市场数据列表
            
        Returns:
            提示文本
        """
        # 提取关键数据
        latest_data = market_data[-1]
        price_changes = [
            (d['close'] - d['open']) / d['open'] * 100
            for d in market_data
        ]
        
        # 构建提示文本
        prompt = f"""请分析以下市场数据：

最新价格: {latest_data['close']}
24小时变化: {price_changes[-1]:.2f}%
7日变化: {sum(price_changes[-7:]):.2f}%
交易量: {latest_data['volume']}

请提供：
1. 市场趋势（上涨/下跌/震荡）
2. 趋势强度（强/中/弱）
3. 建议操作
4. 分析依据

分析："""
        
        return prompt
        
    def _parse_analysis(self, analysis: str) -> Dict[str, Any]:
        """
        解析分析文本
        
        Args:
            analysis: 分析文本
            
        Returns:
            解析后的结构化数据
        """
        # 这里应该实现更复杂的解析逻辑
        # 当前使用简单的关键词匹配
        result = {
            'trend': 'neutral',
            'strength': 'moderate',
            'action': 'hold',
            'confidence': 0.5,
            'analysis': analysis
        }
        
        # 趋势判断
        if '上涨' in analysis:
            result['trend'] = 'bullish'
        elif '下跌' in analysis:
            result['trend'] = 'bearish'
            
        # 强度判断
        if '强' in analysis:
            result['strength'] = 'strong'
            result['confidence'] = 0.8
        elif '弱' in analysis:
            result['strength'] = 'weak'
            result['confidence'] = 0.3
            
        # 操作建议
        if '买入' in analysis:
            result['action'] = 'buy'
        elif '卖出' in analysis:
            result['action'] = 'sell'
            
        return result 