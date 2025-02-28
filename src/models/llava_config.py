"""
LLaVA 模型配置模块
"""

from typing import Optional
from pydantic import BaseModel, Field

class LLaVAConfig(BaseModel):
    """LLaVA模型配置"""
    
    # 基础配置
    model_path: str = Field("models/llava-sbs", description="模型路径")
    max_length: int = Field(4096, description="最大序列长度")
    attn_implementation: str = Field("flash_attention_2", description="注意力实现方式")
    
    # 词元配置
    bos_token_id: int = Field(1, description="开始词元ID")
    eos_token_id: int = Field(2, description="结束词元ID")
    pad_token_id: int = Field(0, description="填充词元ID")
    
    # 模型参数
    hidden_size: int = Field(4096, description="隐藏层大小")
    num_attention_heads: int = Field(32, description="注意力头数量")
    num_hidden_layers: int = Field(32, description="隐藏层数量")
    intermediate_size: int = Field(11008, description="中间层大小")
    
    # 训练配置
    batch_size: int = Field(4, description="批处理大小")
    gradient_accumulation_steps: int = Field(4, description="梯度累积步数")
    learning_rate: float = Field(2e-5, description="学习率")
    weight_decay: float = Field(0.01, description="权重衰减")
    warmup_steps: int = Field(100, description="预热步数")
    
    # 生成配置
    temperature: float = Field(0.7, description="温度参数")
    top_p: float = Field(0.9, description="top-p采样参数")
    top_k: int = Field(50, description="top-k采样参数")
    repetition_penalty: float = Field(1.1, description="重复惩罚参数")
    
    # 设备配置
    use_gpu: bool = Field(True, description="是否使用GPU")
    device_map: str = Field("auto", description="设备映射")
    
    # 优化配置
    use_flash_attention: bool = Field(True, description="是否使用Flash Attention")
    use_bettertransformer: bool = Field(True, description="是否使用BetterTransformer")
    use_cuda_graph: bool = Field(True, description="是否使用CUDA Graph")
    
    class Config:
        validate_assignment = True 