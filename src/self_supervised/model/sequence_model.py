import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import os
import json
from safetensors.torch import load_file
import logging
from transformers import LlamaForCausalLM, LlamaConfig, CLIPVisionModel, LlamaTokenizer
import torch.cuda.amp as amp

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    image_size: Tuple[int, int]  # 图像尺寸 (高, 宽)
    sequence_length: int         # 序列长度
    hidden_size: int             # 隐藏层大小
    num_heads: int               # 注意力头数
    num_layers: int              # Transformer层数
    dropout_rate: float = 0.1    # Dropout比率
    activation: str = "gelu"     # 激活函数


class SelfAttention(nn.Module):
    """自注意力模块"""
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 线性投影到查询、键、值
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_size ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权值并拼接
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(context)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.attention = SelfAttention(hidden_size, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        # 自注意力 + 残差连接
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class SequenceModel(nn.Module):
    """序列模型主类，基于LLaVA架构"""
    def __init__(self, config: Dict = None):
        super().__init__()
        # 加载默认配置
        self.config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "vocab_size": 32000,
            "mm_vision_tower": "openai/clip-vit-large-patch14-336",
            "mm_hidden_size": 1024,
            "tokenizer_path": "models/llava-sbs"  # tokenizer路径
        }
        if config:
            self.config.update(config)
            
        # 初始化LLaMA模型
        llama_config = LlamaConfig(
            hidden_size=self.config["hidden_size"],
            num_attention_heads=self.config["num_attention_heads"],
            num_hidden_layers=self.config["num_hidden_layers"],
            intermediate_size=self.config["intermediate_size"],
            max_position_embeddings=self.config["max_position_embeddings"],
            vocab_size=self.config["vocab_size"]
        )
        self.model = LlamaForCausalLM(llama_config)
        
        # 初始化视觉编码器
        self.vision_tower = CLIPVisionModel.from_pretrained(self.config["mm_vision_tower"])
        
        # 初始化tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config["tokenizer_path"])
        
        # 视觉投影层
        self.mm_projector = nn.Sequential(
            nn.Linear(self.vision_tower.config.hidden_size, self.config["mm_hidden_size"]),
            nn.GELU(),
            nn.Linear(self.config["mm_hidden_size"], self.config["hidden_size"])
        )
        
        # SBS分析的prompt模板
        self.base_prompt = {
            'role_definition': """
            你是一个专业的金融图表分析专家，专注于识别和分析SBS（Structure Breakout Sequence）交易序列。
            SBS是一种基于市场结构突破的交易策略，通过识别关键的突破点和回调点，形成12345交易序列，帮助交易者捕捉市场趋势。
            你的任务是：
            1. 评估SBS序列的完整性和有效性
            2. 确定关键点位的位置及其重要性
            3. 生成和确认交易信号
            4. 分析市场结构和趋势状态
            """,
            
            'task_focus': """
            请分析图表并关注以下要素：
            - 突破位置和有效性：市场结构是否被实体K线突破。
            - 回调的深度和形态：回调是否在0.382-0.618斐波那契范围内。
            - 双顶/双底的形成：是否存在双顶或双底形态，以及其在SBS序列中的角色。
            - 流动性获取区域：点3和点4附近是否发生流动性获取（Liquidation）。
            - SCE信号确认：是否存在单根K线入场（Single Candle Entry）信号。
            - SMA20和SMA200趋势辅助：结合短期和长期均线判断市场趋势。
            """
        }
        
        self.analysis_requirements = {
            'sequence_validation': """
            对于SBS序列，请确认：
            1. 突破的清晰度和有效性：市场结构是否被实体K线突破（例如，价格创出前低的高点后向上突破，或创出前高的低点后向下突破）。
            2. 回调的规范性：回调深度是否在0.382-0.618斐波那契范围内。
            3. 确认阶段的完整性：点3和点4是否形成双顶/双底或流动性获取区域。
            4. 整体序列的时间结构：序列的时间跨度是否合理（避免过于短暂或拉长）。
            """,
            
            'key_points': """
            请标识以下关键点位：
            1. 突破点：市场结构被实体K线突破的具体价格位置。
            2. Point 1：突破后第一次回调形成的高点（做多）或低点（做空）。
            3. Point 2：由点1创造的最高高点（做多）或最低低点（做空），作为主要目标。
            4. Point 3：点1附近的流动性获取点，或回调突破点1的高/低点。
            5. Point 4：点3附近的确认点，可能形成双顶/双底或SCE信号。
            """,
            
            'signal_generation': """
            请提供以下交易信号：
            1. 信号类型：做多或做空，基于SBS序列的方向。
            2. 入场区域建议：点4确认后的具体价格范围。
            3. 止损位置：做多信号时位于点4下方，做空信号时位于点4上方。
            4. 目标位置：点2的价格水平，或点2至点3的61.8%斐波那契回撤位作为第一止盈位。
            """
        }
        
        self.output_format = {
            'structured_response': """
            请按以下格式输出分析结果：
            
            序列评估：
            - 有效性：[是/否]
            - 完整度：[0-100%]
            - 可信度：[0-100%]
            
            关键点位：
            - 突破点：[价格水平，精确到小数点后2位]
            - Point 1：[价格水平，精确到小数点后2位]
            - Point 2：[价格水平，精确到小数点后2位]
            - Point 3：[价格水平，精确到小数点后2位]
            - Point 4：[价格水平，精确到小数点后2位]
            
            交易信号：
            - 方向：[多/空]
            - 入场区域：[价格范围，精确到小数点后2位]
            - 止损位：[价格水平，精确到小数点后2位]
            - 目标位：[价格水平，精确到小数点后2位]
            
            趋势辅助分析：
            - SMA20趋势：[上升/下降/盘整]
            - SMA200趋势：[上升/下降/盘整]
            - 整体趋势评估：[简述市场趋势方向及强度]
            
            风险评估：
            - 风险等级：[低/中/高]
            - 主要风险点：[描述，例如假突破、流动性不足等]
            """
        }
        
        self.additional_info = """
        SBS模型关键概念：
        - 有效突破：价格创出前低的高点后被实体K线向上突破，或创出前高的低点后被实体K线向下突破，标志市场结构变化。
        - 双顶/双底：双顶为上升趋势中两个接近高点，预示下跌；双底为下降趋势中两个接近低点，预示上涨，通常出现在点3和点4。
        - 流动性获取（Liquidation）：价格短暂突破关键水平，触发止损订单后迅速反转，常发生于点4。
        - SCE（Single Candle Entry）：一根K线收盘后高低点超越前一根不同颜色K线，后续第一根同色K线确认入场，通常在点4。
        - Swing结构：Higher High (HH)、Higher Low (HL)表示上升趋势，Lower Low (LL)、Lower High (LH)表示下降趋势。
        """
        
        # 构建完整的prompt模板
        self.default_prompt_template = f"""
        {self.base_prompt['role_definition']}

        {self.base_prompt['task_focus']}

        分析要求：
        {self.analysis_requirements['sequence_validation']}
        {self.analysis_requirements['key_points']}
        {self.analysis_requirements['signal_generation']}

        请按照以下格式输出：
        {self.output_format['structured_response']}

        附加信息：
        {self.additional_info}

        注意事项：
        1. 仅在确认清晰的SBS序列时生成交易信号。
        2. 对于不完整或不确定的形态，请说明原因。
        3. 若发现潜在风险（如假突破），在风险评估中详细说明。
        4. 所有价格水平精确到小数点后4位。
        5. 确保关键点位时间顺序正确。
        6. 结合SMA20和SMA200趋势验证SBS序列有效性。
        7. 关注双顶/双底和SCE信号的确认。
        """
        
        # 添加混合精度训练支持
        self.scaler = amp.GradScaler()
        self.cpu_offload = config.get('gpu_settings', {}).get('offload_to_cpu', False)
        
    def to_device(self, tensor, device):
        """处理张量设备分配"""
        if self.cpu_offload and device.type == 'cuda':
            # 如果启用了CPU卸载，保持在CPU上
            return tensor
        return tensor.to(device)
        
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str):
        """从检查点加载模型
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            加载了权重的模型实例
        """
        try:
            # 加载配置文件
            config_path = os.path.join(checkpoint_path, 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # 创建模型实例
            model = cls(config)
            
            # 加载模型权重
            model_files = [f for f in os.listdir(checkpoint_path) if f.startswith('model-') and f.endswith('.safetensors')]
            if not model_files:
                raise ValueError(f"在 {checkpoint_path} 中未找到模型文件")
                
            # 按照文件名排序以确保正确的加载顺序
            model_files.sort()
            
            # 加载所有模型文件
            state_dict = {}
            for model_file in model_files:
                file_path = os.path.join(checkpoint_path, model_file)
                state_dict.update(load_file(file_path))
                
            # 加载权重
            missing_keys, unexpected_keys = model.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"缺失的键: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"意外的键: {unexpected_keys}")
                
            return model
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
            
    def encode_images(self, images):
        """编码图像
        
        Args:
            images: 图像张量 [batch_size, num_images, channels, height, width]
            
        Returns:
            编码后的图像特征
        """
        batch_size, num_images = images.shape[:2]
        
        # 重塑图像以批量处理
        images = images.view(-1, *images.shape[2:])
        
        # 通过视觉编码器
        vision_outputs = self.vision_tower(images)
        image_features = vision_outputs.last_hidden_state
        
        # 投影到LLaMA隐藏维度
        image_features = self.mm_projector(image_features)
        
        # 重塑回批次和序列维度
        image_features = image_features.view(batch_size, num_images, -1, self.config["hidden_size"])
        
        return image_features
            
    def prepare_inputs(self, images: torch.Tensor, prompt: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """准备模型输入
        
        Args:
            images: 图像张量 [batch_size, channels, height, width]
            prompt: 可选的提示文本，如果为None则使用默认模板
            
        Returns:
            包含处理后的输入的字典
        """
        if prompt is None:
            prompt = self.default_prompt_template
            
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 将输入移动到与图像相同的设备
        inputs = {k: v.to(images.device) for k, v in inputs.items()}
        
        return inputs
            
    def forward(self, input_ids=None, attention_mask=None, images=None, labels=None, prompt=None):
        """前向传播"""
        device = next(self.parameters()).device
        
        if images is not None:
            # 将图像移动到适当的设备
            images = self.to_device(images, device)
            
            if input_ids is None and prompt is not None:
                inputs = self.prepare_inputs(images, prompt)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
            
            # 使用混合精度
            with amp.autocast():
                # 编码图像
                image_features = self.encode_images(images)
                
                # 将图像特征与文本特征拼接
                if input_ids is not None:
                    input_ids = self.to_device(input_ids, device)
                    text_embeds = self.model.get_input_embeddings()(input_ids)
                    combined_embeds = torch.cat([image_features, text_embeds], dim=1)
                    
                    if attention_mask is not None:
                        attention_mask = self.to_device(attention_mask, device)
                        image_mask = torch.ones((*attention_mask.shape[:-1], image_features.size(1)), 
                                             device=attention_mask.device)
                        attention_mask = torch.cat([image_mask, attention_mask], dim=-1)
                else:
                    combined_embeds = image_features
                    attention_mask = None
                
                # 通过LLaMA模型
                outputs = self.model(
                    inputs_embeds=combined_embeds,
                    attention_mask=attention_mask,
                    labels=self.to_device(labels, device) if labels is not None else None,
                    return_dict=True
                )
        else:
            # 仅文本输入的混合精度处理
            with amp.autocast():
                outputs = self.model(
                    input_ids=self.to_device(input_ids, device),
                    attention_mask=self.to_device(attention_mask, device),
                    labels=self.to_device(labels, device) if labels is not None else None,
                    return_dict=True
                )
        
        return outputs
        
    def analyze_sbs(self, images: torch.Tensor, custom_prompt: Optional[str] = None) -> str:
        """分析SBS序列
        
        Args:
            images: 图像张量
            custom_prompt: 可选的自定义提示文本
            
        Returns:
            模型的分析结果文本
        """
        # 准备输入
        inputs = self.prepare_inputs(images, custom_prompt)
        
        # 生成分析
        outputs = self.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            images=images
        )
        
        # 解码输出
        generated_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7
        )
        
        # 将生成的ID转换为文本
        analysis = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return analysis
        
    def encode_images(self, x):
        """编码图像序列
        
        Args:
            x: 形状为 [batch_size, sequence_length, channels, height, width] 的张量
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # 展平批次和序列维度以便于处理
        x_flat = x.view(batch_size * seq_len, channels, height, width)
        
        # 通过图像编码器
        encoded = self.image_encoder(x_flat)
        
        # 重塑回 [batch_size, sequence_length, hidden_size]
        encoded = encoded.view(batch_size, seq_len, self.config.hidden_size)
        
        return encoded
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 形状为 [batch_size, sequence_length, channels, height, width] 的张量
        """
        # 编码图像序列
        x = self.encode_images(x)
        
        # 添加位置编码
        x = x + self.position_embeddings
        
        # 应用Transformer块
        for block in self.transformer_blocks:
            x = block(x)
        
        # 获取序列表示
        sequence_repr = x[:, 0]  # 使用第一个位置的输出作为整个序列的表示
        
        # 根据当前训练阶段返回不同的输出
        if self.current_stage == 1:
            # 阶段1: 序列识别预训练
            return {
                'sequence_points': self.sequence_classifier(sequence_repr)
            }
        elif self.current_stage == 2:
            # 阶段2: 市场结构分析
            return {
                'sequence_points': self.sequence_classifier(sequence_repr),
                'market_structure': self.dropout(F.relu(sequence_repr))
            }
        else:
            # 阶段3: 交易信号生成
            return {
                'sequence_points': self.sequence_classifier(sequence_repr),
                'signal': self.signal_classifier(sequence_repr),
                'prices': self.price_regressor(sequence_repr)
            }
    
    def train_stage1(self):
        """设置为阶段1训练模式：序列识别预训练"""
        self.current_stage = 1
        
    def train_stage2(self):
        """设置为阶段2训练模式：市场结构分析"""
        self.current_stage = 2
        
    def train_stage3(self):
        """设置为阶段3训练模式：交易信号生成"""
        self.current_stage = 3 

    def train_step(self, batch, optimizer):
        """训练步骤"""
        # 启用梯度检查点
        self.model.gradient_checkpointing_enable()
        
        # 使用混合精度训练
        with amp.autocast():
            outputs = self.forward(**batch)
            loss = outputs.loss
        
        # 使用梯度缩放器
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        optimizer.zero_grad()
        
        return loss.item() 