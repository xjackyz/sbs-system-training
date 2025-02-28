"""SBS Prompt系统"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import os

@dataclass
class SBSPrompt:
    """SBS提示模板"""
    role_definition: str
    task_focus: str
    sequence_validation: str
    key_points: str
    signal_generation: str
    structured_response: str
    additional_info: str

    @classmethod
    def from_config(cls, config_path: str) -> 'SBSPrompt':
        """从配置文件加载提示模板"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls(**config)

    def get_complete_prompt(self, custom_instructions: Optional[str] = None) -> str:
        """获取完整的提示模板
        
        Args:
            custom_instructions: 可选的自定义指令
            
        Returns:
            完整的提示文本
        """
        base_prompt = f"""
{self.role_definition}

{self.task_focus}

分析要求：
{self.sequence_validation}
{self.key_points}
{self.signal_generation}

请按照以下格式输出：
{self.structured_response}

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
        if custom_instructions:
            base_prompt += f"\n\n自定义指令：\n{custom_instructions}"
            
        return base_prompt

    def get_validation_prompt(self) -> str:
        """获取验证专用提示模板"""
        return f"""
{self.role_definition}

验证重点：
1. 序列完整性：确保所有关键点位（1-4点）都已形成。
2. 形态规范性：验证每个点位是否符合SBS理论要求。
3. 技术指标确认：结合均线和其他技术指标验证。
4. 风险评估：识别潜在的假突破和风险因素。

请重点关注：
{self.sequence_validation}

输出格式：
{self.structured_response}
"""

    def get_feature_extraction_prompt(self) -> str:
        """获取特征提取专用提示模板"""
        return f"""
{self.role_definition}

特征提取重点：
1. 市场结构：识别Higher Highs/Lower Lows等结构特征
2. 价格行为：分析K线形态、成交量特征
3. 技术指标：提取关键技术指标信号
4. 动量特征：计算价格动量和趋势强度

关键点位分析：
{self.key_points}

输出要求：
- 提供每个特征的具体数值和重要性评分
- 标注特征之间的相关性
- 突出异常值和关键信号
"""

    def get_signal_generation_prompt(self) -> str:
        """获取信号生成专用提示模板"""
        return f"""
{self.role_definition}

信号生成重点：
1. 入场条件：明确定义入场触发条件
2. 风险控制：详细的止损和止盈设置
3. 信号强度：评估信号的可信度和潜在收益
4. 执行建议：具体的交易执行计划

信号要求：
{self.signal_generation}

输出格式：
{self.structured_response}
"""

# 默认提示模板
DEFAULT_PROMPT = SBSPrompt(
    role_definition="""
你是一个专业的金融图表分析专家，专注于识别和分析SBS（Structure Breakout Sequence）交易序列。
SBS是一种基于市场结构突破的交易策略，通过识别关键的突破点和回调点，形成12345交易序列，帮助交易者捕捉市场趋势。
""",
    task_focus="""
请分析图表并关注以下要素：
- 突破位置和有效性
- 回调的深度和形态
- 双顶/双底的形成
- 流动性获取区域
- SCE信号确认
- 趋势辅助指标
""",
    sequence_validation="""
1. 突破的清晰度和有效性
2. 回调的规范性
3. 确认阶段的完整性
4. 整体序列的时间结构
""",
    key_points="""
1. 突破点位置
2. Point 1形成位置
3. Point 2目标位置
4. Point 3回调位置
5. Point 4确认位置
""",
    signal_generation="""
1. 信号类型和方向
2. 入场区域建议
3. 止损位置设置
4. 目标位置规划
""",
    structured_response="""
序列评估：
- 有效性
- 完整度
- 可信度

关键点位：
- 各点位价格水平

交易信号：
- 具体建议
""",
    additional_info="""
- 有效突破定义
- 双顶/双底形态
- 流动性获取特征
- SCE信号确认
- Swing结构分析
"""
) 