"""
SBS序列识别的提示模板
用于引导LLaVA模型识别K线图中的SBS序列模式
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

class SBSPromptTemplate:
    """SBS提示模板类，用于生成LLaVA模型的输入提示"""
    
    def __init__(self, config: Dict = None):
        """
        初始化SBS提示模板
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.templates_dir = self.config.get('templates_dir', 'prompts/templates')
        self._load_templates()
        
    def _load_templates(self):
        """加载提示模板"""
        templates_path = Path(self.templates_dir)
        
        # 默认模板
        self.default_template = """
        请分析这张K线图，并识别其中可能的SBS (12345) 序列交易信号。
        
        SBS序列有以下特点:
        1. 五个关键点，按照1-2-3-4-5的顺序出现
        2. 点1是第一次回调点
        3. 点2是极值点，可能是顶部或底部
        4. 点3是流动性获取点
        5. 点4是确认点，确认市场结构变化
        6. 点5是目标点，预示着序列结束
        
        请识别图中是否存在SBS序列，并指出序列的当前状态以及关键点位置。
        如果序列已形成到点4以后，请给出交易方向建议（多/空）。
        
        请按照以下JSON格式输出结果:
        ```json
        {
            "sequence_status": {
                "label": "未形成/形成中/已完成",
                "is_active": true/false
            },
            "points": {
                "point1": <点1的K线索引或null>,
                "point2": <点2的K线索引或null>,
                "point3": <点3的K线索引或null>,
                "point4": <点4的K线索引或null>,
                "point5": <点5的K线索引或null>
            },
            "trade_setup": {
                "direction": "多/空/无信号",
                "reason": "交易理由"
            }
        }
        ```
        
        必须以有效的JSON格式返回结果，不要包含其他文本解释。
        """
        
        # 尝试从文件加载模板
        try:
            if templates_path.exists():
                for template_file in templates_path.glob("*.txt"):
                    template_name = template_file.stem
                    with open(template_file, "r", encoding="utf-8") as f:
                        setattr(self, f"{template_name}_template", f.read())
        except Exception as e:
            print(f"加载模板文件时出错: {str(e)}")
            
    def generate_prompt(self, 
                       template_name: str = "default", 
                       context: Dict[str, Any] = None) -> str:
        """
        生成提示
        
        Args:
            template_name: 模板名称
            context: 上下文信息
            
        Returns:
            生成的提示文本
        """
        context = context or {}
        template_attr = f"{template_name}_template"
        
        # 获取指定模板，如果不存在则使用默认模板
        if hasattr(self, template_attr):
            template = getattr(self, template_attr)
        else:
            template = self.default_template
            
        # 替换模板变量
        prompt = template
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            prompt = prompt.replace(placeholder, str(value))
            
        return prompt
        
    def get_default_prompt(self) -> str:
        """获取默认提示模板"""
        return self.default_template
        
    def get_detailed_prompt(self, 
                          timeframe: str = "4小时", 
                          symbol: str = "BTCUSDT",
                          trade_context: Dict = None) -> str:
        """
        获取详细提示模板
        
        Args:
            timeframe: 时间周期
            symbol: 交易对
            trade_context: 交易上下文
            
        Returns:
            详细提示文本
        """
        trade_context = trade_context or {}
        
        detailed_prompt = f"""
        请详细分析这张{timeframe}周期的{symbol} K线图，并识别其中的SBS (12345) 序列交易信号。
        
        SBS序列交易策略专注于市场结构变化和有效突破:
        
        1. 点1（第一次回调）: 在一段趋势后的第一次回调点，标志着可能的结构变化开始
        2. 点2（极值点）: 可能是高点或低点，表示趋势转折的极值
        3. 点3（流动性获取）: 价格获取流动性的点位，可能突破点2造成多/空陷阱
        4. 点4（确认点）: 确认市场结构变化的点位，通常在点3之后的反向移动中出现
        5. 点5（目标点）: 序列完成后的目标位置，可用于设置获利目标
        
        请注意以下关键要素:
        - 有效突破: 必须伴随成交量放大、突破前的蓄能和多根K线确认
        - 双顶双底: 点2和点3可能形成双顶或双底结构
        - SMA均线: 可使用20和50 SMA判断整体趋势方向
        
        """
        
        # 添加当前市场信息
        if trade_context:
            detailed_prompt += """
            当前市场信息:
            """
            for key, value in trade_context.items():
                detailed_prompt += f"- {key}: {value}\n"
                
        # 添加输出格式要求
        detailed_prompt += """
        请按照以下JSON格式输出分析结果:
        ```json
        {
            "sequence_status": {
                "label": "未形成/形成中/已完成",
                "is_active": true/false,
                "confidence": 0.95,
                "analysis": "对序列状态的简要分析"
            },
            "points": {
                "point1": {"index": <K线索引>, "description": "点1特征描述"},
                "point2": {"index": <K线索引>, "description": "点2特征描述"},
                "point3": {"index": <K线索引>, "description": "点3特征描述"},
                "point4": {"index": <K线索引>, "description": "点4特征描述"},
                "point5": {"index": <K线索引>, "description": "点5特征描述"}
            },
            "trade_setup": {
                "direction": "多/空/无信号",
                "entry_price": <入场价格>,
                "stop_loss": <止损价格>,
                "take_profit": <止盈价格>,
                "risk_reward_ratio": <风险回报比>,
                "confidence": <置信度>,
                "reasoning": "交易理由详细分析"
            },
            "market_structure": {
                "trend": "上升/下降/盘整",
                "key_levels": [<关键价格水平>],
                "volume_analysis": "成交量分析"
            }
        }
        ```
        
        请确保返回有效的JSON格式，不要包含其他文本。对于未能确定的字段，请使用null。
        """
        
        return detailed_prompt
        
    def get_system_prompt(self) -> str:
        """获取系统提示"""
        return """
        你是一位专业的交易员，精通SBS (12345) 序列交易策略。你的专长是识别K线图中的SBS序列模式并提供交易建议。
        请基于图表数据提供客观分析，只在确定看到清晰信号时才给出交易建议。
        回答必须是有效的JSON格式，包含序列状态、关键点位和交易设置。
        """
        
    def get_continuation_prompt(self, previous_analysis: Dict) -> str:
        """
        获取后续分析提示
        
        Args:
            previous_analysis: 之前的分析结果
            
        Returns:
            后续分析提示
        """
        try:
            prev_status = previous_analysis.get('sequence_status', {}).get('label', '未知')
            
            prompt = f"""
            这是新的K线图更新。请基于之前的分析结果继续分析SBS序列的发展情况。
            
            你之前的分析显示序列状态为: {prev_status}
            
            请重点关注:
            1. 序列状态是否有变化
            2. 新的关键点是否出现
            3. 是否出现新的交易信号
            4. 如果有活跃交易，评估其是否应该持有或平仓
            
            请使用标准JSON格式返回更新后的分析结果。
            """
            
            return prompt
        except Exception as e:
            print(f"生成后续分析提示时出错: {str(e)}")
            return self.get_default_prompt()
            
    def save_template(self, template_name: str, template_content: str) -> bool:
        """
        保存模板到文件
        
        Args:
            template_name: 模板名称
            template_content: 模板内容
            
        Returns:
            是否保存成功
        """
        try:
            templates_path = Path(self.templates_dir)
            templates_path.mkdir(parents=True, exist_ok=True)
            
            template_file = templates_path / f"{template_name}.txt"
            with open(template_file, "w", encoding="utf-8") as f:
                f.write(template_content)
                
            # 更新实例属性
            setattr(self, f"{template_name}_template", template_content)
            
            return True
        except Exception as e:
            print(f"保存模板失败: {str(e)}")
            return False
            
    def format_response(self, response: str) -> Optional[Dict]:
        """
        格式化LLaVA响应为字典
        
        Args:
            response: LLaVA响应文本
            
        Returns:
            解析后的字典或None
        """
        try:
            # 尝试提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                print("无法在响应中找到有效的JSON")
                return None
        except json.JSONDecodeError as e:
            print(f"解析JSON失败: {str(e)}")
            return None
        except Exception as e:
            print(f"格式化响应时出错: {str(e)}")
            return None 