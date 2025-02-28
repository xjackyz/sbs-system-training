"""
LLaVA分析器组件
负责分析市场数据，提供趋势和强度信息
"""

import logging
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger('llava_analyzer')

class LLaVAAnalyzer:
    """LLaVA分析器类"""
    
    def __init__(self, model_path: str = None, config: Dict = None):
        """
        初始化LLaVA分析器
        
        Args:
            model_path: 模型路径
            config: 配置参数
        """
        self.config = config or load_config()
        self.model_path = model_path or self.config.get('MODEL_PATH')
        
        # 加载模型和分词器
        try:
            self.device = 'cuda' if torch.cuda.is_available() and self.config.get('USE_GPU', True) else 'cpu'
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f'模型已加载到设备: {self.device}')
        except Exception as e:
            logger.error(f'加载模型失败: {str(e)}')
            raise
            
    def analyze_market_data(self, 
                          data: List[Dict],
                          window_size: int = 20) -> List[Dict]:
        """
        分析市场数据
        
        Args:
            data: 市场数据
            window_size: 分析窗口大小
            
        Returns:
            分析结果列表
        """
        try:
            results = []
            for i in range(len(data) - window_size + 1):
                window = data[i:i + window_size]
                analysis = self._analyze_window(window)
                results.append({
                    'timestamp': data[i + window_size - 1]['timestamp'],
                    'trend': analysis['trend'],
                    'strength': analysis['strength'],
                    'confidence': analysis['confidence']
                })
            return results
        except Exception as e:
            logger.error(f'分析市场数据失败: {str(e)}')
            raise
            
    def _analyze_window(self, window: List[Dict]) -> Dict:
        """
        分析单个时间窗口的数据
        
        Args:
            window: 时间窗口数据
            
        Returns:
            分析结果
        """
        try:
            # 准备输入数据
            features = self._extract_features(window)
            inputs = self.tokenizer(features, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # 处理输出
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            # 映射预测结果
            trend_mapping = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
            strength_mapping = {0: 'weak', 1: 'moderate', 2: 'strong'}
            
            return {
                'trend': trend_mapping[prediction[0].item()],
                'strength': strength_mapping[torch.argmax(outputs.logits[:, 3:6]).item()],
                'confidence': confidence[0].item()
            }
            
        except Exception as e:
            logger.error(f'分析窗口数据失败: {str(e)}')
            raise
            
    def _extract_features(self, window: List[Dict]) -> str:
        """
        从时间窗口数据中提取特征
        
        Args:
            window: 时间窗口数据
            
        Returns:
            特征字符串
        """
        try:
            # 计算基本特征
            price_changes = []
            volumes = []
            for i in range(1, len(window)):
                price_change = (window[i]['close'] - window[i-1]['close']) / window[i-1]['close']
                price_changes.append(price_change)
                volumes.append(window[i]['volume'])
                
            # 构建特征字符串
            features = f"""
            Price Movement:
            - Latest close: {window[-1]['close']}
            - Price change: {sum(price_changes):.2%}
            - Max price: {max(w['high'] for w in window)}
            - Min price: {min(w['low'] for w in window)}
            
            Volume Analysis:
            - Average volume: {sum(volumes) / len(volumes):.2f}
            - Volume trend: {'increasing' if volumes[-1] > volumes[0] else 'decreasing'}
            
            Pattern Indicators:
            - Trend direction: {'up' if sum(price_changes) > 0 else 'down'}
            - Volatility: {(max(w['high'] for w in window) - min(w['low'] for w in window)) / window[0]['close']:.2%}
            """
            
            return features.strip()
            
        except Exception as e:
            logger.error(f'提取特征失败: {str(e)}')
            raise
            
    def save_analysis(self, analysis: List[Dict], filepath: str) -> None:
        """
        保存分析结果
        
        Args:
            analysis: 分析结果
            filepath: 保存路径
        """
        try:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            logger.info(f'分析结果已保存到: {filepath}')
        except Exception as e:
            logger.error(f'保存分析结果失败: {str(e)}')
            raise 