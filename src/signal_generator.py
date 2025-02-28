"""
信号生成器组件
基于分析结果生成交易信号
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from .utils.logger import setup_logger
from .utils.config import load_config

logger = setup_logger('signal_generator')

class SignalGenerator:
    """信号生成器类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化信号生成器
        
        Args:
            config: 配置参数
        """
        self.config = config or load_config()
        self.thresholds = self.config.get('signal_thresholds', {
            'confidence': 0.8,
            'strength_required': {'buy': 'strong', 'sell': 'strong'},
            'trend_required': {'buy': 'bullish', 'sell': 'bearish'}
        })
        
    def generate_signals(self, analysis_results: List[Dict]) -> List[Dict]:
        """
        生成交易信号
        
        Args:
            analysis_results: 分析结果列表
            
        Returns:
            交易信号列表
        """
        try:
            signals = []
            for result in analysis_results:
                signal = self._evaluate_signal(result)
                if signal:
                    signals.append({
                        'timestamp': result['timestamp'],
                        'type': signal['type'],
                        'confidence': signal['confidence'],
                        'reason': signal['reason']
                    })
            
            logger.info(f'生成了 {len(signals)} 个交易信号')
            return signals
            
        except Exception as e:
            logger.error(f'生成交易信号失败: {str(e)}')
            raise
            
    def _evaluate_signal(self, analysis: Dict) -> Optional[Dict]:
        """
        评估单个分析结果并生成信号
        
        Args:
            analysis: 分析结果
            
        Returns:
            交易信号或None
        """
        try:
            # 检查置信度
            if analysis['confidence'] < self.thresholds['confidence']:
                return None
                
            # 评估买入信号
            if (analysis['trend'] == self.thresholds['trend_required']['buy'] and
                analysis['strength'] == self.thresholds['strength_required']['buy']):
                return {
                    'type': 'buy',
                    'confidence': analysis['confidence'],
                    'reason': f"强烈上涨趋势 (置信度: {analysis['confidence']:.2%})"
                }
                
            # 评估卖出信号
            if (analysis['trend'] == self.thresholds['trend_required']['sell'] and
                analysis['strength'] == self.thresholds['strength_required']['sell']):
                return {
                    'type': 'sell',
                    'confidence': analysis['confidence'],
                    'reason': f"强烈下跌趋势 (置信度: {analysis['confidence']:.2%})"
                }
                
            return None
            
        except Exception as e:
            logger.error(f'评估信号失败: {str(e)}')
            raise
            
    def filter_signals(self, 
                      signals: List[Dict],
                      min_interval: int = 24) -> List[Dict]:
        """
        过滤信号，去除过于频繁的信号
        
        Args:
            signals: 信号列表
            min_interval: 最小信号间隔（小时）
            
        Returns:
            过滤后的信号列表
        """
        try:
            if not signals:
                return []
                
            filtered = [signals[0]]
            last_signal_time = datetime.fromtimestamp(signals[0]['timestamp'])
            
            for signal in signals[1:]:
                current_time = datetime.fromtimestamp(signal['timestamp'])
                hours_diff = (current_time - last_signal_time).total_seconds() / 3600
                
                if hours_diff >= min_interval:
                    filtered.append(signal)
                    last_signal_time = current_time
                    
            logger.info(f'过滤后保留了 {len(filtered)} 个信号')
            return filtered
            
        except Exception as e:
            logger.error(f'过滤信号失败: {str(e)}')
            raise
            
    def save_signals(self, signals: List[Dict], filepath: str) -> None:
        """
        保存交易信号
        
        Args:
            signals: 交易信号列表
            filepath: 保存路径
        """
        try:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(signals, f, ensure_ascii=False, indent=2)
            logger.info(f'信号已保存到: {filepath}')
        except Exception as e:
            logger.error(f'保存信号失败: {str(e)}')
            raise
            
    def validate_signals(self, signals: List[Dict]) -> bool:
        """
        验证交易信号
        
        Args:
            signals: 交易信号列表
            
        Returns:
            信号是否有效
        """
        try:
            if not signals:
                logger.warning('没有生成任何信号')
                return False
                
            required_fields = ['timestamp', 'type', 'confidence', 'reason']
            for signal in signals:
                if not all(field in signal for field in required_fields):
                    logger.warning(f'信号缺少必要字段: {required_fields}')
                    return False
                    
                if signal['type'] not in ['buy', 'sell']:
                    logger.warning(f'无效的信号类型: {signal["type"]}')
                    return False
                    
                if not isinstance(signal['confidence'], (int, float)):
                    logger.warning(f'无效的置信度值: {signal["confidence"]}')
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f'验证信号失败: {str(e)}')
            raise 