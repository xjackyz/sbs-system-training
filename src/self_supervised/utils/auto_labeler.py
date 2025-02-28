"""
半自动标注工具
用于标注K线图中的SBS序列，支持人工修正和模型反馈学习
"""

import pandas as pd
import numpy as np
import talib
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from ..utils.logger import setup_logger

logger = setup_logger('auto_labeler')

class AutoLabeler:
    """半自动标注工具类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化标注工具
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.sbs_points = {}  # 存储SBS序列点位
        self.indicators = {}  # 存储技术指标
        self.feedback_data = []  # 存储人工修正的反馈数据
        self.label_history = {}  # 存储标注历史
        
    def process_data(self,
                    data: pd.DataFrame,
                    output_dir: str,
                    batch_size: int = 1000) -> Tuple[List[str], Dict]:
        """
        处理数据并标注SBS序列
        
        Args:
            data: K线数据
            output_dir: 输出目录
            batch_size: 批处理大小
            
        Returns:
            需要人工审核的样本列表和统计信息
        """
        try:
            # 计算技术指标
            self._calculate_indicators(data)
            
            # 按批次处理数据
            results = {}
            for i in range(0, len(data), batch_size):
                batch_data = data.iloc[i:i+batch_size].copy()
                batch_results = self._process_batch(batch_data, i)
                results.update(batch_results)
            
            # 保存需要人工审核的样本
            review_samples = []
            for idx, result in results.items():
                if result['confidence'] > 0.5:  # 置信度阈值
                    sample_path = self._save_sample(data, idx, result, output_dir)
                    review_samples.append(sample_path)
            
            # 统计信息
            stats = {
                'total_samples': len(data),
                'identified_sequences': len(results),
                'review_samples': len(review_samples),
                'avg_confidence': np.mean([r['confidence'] for r in results.values()])
            }
            
            return review_samples, stats
            
        except Exception as e:
            logger.error(f"处理数据失败: {str(e)}")
            raise
            
    
            
    def _detect_structure_break(self, data: pd.DataFrame, idx: int) -> Optional[Dict]:
        """检测市场结构突破"""
        try:
            if idx < 20:  # 需要足够的历史数据
                return None
                
            # 获取当前和历史数据
            current = data.iloc[idx]
            history = data.iloc[idx-20:idx]
            
            # 检查是否是实体突破
            if current['close'] > max(history['high']):  # 向上突破
                return {
                    'type': 'bullish',
                    'break_price': current['close'],
                    'previous_high': max(history['high'])
                }
            elif current['close'] < min(history['low']):  # 向下突破
                return {
                    'type': 'bearish',
                    'break_price': current['close'],
                    'previous_low': min(history['low'])
                }
                
            return None
            
        except Exception as e:
            logger.error(f"检测结构突破失败: {str(e)}")
            return None
            
    def _identify_sbs_points(self, data: pd.DataFrame, start_idx: int) -> Optional[Dict]:
        """识别SBS序列的12345点位"""
        try:
            # 检测结构突破
            break_info = self._detect_structure_break(data, start_idx)
            if not break_info:
                return None
                
            is_bullish = break_info['type'] == 'bullish'
            points = {'break': start_idx}
            
            # 寻找点1（第一次回调）
            for i in range(start_idx + 1, min(start_idx + 20, len(data))):
                if is_bullish and data.iloc[i]['low'] < data.iloc[i-1]['low']:
                    points['point1'] = i
                    break
                elif not is_bullish and data.iloc[i]['high'] > data.iloc[i-1]['high']:
                    points['point1'] = i
                    break
                    
            if 'point1' not in points:
                return None
                
            # 寻找点2（极值点）
            if is_bullish:
                point2_idx = data.iloc[points['point1']:points['point1']+20]['high'].idxmax()
            else:
                point2_idx = data.iloc[points['point1']:points['point1']+20]['low'].idxmin()
            points['point2'] = point2_idx
            
            # 寻找点3（流动性获取）
            for i in range(point2_idx + 1, min(point2_idx + 30, len(data))):
                if is_bullish and data.iloc[i]['low'] < data.iloc[points['point1']]['low']:
                    points['point3'] = i
                    break
                elif not is_bullish and data.iloc[i]['high'] > data.iloc[points['point1']]['high']:
                    points['point3'] = i
                    break
                    
            if 'point3' not in points:
                return None
                
            # 寻找点4（确认点）
            for i in range(points['point3'] + 1, min(points['point3'] + 20, len(data))):
                # 检查SCE形态
                if self._check_sce(data, i, is_bullish):
                    points['point4'] = i
                    break
                # 检查双顶/双底形态
                elif self._check_double_pattern(data, i, points['point3'], is_bullish):
                    points['point4'] = i
                    break
                    
            if 'point4' not in points:
                return None
                
            # 寻找点5（目标点）
            target_price = data.iloc[points['point2']]['high' if is_bullish else 'low']
            for i in range(points['point4'] + 1, min(points['point4'] + 50, len(data))):
                if is_bullish and data.iloc[i]['high'] >= target_price:
                    points['point5'] = i
                    break
                elif not is_bullish and data.iloc[i]['low'] <= target_price:
                    points['point5'] = i
                    break
                    
            return points if 'point5' in points else None
            
        except Exception as e:
            logger.error(f"识别SBS点位失败: {str(e)}")
            return None
            
    def _check_sce(self, data: pd.DataFrame, idx: int, is_bullish: bool) -> bool:
        """检查SCE（Single Candle Entry）形态"""
        try:
            if idx < 2:
                return False
                
            current = data.iloc[idx]
            prev = data.iloc[idx-1]
            prev2 = data.iloc[idx-2]
            
            if is_bullish:
                return (current['close'] > current['open'] and  # 当前为阳线
                        prev['close'] < prev['open'] and  # 前一根为阴线
                        current['high'] > prev['high'] and  # 高点突破
                        current['low'] > prev['low'])  # 低点抬升
            else:
                return (current['close'] < current['open'] and  # 当前为阴线
                        prev['close'] > prev['open'] and  # 前一根为阳线
                        current['low'] < prev['low'] and  # 低点突破
                        current['high'] < prev['high'])  # 高点下降
                        
        except Exception as e:
            logger.error(f"检查SCE失败: {str(e)}")
            return False
            
    def _check_double_pattern(self, data: pd.DataFrame, idx: int, point3_idx: int, is_bullish: bool) -> bool:
        """检查双顶/双底形态"""
        try:
            if idx <= point3_idx:
                return False
                
            point3_price = data.iloc[point3_idx]['low' if is_bullish else 'high']
            current_price = data.iloc[idx]['low' if is_bullish else 'high']
            
            # 检查价格是否接近
            price_diff = abs(current_price - point3_price)
            avg_price = (current_price + point3_price) / 2
            if price_diff / avg_price > 0.01:  # 允许1%的差异
                return False
                
            # 检查中间是否有明显回调
            middle_data = data.iloc[point3_idx+1:idx]
            if is_bullish:
                max_middle = middle_data['high'].max()
                return max_middle > point3_price + price_diff
            else:
                min_middle = middle_data['low'].min()
                return min_middle < point3_price - price_diff
                
        except Exception as e:
            logger.error(f"检查双顶/双底失败: {str(e)}")
            return False
            
    def _process_batch(self, batch_data: pd.DataFrame, start_idx: int) -> Dict[int, Dict]:
        """处理数据批次"""
        results = {}
        
        try:
            for i in range(len(batch_data)):
                idx = start_idx + i
                sbs_points = self._identify_sbs_points(batch_data, i)
                
                if sbs_points:
                    # 计算置信度
                    confidence = self._calculate_confidence(batch_data, sbs_points)
                    
                    results[idx] = {
                        'points': sbs_points,
                        'confidence': confidence,
                        'trend': self._check_trend(batch_data, i),
                        'indicators': self._get_indicator_values(idx)
                    }
                    
        except Exception as e:
            logger.error(f"处理批次失败: {str(e)}")
            
        return results
            
    def _calculate_confidence(self, data: pd.DataFrame, points: Dict) -> float:
        """计算标注置信度"""
        try:
            confidence = 0.0
            
            # 检查趋势一致性
            trend_score = self._check_trend_consistency(data, points)
            confidence += trend_score * 0.3
            
            # 检查技术指标确认
            indicator_score = self._check_indicator_confirmation(data, points)
            confidence += indicator_score * 0.3
            
            # 检查成交量确认
            volume_score = self._check_volume_confirmation(data, points)
            confidence += volume_score * 0.2
            
            # 检查形态完整性
            pattern_score = self._check_pattern_completeness(data, points)
            confidence += pattern_score * 0.2
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"计算置信度失败: {str(e)}")
            return 0.0
            
    def _save_sample(self, data: pd.DataFrame, index: int, result: Dict, output_dir: str) -> str:
        """保存样本数据"""
        try:
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 准备保存的数据
            sample_data = {
                'timestamp': data.iloc[index]['datetime'],
                'sbs_points': result['points'],
                'confidence': result['confidence'],
                'trend': result['trend'],
                'indicators': result['indicators']
            }
            
            # 保存为JSON文件
            filename = f"sample_{index}_{result['confidence']:.2f}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
                
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存样本失败: {str(e)}")
            raise
            
    def save_human_feedback(self, 
                          sample_id: str, 
                          original_points: Dict, 
                          corrected_points: Dict,
                          feedback_notes: Optional[str] = None) -> None:
        """
        保存人工修正的反馈数据
        
        Args:
            sample_id: 样本ID
            original_points: 原始标注点位
            corrected_points: 人工修正的点位
            feedback_notes: 修正说明
        """
        try:
            feedback = {
                'sample_id': sample_id,
                'timestamp': pd.Timestamp.now(),
                'original_points': original_points,
                'corrected_points': corrected_points,
                'feedback_notes': feedback_notes,
                'differences': self._analyze_point_differences(
                    original_points, 
                    corrected_points
                )
            }
            
            self.feedback_data.append(feedback)
            self._update_confidence_model(feedback)
            
            # 保存反馈数据
            feedback_path = Path(self.config.get('feedback_dir', 'data/feedback'))
            feedback_path.mkdir(parents=True, exist_ok=True)
            
            with open(feedback_path / f"{sample_id}_feedback.json", 'w') as f:
                json.dump(feedback, f, indent=2)
                
        except Exception as e:
            logger.error(f"保存反馈数据失败: {str(e)}")
            raise
            
    def _analyze_point_differences(self, 
                                original: Dict, 
                                corrected: Dict) -> Dict:
        """分析原始标注和人工修正的差异"""
        differences = {}
        for point in ['point1', 'point2', 'point3', 'point4', 'point5']:
            if point in original and point in corrected:
                diff = {
                    'index_diff': corrected[point] - original[point],
                    'price_diff': self._get_price_diff(
                        original[point], 
                        corrected[point]
                    ),
                    'pattern_change': self._analyze_pattern_change(
                        point, 
                        original, 
                        corrected
                    )
                }
                differences[point] = diff
        return differences
        
    def generate_label_suggestions(self, 
                                data: pd.DataFrame, 
                                index: int,
                                window_size: int = 100) -> Dict:
        """
        生成标注建议
        
        Args:
            data: K线数据
            index: 当前位置
            window_size: 窗口大小
            
        Returns:
            标注建议，包含可能的点位和置信度
        """
        try:
            window_data = data.iloc[max(0, index-window_size):min(len(data), index+window_size)]
            suggestions = {
                'primary': self._identify_sbs_points(window_data, window_size//2),
                'alternatives': self._find_alternative_points(window_data, window_size//2),
                'confidence_scores': self._calculate_point_confidence(window_data, window_size//2)
            }
            return suggestions
            
    def _find_alternative_points(self, 
                              data: pd.DataFrame, 
                              center_idx: int) -> Dict[str, List]:
        """查找每个点位的候选位置"""
        alternatives = {}
        for point in ['point1', 'point2', 'point3', 'point4', 'point5']:
            alternatives[point] = self._find_point_candidates(
                data, 
                center_idx, 
                point
            )
        return alternatives
        
    def _calculate_point_confidence(self, 
                                data: pd.DataFrame, 
                                center_idx: int) -> Dict[str, float]:
        """计算每个点位的置信度分数"""
        confidence = {}
        for point in ['point1', 'point2', 'point3', 'point4', 'point5']:
            confidence[point] = self._get_point_confidence(
                data, 
                center_idx, 
                point
            )
        return confidence
        
    def update_from_feedback(self, feedback_dir: str) -> None:
        """从反馈数据更新模型参数"""
        try:
            feedback_path = Path(feedback_dir)
            feedback_files = list(feedback_path.glob('*_feedback.json'))
            
            for file in feedback_files:
                with open(file, 'r') as f:
                    feedback = json.load(f)
                    self._update_confidence_model(feedback)
                    
            logger.info(f"已从 {len(feedback_files)} 个反馈文件更新模型")
            
        except Exception as e:
            logger.error(f"更新模型参数失败: {str(e)}")
            raise
            
    def export_label_history(self, output_path: str) -> None:
        """导出标注历史"""
        try:
            history_data = {
                'labels': self.label_history,
                'feedback': self.feedback_data,
                'statistics': self._calculate_labeling_stats()
            }
            
            with open(output_path, 'w') as f:
                json.dump(history_data, f, indent=2)
                
            logger.info(f"标注历史已导出到: {output_path}")
            
        except Exception as e:
            logger.error(f"导出标注历史失败: {str(e)}")
            raise
            
    def _calculate_labeling_stats(self) -> Dict:
        """计算标注统计信息"""
        stats = {
            'total_samples': len(self.label_history),
            'human_corrections': len(self.feedback_data),
            'confidence_distribution': {},
            'common_corrections': self._analyze_common_corrections(),
            'model_improvement': self._calculate_model_improvement()
        }
        return stats
        
    def _update_confidence_model(self, feedback: Dict) -> None:
        """从反馈数据更新模型参数"""
        # Implementation of _update_confidence_model method
        pass
        
    def _analyze_common_corrections(self) -> Dict:
        """分析常见的修正"""
        # Implementation of _analyze_common_corrections method
        pass
        
    def _calculate_model_improvement(self) -> float:
        """计算模型改进"""
        # Implementation of _calculate_model_improvement method
        pass
        
    def _get_price_diff(self, point1: int, point2: int) -> float:
        """计算两点之间的价格差异"""
        # Implementation of _get_price_diff method
        pass
        
    def _analyze_pattern_change(self, point: str, original: Dict, corrected: Dict) -> str:
        """分析标注点的模式变化"""
        # Implementation of _analyze_pattern_change method
        pass
        
    def _find_point_candidates(self, data: pd.DataFrame, center_idx: int, point: str) -> List:
        """查找每个点位的候选位置"""
        # Implementation of _find_point_candidates method
        pass
        
    def _get_point_confidence(self, data: pd.DataFrame, center_idx: int, point: str) -> float:
        """计算每个点位的置信度"""
        # Implementation of _get_point_confidence method
        pass
        
    def _check_trend(self, data: pd.DataFrame, idx: int) -> str:
        """检查趋势"""
        # Implementation of _check_trend method
        pass
        
    def _check_trend_consistency(self, data: pd.DataFrame, points: Dict) -> float:
        """检查趋势一致性"""
        # Implementation of _check_trend_consistency method
        pass
        
    def _check_indicator_confirmation(self, data: pd.DataFrame, points: Dict) -> float:
        """检查技术指标确认"""
        # Implementation of _check_indicator_confirmation method
        pass
        
    def _check_volume_confirmation(self, data: pd.DataFrame, points: Dict) -> float:
        """检查成交量确认"""
        # Implementation of _check_volume_confirmation method
        pass
        
    def _check_pattern_completeness(self, data: pd.DataFrame, points: Dict) -> float:
        """检查形态完整性"""
        # Implementation of _check_pattern_completeness method
        pass
        
    def _get_indicator_values(self, idx: int) -> Dict:
        """获取技术指标值"""
        # Implementation of _get_indicator_values method
        pass 