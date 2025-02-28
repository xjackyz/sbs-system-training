"""
主动学习采样器
用于选择需要人工标注的样本
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import shutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from datetime import datetime

from ..utils.logger import setup_logger
from ..model.llava_processor import LlavaProcessor

logger = setup_logger('active_learner')

class ActiveLearner:
    """主动学习采样器"""
    
    def __init__(self, 
                 llava_processor: LlavaProcessor,
                 unlabeled_dir: str,
                 output_dir: str,
                 config: Dict = None):
        """
        初始化主动学习采样器
        
        Args:
            llava_processor: LLaVA处理器
            unlabeled_dir: 未标注数据目录
            output_dir: 输出目录
            config: 配置参数
        """
        self.processor = llava_processor
        self.unlabeled_dir = Path(unlabeled_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.uncertainty_ratio = self.config.get('uncertainty_ratio', 0.05)
        self.diversity_ratio = self.config.get('diversity_ratio', 0.15)  # 新增多样性采样比例
        self.random_ratio = self.config.get('random_ratio', 0.10)
        self.n_clusters = self.config.get('n_clusters', 5)
        
        self.logger = setup_logger('active_learner')
        
    def run_sampling(self) -> Tuple[List[str], Dict[str, Any]]:
        """
        运行采样过程
        
        Returns:
            选中的文件列表和统计信息
        """
        try:
            # 获取所有未标注文件
            unlabeled_files = list(self.unlabeled_dir.glob('*.png'))
            if not unlabeled_files:
                raise ValueError(f"未找到未标注数据: {self.unlabeled_dir}")
            self.logger.info(f"发现 {len(unlabeled_files)} 个未标注样本")
            
            # 获取预测结果
            predictions = []
            for file in unlabeled_files:
                pred = self.processor.process_image(str(file))
                predictions.append(pred)
                
            # 计算不确定性分数
            uncertainties = self._calculate_uncertainty(predictions)
            
            # 提取特征并进行聚类
            features = [self._extract_features(pred) for pred in predictions]
            features = np.array(features)
            
            # 计算市场状态特征
            market_states = self._analyze_market_states(predictions)
            
            # 多样性采样
            diversity_indices = self._diversity_sampling(features, market_states)
            n_diversity = int(len(unlabeled_files) * self.diversity_ratio)
            selected_diversity = diversity_indices[:n_diversity]
            
            # 不确定性采样
            n_uncertainty = int(len(unlabeled_files) * self.uncertainty_ratio)
            uncertainty_indices = np.argsort(uncertainties)[-n_uncertainty:]
            
            # 随机采样
            remaining_indices = list(set(range(len(unlabeled_files))) - 
                                  set(selected_diversity) - 
                                  set(uncertainty_indices))
            n_random = int(len(unlabeled_files) * self.random_ratio)
            random_indices = np.random.choice(remaining_indices, n_random, replace=False)
            
            # 合并所有选中的样本
            selected_indices = np.concatenate([
                selected_diversity,
                uncertainty_indices,
                random_indices
            ])
            
            # 准备输出
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = self.output_dir / f'samples_for_review_{timestamp}'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制选中的文件
            selected_files = []
            for idx in selected_indices:
                src_file = unlabeled_files[idx]
                dst_file = output_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                selected_files.append(str(dst_file))
                
            # 统计信息
            stats = {
                'total_samples': len(unlabeled_files),
                'selected_samples': len(selected_files),
                'diversity_samples': len(selected_diversity),
                'uncertainty_samples': len(uncertainty_indices),
                'random_samples': len(random_indices),
                'mean_uncertainty': float(np.mean(uncertainties)),
                'market_states_distribution': self._get_market_states_stats(market_states),
                'timestamp': timestamp
            }
            
            self.logger.info(f"采样完成，选中 {len(selected_files)} 个样本")
            profit = self.calculate_profit()  # 计算当前盈利
            max_drawdown = self.calculate_max_drawdown()  # 计算最大回撤
            compound_reward = self.reward_calculator.calculate_compound_reward(profit, max_drawdown)
            logger.info(f"当前复合奖励: {compound_reward:.4f}")
            return selected_files, stats
            
        except Exception as e:
            self.logger.error(f"采样过程失败: {str(e)}")
            raise
            
    def _analyze_market_states(self, predictions: List[Dict]) -> np.ndarray:
        """
        分析市场状态
        
        Args:
            predictions: 预测结果列表
            
        Returns:
            市场状态特征矩阵
        """
        states = []
        for pred in predictions:
            # 提取市场特征
            volatility = pred.get('volatility', 0)
            trend = pred.get('trend', 'neutral')
            volume = pred.get('volume_ratio', 1.0)
            
            # 编码市场状态
            state_vector = [
                float(volatility),
                float(trend == 'bullish'),
                float(trend == 'bearish'),
                float(volume)
            ]
            states.append(state_vector)
            
        return np.array(states)
        
    def _diversity_sampling(self, 
                          features: np.ndarray,
                          market_states: np.ndarray) -> np.ndarray:
        """
        多样性采样
        
        Args:
            features: 特征矩阵
            market_states: 市场状态矩阵
            
        Returns:
            选中的样本索引
        """
        from sklearn.cluster import KMeans
        
        # 合并特征和市场状态
        combined_features = np.hstack([features, market_states])
        
        # 标准化特征
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(combined_features)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # 从每个簇中选择样本
        selected_indices = []
        for i in range(self.n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            if len(cluster_indices) > 0:
                # 选择离簇中心最近的样本
                cluster_center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(
                    scaled_features[cluster_indices] - cluster_center,
                    axis=1
                )
                closest_indices = cluster_indices[np.argsort(distances)]
                # 选择前N个样本，N与簇的大小成比例
                n_select = max(1, int(len(cluster_indices) * self.diversity_ratio))
                selected_indices.extend(closest_indices[:n_select])
                
        return np.array(selected_indices)
        
    def _get_market_states_stats(self, market_states: np.ndarray) -> Dict:
        """
        获取市场状态统计信息
        
        Args:
            market_states: 市场状态矩阵
            
        Returns:
            统计信息字典
        """
        return {
            'volatility_mean': float(np.mean(market_states[:, 0])),
            'bullish_ratio': float(np.mean(market_states[:, 1])),
            'bearish_ratio': float(np.mean(market_states[:, 2])),
            'volume_mean': float(np.mean(market_states[:, 3]))
        }
    
    def _calculate_uncertainty(self, predictions: List[Dict]) -> np.ndarray:
        """
        计算预测的不确定性得分
        
        Args:
            predictions: 预测结果列表
            
        Returns:
            不确定性得分数组
        """
        uncertainty_scores = []
        
        for pred in predictions:
            score = 0.0
            
            # 1. 基于置信度的不确定性
            if 'market_structure' in pred and pred['market_structure'].get('has_breakout') is not None:
                confidence = pred.get('confidence', 1.0)
                score += (1 - confidence)
            
            # 2. 基于关键点识别的不确定性
            if 'key_points' in pred:
                null_points = sum(1 for point in pred['key_points'].values() if point is None)
                score += (null_points / len(pred['key_points']))
            
            # 3. 基于预测熵的不确定性
            if 'pattern_features' in pred:
                features = pred['pattern_features']
                if all(v is not None for v in features.values()):
                    # 计算特征预测的熵
                    probs = [float(bool(v)) for v in features.values()]
                    if probs:
                        probs = np.array(probs) / sum(probs)
                        entropy = -np.sum(probs * np.log2(probs + 1e-10))
                        score += entropy
            
            uncertainty_scores.append(score)
        
        return np.array(uncertainty_scores)
    
    def _extract_features(self, prediction: Dict) -> np.ndarray:
        """
        从预测结果中提取特征用于聚类
        
        Args:
            prediction: 预测结果
            
        Returns:
            特征向量
        """
        features = []
        
        # 1. 市场结构特征
        if 'market_structure' in prediction:
            features.append(float(prediction['market_structure'].get('has_breakout', False)))
        
        # 2. 关键点特征
        if 'key_points' in prediction:
            for point in prediction['key_points'].values():
                if point and 'price' in point:
                    features.append(float(point['price']))
                else:
                    features.append(0.0)
        
        # 3. 形态特征
        if 'pattern_features' in prediction:
            features.append(float(prediction['pattern_features'].get('pullback_depth', 0)))
        
        # 4. 交易信号特征
        if 'trading_signal' in prediction:
            signal = prediction['trading_signal']
            features.append(1.0 if signal.get('direction') == 'long' else 0.0)
            if 'entry_zone' in signal:
                features.extend([
                    float(signal['entry_zone'].get('upper', 0)),
                    float(signal['entry_zone'].get('lower', 0))
                ])
        
        return np.array(features)

    def uncertainty_sampling(self, predictions: List[Dict], n_samples: int) -> List[Dict]:
        """
        不确定性采样
        
        Args:
            predictions: 模型预测结果列表
            n_samples: 需要选择的样本数量
        
        Returns:
            不确定性样本列表
        """
        # 按照置信度排序，选择不确定性高的样本
        uncertain_samples = sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:n_samples]
        return uncertain_samples

    def collect_human_feedback(self, feedback_data: List[Dict]) -> None:
        """
        收集人工标注反馈数据
        
        Args:
            feedback_data: 人工标注数据列表
        """
        for feedback in feedback_data:
            # 假设feedback包含样本ID和标注信息
            sample_id = feedback['sample_id']
            label_data = feedback['label_data']
            self.reward_calculator.add_human_label(sample_id, label_data)
            logger.info(f"已收集样本 {sample_id} 的人工标注数据")

    def adjust_human_labeling_ratio(self, performance_metric: float) -> None:
        """
        动态调整人工标注比例
        
        Args:
            performance_metric: 当前模型性能指标
        """
        if performance_metric < self.config.get('performance_threshold', 0.75):
            self.uncertainty_ratio += 0.05  # 增加人工标注比例
            logger.info("模型性能低于阈值，增加人工标注比例")
        else:
            self.uncertainty_ratio = max(0.05, self.uncertainty_ratio - 0.05)  # 减少人工标注比例
            logger.info("模型性能稳定，减少人工标注比例") 