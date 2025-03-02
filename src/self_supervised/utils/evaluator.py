#!/usr/bin/env python
"""
SBS评估器
用于评估SBS序列预测模型的性能
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from datetime import datetime

from ..utils.logger import setup_logger

logger = setup_logger('sbs_evaluator')

class SBSEvaluator:
    """SBS序列预测模型评估器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化评估器
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        self.results_dir = Path(self.config.get('results_dir', 'results/evaluation'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 评估指标
        self.metrics = {}
        
        # 评估结果
        self.predictions = []
        self.ground_truth = []
        self.trade_results = []
        
    def evaluate_model(self, model, dataloader, device):
        """
        评估模型性能
        
        参数:
            model: 要评估的模型
            dataloader: 数据加载器
            device: 计算设备
            
        返回:
            评估指标字典
        """
        logger.info("开始评估模型...")
        
        model.eval()
        self.predictions = []
        self.ground_truth = []
        
        total_loss = 0.0
        sample_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                
                # 计算损失
                loss = model.calculate_loss(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                sample_count += inputs.size(0)
                
                # 收集预测结果和真实标签
                preds = model.get_predictions(outputs)
                
                self.predictions.append(preds.cpu().numpy())
                self.ground_truth.append(labels.cpu().numpy())
                
        # 合并批次结果
        self.predictions = np.concatenate(self.predictions, axis=0)
        self.ground_truth = np.concatenate(self.ground_truth, axis=0)
        
        # 计算评估指标
        self._calculate_metrics()
        
        # 添加平均损失
        self.metrics['loss'] = total_loss / max(1, sample_count)
        
        logger.info(f"模型评估完成, 指标: {self.metrics}")
        
        return self.metrics
        
    def evaluate_trades(self, trade_tracker):
        """
        评估交易结果
        
        参数:
            trade_tracker: 交易跟踪器实例
            
        返回:
            交易评估指标
        """
        logger.info("评估交易结果...")
        
        # 获取已完成的交易
        completed_trades = trade_tracker.get_completed_trades()
        self.trade_results = completed_trades
        
        if not completed_trades:
            logger.warning("没有已完成的交易用于评估")
            self.metrics.update({
                'trade_count': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_profit': 0.0,
                'max_drawdown': 0.0
            })
            return self.metrics
            
        # 计算交易指标
        total_trades = len(completed_trades)
        winning_trades = sum(1 for trade in completed_trades if trade['profit'] > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算总盈利和总亏损
        total_profit = sum(trade['profit'] for trade in completed_trades if trade['profit'] > 0)
        total_loss = abs(sum(trade['profit'] for trade in completed_trades if trade['profit'] < 0))
        
        # 计算盈亏比
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # 计算平均盈利
        avg_profit = sum(trade['profit'] for trade in completed_trades) / total_trades if total_trades > 0 else 0
        
        # 计算最大回撤
        balances = [0]
        for trade in completed_trades:
            balances.append(balances[-1] + trade['profit'])
            
        cumulative_max = np.maximum.accumulate(balances)
        drawdowns = cumulative_max - balances
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # 更新指标
        trade_metrics = {
            'trade_count': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'max_drawdown': max_drawdown,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }
        
        self.metrics.update(trade_metrics)
        
        logger.info(f"交易评估完成: 总交易 {total_trades}, 胜率 {win_rate:.2f}, 盈亏比 {profit_factor:.2f}")
        
        return self.metrics
        
    def _calculate_metrics(self):
        """计算评估指标"""
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            logger.warning("没有预测结果或真实标签用于计算指标")
            return
            
        # 分离SBS序列点位和方向
        # 假设格式：[点1_x, 点1_y, ..., 点5_x, 点5_y, 方向]
        sequence_preds = self.predictions[:, :-1]
        sequence_truth = self.ground_truth[:, :-1]
        
        direction_preds = np.argmax(self.predictions[:, -1].reshape(-1, 2), axis=1) if self.predictions.shape[-1] > 1 else np.round(self.predictions[:, -1])
        direction_truth = np.argmax(self.ground_truth[:, -1].reshape(-1, 2), axis=1) if self.ground_truth.shape[-1] > 1 else self.ground_truth[:, -1]
        
        # 计算序列点位的平均误差
        sequence_error = np.mean(np.abs(sequence_preds - sequence_truth))
        
        # 计算方向预测的精确率、召回率、F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(
            direction_truth, direction_preds, average='binary'
        )
        
        # 计算准确率
        accuracy = accuracy_score(direction_truth, direction_preds)
        
        # 计算混淆矩阵
        cm = confusion_matrix(direction_truth, direction_preds)
        
        # 更新指标
        self.metrics.update({
            'sequence_error': float(sequence_error),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist()
        })
        
    def save_results(self, filename: str = None):
        """
        保存评估结果
        
        参数:
            filename: 文件名，默认为时间戳命名
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.json"
            
        file_path = self.results_dir / filename
        
        # 将NumPy数组转换为列表
        results = {
            'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.metrics.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存预测结果样本（最多100个）
        sample_size = min(100, len(self.predictions))
        if sample_size > 0:
            results['predictions_sample'] = self.predictions[:sample_size].tolist()
            results['ground_truth_sample'] = self.ground_truth[:sample_size].tolist()
        
        # 保存交易结果样本（最多100个）
        trade_sample_size = min(100, len(self.trade_results))
        if trade_sample_size > 0:
            results['trade_results_sample'] = self.trade_results[:trade_sample_size]
        
        try:
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=4)
                
            logger.info(f"评估结果已保存至: {file_path}")
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")
            
    def visualize_results(self, save_path: str = None):
        """
        可视化评估结果
        
        参数:
            save_path: 保存路径，如果提供则保存图表
        """
        if not self.metrics:
            logger.warning("没有指标用于可视化")
            return
            
        # 创建图表
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 混淆矩阵
        if 'confusion_matrix' in self.metrics:
            cm = np.array(self.metrics['confusion_matrix'])
            axs[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axs[0, 0].set_title('混淆矩阵')
            axs[0, 0].set_xticks([0, 1])
            axs[0, 0].set_yticks([0, 1])
            axs[0, 0].set_xticklabels(['下跌', '上涨'])
            axs[0, 0].set_yticklabels(['下跌', '上涨'])
            
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axs[0, 0].text(j, i, str(cm[i, j]), 
                             horizontalalignment="center", 
                             verticalalignment="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        # 2. 准确率指标
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        values = [self.metrics.get(m, 0) for m in metrics_to_plot]
        
        axs[0, 1].bar(metrics_to_plot, values)
        axs[0, 1].set_title('准确率指标')
        axs[0, 1].set_ylim(0, 1)
        for i, v in enumerate(values):
            axs[0, 1].text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        # 3. 交易指标
        if 'trade_count' in self.metrics and self.metrics['trade_count'] > 0:
            trade_metrics = ['win_rate', 'profit_factor', 'avg_profit']
            trade_values = [self.metrics.get(m, 0) for m in trade_metrics]
            
            axs[1, 0].bar(trade_metrics, trade_values)
            axs[1, 0].set_title('交易指标')
            for i, v in enumerate(trade_values):
                axs[1, 0].text(i, v + 0.02, f"{v:.2f}", ha='center')
                
            # 4. 交易分布
            winning = self.metrics.get('winning_trades', 0)
            losing = self.metrics.get('losing_trades', 0)
            
            axs[1, 1].pie([winning, losing], labels=['盈利交易', '亏损交易'], autopct='%1.1f%%')
            axs[1, 1].set_title('交易分布')
        else:
            axs[1, 0].text(0.5, 0.5, '没有交易数据', ha='center', va='center')
            axs[1, 1].text(0.5, 0.5, '没有交易数据', ha='center', va='center')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"评估可视化结果已保存至: {save_path}")
        else:
            plt.show()
            
    def compare_models(self, model_results: List[Dict], model_names: List[str], save_path: str = None):
        """
        比较多个模型的性能
        
        参数:
            model_results: 多个模型的评估结果列表
            model_names: 模型名称列表
            save_path: 保存路径，如果提供则保存图表
        """
        if len(model_results) != len(model_names):
            logger.error("模型结果数量与名称数量不匹配")
            return
            
        if len(model_results) == 0:
            logger.warning("没有模型结果用于比较")
            return
            
        # 创建图表
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. F1分数比较
        f1_scores = [result.get('f1', 0) for result in model_results]
        axs[0, 0].bar(model_names, f1_scores)
        axs[0, 0].set_title('F1分数比较')
        axs[0, 0].set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            axs[0, 0].text(i, v + 0.02, f"{v:.2f}", ha='center')
            
        # 2. 准确率比较
        accuracies = [result.get('accuracy', 0) for result in model_results]
        axs[0, 1].bar(model_names, accuracies)
        axs[0, 1].set_title('准确率比较')
        axs[0, 1].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axs[0, 1].text(i, v + 0.02, f"{v:.2f}", ha='center')
            
        # 3. 序列误差比较
        seq_errors = [result.get('sequence_error', 0) for result in model_results]
        axs[1, 0].bar(model_names, seq_errors)
        axs[1, 0].set_title('序列误差比较 (越低越好)')
        for i, v in enumerate(seq_errors):
            axs[1, 0].text(i, v + 0.02, f"{v:.2f}", ha='center')
            
        # 4. 交易胜率比较
        win_rates = [result.get('win_rate', 0) for result in model_results]
        axs[1, 1].bar(model_names, win_rates)
        axs[1, 1].set_title('交易胜率比较')
        axs[1, 1].set_ylim(0, 1)
        for i, v in enumerate(win_rates):
            axs[1, 1].text(i, v + 0.02, f"{v:.2f}", ha='center')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"模型比较结果已保存至: {save_path}")
        else:
            plt.show()
            
    def performance_report(self, output_format: str = 'text'):
        """
        生成性能报告
        
        参数:
            output_format: 输出格式，可选 'text' 或 'html'
            
        返回:
            性能报告字符串
        """
        if not self.metrics:
            return "没有可用的评估指标"
            
        if output_format == 'html':
            # HTML格式报告
            report = """
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; }
                    table { border-collapse: collapse; width: 80%; margin: 20px auto; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .section { margin: 20px; }
                    h1, h2 { color: #333; }
                </style>
            </head>
            <body>
                <h1 align="center">SBS模型性能报告</h1>
                <div class="section">
                    <h2>预测准确率</h2>
                    <table>
                        <tr>
                            <th>指标</th>
                            <th>值</th>
                        </tr>
            """
            
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'sequence_error']:
                if metric in self.metrics:
                    report += f"""
                        <tr>
                            <td>{metric}</td>
                            <td>{self.metrics[metric]:.4f}</td>
                        </tr>
                    """
                    
            report += """
                    </table>
                </div>
            """
            
            # 如果有交易数据，添加交易部分
            if 'trade_count' in self.metrics and self.metrics['trade_count'] > 0:
                report += """
                <div class="section">
                    <h2>交易性能</h2>
                    <table>
                        <tr>
                            <th>指标</th>
                            <th>值</th>
                        </tr>
                """
                
                for metric in ['trade_count', 'win_rate', 'profit_factor', 'avg_profit', 'max_drawdown']:
                    if metric in self.metrics:
                        report += f"""
                            <tr>
                                <td>{metric}</td>
                                <td>{self.metrics[metric]:.4f}</td>
                            </tr>
                        """
                        
                report += """
                    </table>
                </div>
                """
                
            report += """
            </body>
            </html>
            """
            
            return report
        else:
            # 文本格式报告
            report = "=" * 50 + "\n"
            report += "SBS模型性能报告\n"
            report += "=" * 50 + "\n\n"
            
            report += "预测准确率:\n"
            report += "-" * 30 + "\n"
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'sequence_error']:
                if metric in self.metrics:
                    report += f"{metric}: {self.metrics[metric]:.4f}\n"
                    
            if 'trade_count' in self.metrics and self.metrics['trade_count'] > 0:
                report += "\n交易性能:\n"
                report += "-" * 30 + "\n"
                for metric in ['trade_count', 'win_rate', 'profit_factor', 'avg_profit', 'max_drawdown']:
                    if metric in self.metrics:
                        report += f"{metric}: {self.metrics[metric]:.4f}\n"
                        
            report += "\n" + "=" * 50
            
            return report 