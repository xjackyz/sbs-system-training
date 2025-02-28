#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用信号跟踪和奖励机制进行自监督学习训练的示例脚本
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_supervised.model.sequence_model import SequenceModel, ModelConfig
from src.self_supervised.trainer.self_supervised_trainer import SelfSupervisedTrainer
from src.self_supervised.utils.signal_tracker import SignalTracker
from src.self_supervised.utils.reward_mechanism import RewardMechanism
from src.self_supervised.data.data_processor import DataProcessor
from src.utils.logger import setup_logger
from config.config import SELF_SUPERVISED_CONFIG


logger = setup_logger('reward_training')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用信号跟踪和奖励机制进行自监督学习训练')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='训练数据目录')
    parser.add_argument('--model_dir', type=str, default='models/self_supervised',
                       help='模型保存目录')
    parser.add_argument('--stage', type=int, default=1,
                       help='训练阶段(1-3)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数，如果不指定则使用配置中的默认值')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--validate_every', type=int, default=1,
                       help='每多少轮进行一次验证')
    parser.add_argument('--use_llava', action='store_true',
                       help='是否使用LLaVA模型生成信号')
    parser.add_argument('--evaluate', action='store_true',
                       help='是否只进行评估')
    
    return parser.parse_args()


def simulate_trading_signals(trainer, num_signals=10):
    """模拟交易信号
    
    为了演示信号跟踪和奖励机制，生成一些模拟的交易信号
    
    Args:
        trainer: 训练器实例
        num_signals: 生成的信号数量
    """
    logger.info(f"生成 {num_signals} 个模拟交易信号...")
    
    # 模拟图表数据
    for i in range(num_signals):
        # 模拟图表数据
        chart_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': f"BTC/USDT",
            'timeframe': '1h',
            'price_data': {
                'open': np.random.normal(50000, 1000, 100).tolist(),
                'high': np.random.normal(50500, 1000, 100).tolist(),
                'low': np.random.normal(49500, 1000, 100).tolist(),
                'close': np.random.normal(50200, 1000, 100).tolist(),
                'volume': np.random.normal(100, 20, 100).tolist()
            }
        }
        
        # 模拟预测结果
        prediction = {
            'sequence_points': [i for i in range(5)],  # 5个点位
            'direction': 'long' if np.random.random() > 0.5 else 'short',
            'entry_price': 50000 + np.random.normal(0, 200),
            'stop_loss': 49500 + np.random.normal(0, 200),
            'target_price': 51000 + np.random.normal(0, 200)
        }
        
        # 模拟置信度
        confidence = np.random.uniform(0.7, 0.95)
        
        # 记录信号
        signal_id = trainer.record_signal(chart_data, prediction, confidence)
        logger.info(f"记录信号 {i+1}/{num_signals}: {signal_id}, 方向: {prediction['direction']}, 置信度: {confidence:.4f}")
        
        # 模拟信号跟踪
        for day in range(trainer.signal_tracker.tracking_window):
            # 模拟价格数据
            if prediction['direction'] == 'long':
                # 多头信号，价格有70%概率上涨
                if np.random.random() < 0.7:
                    # 上涨
                    price_change = np.random.uniform(0, 500)
                else:
                    # 下跌
                    price_change = -np.random.uniform(0, 700)
            else:
                # 空头信号，价格有70%概率下跌
                if np.random.random() < 0.7:
                    # 下跌
                    price_change = -np.random.uniform(0, 500)
                else:
                    # 上涨
                    price_change = np.random.uniform(0, 700)
            
            # 更新价格
            base_price = prediction['entry_price']
            price_data = {
                'open': base_price,
                'high': base_price + abs(price_change) * 0.5,
                'low': base_price - abs(price_change) * 0.5,
                'close': base_price + price_change
            }
            
            # 更新信号跟踪
            trainer.update_signal(signal_id, price_data)
            
            # 如果信号已经完成评估，提前结束跟踪
            signal = trainer.signal_tracker.get_signal(signal_id)
            if signal['status'] != 'pending':
                logger.info(f"信号 {signal_id} 评估完成: {signal['status']}, 奖励值: {signal['reward_value']:.4f}")
                break
    
    # 打印统计数据
    stats = trainer.get_signal_stats()
    logger.info("信号统计数据:")
    logger.info(f"总信号数: {stats['total_signals']}")
    logger.info(f"成功信号数: {stats['successful_signals']}")
    logger.info(f"失败信号数: {stats['failed_signals']}")
    logger.info(f"待定信号数: {stats['pending_signals']}")
    logger.info(f"胜率: {stats['win_rate']:.4f}")
    logger.info(f"平均盈利: {stats['avg_profit']:.4f}")
    logger.info(f"平均亏损: {stats['avg_loss']:.4f}")
    logger.info(f"盈亏比: {stats['profit_factor']:.4f}")


def train_with_reward_mechanism(args):
    """使用奖励机制进行训练"""
    try:
        # 创建保存目录
        os.makedirs(args.model_dir, exist_ok=True)
        
        # 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        # 创建模型配置
        model_config = ModelConfig(
            image_size=SELF_SUPERVISED_CONFIG['model']['image_size'],
            sequence_length=SELF_SUPERVISED_CONFIG['model']['sequence_length'],
            hidden_size=SELF_SUPERVISED_CONFIG['model']['hidden_size'],
            num_heads=SELF_SUPERVISED_CONFIG['model']['num_heads'],
            num_layers=SELF_SUPERVISED_CONFIG['model']['num_layers']
        )
        
        # 创建模型
        model = SequenceModel(config=model_config)
        
        # 创建训练器
        trainer = SelfSupervisedTrainer(
            model=model,
            data_dir=args.data_dir,
            save_dir=args.model_dir,
            device=device
        )
        
        # 设置训练阶段
        trainer.set_stage(args.stage)
        
        # 如果指定了检查点，加载它
        if args.resume:
            trainer.load_checkpoint(args.resume)
            logger.info(f"从检查点恢复: {args.resume}")
        
        # 模拟交易信号
        simulate_trading_signals(trainer, num_signals=20)
        
        # 如果是评估模式
        if args.evaluate:
            logger.info("开始评估...")
            metrics = trainer.evaluate()
            
            logger.info("评估结果:")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
            
            return
        
        # 确定训练轮数
        num_epochs = args.epochs
        if num_epochs is None:
            # 使用配置中的默认值
            num_epochs = SELF_SUPERVISED_CONFIG['training']['num_epochs']
        
        # 开始训练
        logger.info(f"开始训练，阶段 {args.stage}，共 {num_epochs} 轮...")
        trainer.train(
            num_epochs=num_epochs,
            batch_size=args.batch_size,
            validate_every=args.validate_every
        )
        
        # 训练完成后再次评估
        logger.info("训练完成，开始最终评估...")
        metrics = trainer.evaluate()
        
        logger.info("最终评估结果:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def main():
    """主函数"""
    args = parse_args()
    train_with_reward_mechanism(args)


if __name__ == "__main__":
    main() 