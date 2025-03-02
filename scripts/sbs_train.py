#!/usr/bin/env python
"""
SBS系统统一训练入口
用于训练SBS预测模型，支持多种训练模式和灵活配置
"""

import os
import sys
import argparse
import logging
import json
import yaml
import time
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# 确保可以导入项目模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config_manager import ConfigManager
from src.self_supervised.utils.logger import setup_logger
from src.self_supervised.model.sbs_predictor import SBSPredictor
from src.self_supervised.utils.reward_calculator import SBSRewardCalculator
from src.self_supervised.utils.trade_tracker import TradeResultTracker
from src.self_supervised.trainer.trainer import SBSTrainer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SBS系统训练脚本")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["standard", "self_supervised", "reinforcement", "active"], 
                        default="standard", help="训练模式")
    parser.add_argument("--batch_size", type=int, help="批处理大小，覆盖配置文件设置")
    parser.add_argument("--epochs", type=int, help="训练轮数，覆盖配置文件设置")
    parser.add_argument("--learning_rate", type=float, help="学习率，覆盖配置文件设置")
    parser.add_argument("--model_name", type=str, help="模型名称，用于保存")
    parser.add_argument("--gpu", type=str, help="指定使用的GPU设备，如'0'或'0,1'")
    parser.add_argument("--checkpoint", type=str, help="加载检查点路径")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        default="INFO", help="日志级别")
    parser.add_argument("--no_validation", action="store_true", help="禁用验证集评估")
    parser.add_argument("--save_interval", type=int, help="保存模型的间隔步数")
    parser.add_argument("--report_interval", type=int, help="报告训练进度的间隔步数")
    
    return parser.parse_args()

def setup_training_environment(args):
    """设置训练环境"""
    # 设置GPU环境变量
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"使用GPU: {args.gpu}")
    
    # 加载配置
    config_manager = ConfigManager(config_name=args.config)
    config = config_manager.get_full_config()
    
    # 用命令行参数覆盖配置
    training_config = config.get("training", {})
    if args.batch_size:
        training_config["batch_size"] = args.batch_size
    if args.epochs:
        training_config["epochs"] = args.epochs
    if args.learning_rate:
        training_config["learning_rate"] = args.learning_rate
    if args.save_interval:
        training_config["save_interval"] = args.save_interval
    if args.report_interval:
        training_config["report_interval"] = args.report_interval
    
    # 设置日志
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{args.mode}_{timestamp}.log")
    
    logger = setup_logger("sbs_train", log_file=log_file, level=args.log_level)
    logger.info(f"开始SBS训练，模式: {args.mode}")
    logger.info(f"命令行参数: {args}")
    
    # 创建模型保存目录
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 更新并保存配置
    config["training"] = training_config
    config_manager.save()
    
    return config, logger

def load_model(config, args, logger):
    """加载或创建模型"""
    model_config = config.get("model", {})
    if args.model_name:
        model_config["name"] = args.model_name
    
    # 创建模型
    model = SBSPredictor(config=model_config)
    
    # 加载检查点
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            logger.info(f"加载检查点: {args.checkpoint}")
            model.load(args.checkpoint)
        else:
            logger.warning(f"检查点不存在: {args.checkpoint}")
    
    return model

def prepare_data(config, args, logger):
    """准备训练和验证数据"""
    # 根据不同的训练模式准备不同的数据
    data_config = config.get("data", {})
    
    if args.mode == "standard":
        # 标准训练模式使用常规数据集
        from src.data.data_loader import DataLoader
        data_loader = DataLoader(data_config)
        train_data = data_loader.load_train_data()
        val_data = None if args.no_validation else data_loader.load_val_data()
    
    elif args.mode == "self_supervised":
        # 自监督训练模式使用自监督数据集
        from src.self_supervised.data.sbs_data_loader import SBSDataLoader
        data_loader = SBSDataLoader(data_config)
        train_data = data_loader.load_self_supervised_data()
        val_data = None if args.no_validation else data_loader.load_val_data()
    
    elif args.mode == "reinforcement":
        # 强化学习训练模式
        from src.self_supervised.data.rl_data_loader import RLDataLoader
        data_loader = RLDataLoader(data_config)
        train_data = data_loader.load_rl_data()
        val_data = None if args.no_validation else data_loader.load_val_data()
    
    elif args.mode == "active":
        # 主动学习训练模式
        from src.self_supervised.data.active_learning_data_loader import ActiveLearningDataLoader
        data_loader = ActiveLearningDataLoader(data_config)
        train_data = data_loader.load_active_learning_data()
        val_data = None if args.no_validation else data_loader.load_val_data()
    
    logger.info(f"训练数据加载完成，样本数: {len(train_data)}")
    if val_data:
        logger.info(f"验证数据加载完成，样本数: {len(val_data)}")
    
    return train_data, val_data

def train_model(model, train_data, val_data, config, args, logger):
    """训练模型"""
    # 创建训练器
    trainer_config = config.get("training", {})
    trainer = SBSTrainer(model=model, config=trainer_config)
    
    # 创建奖励计算器（用于自监督和强化学习训练）
    if args.mode in ["self_supervised", "reinforcement"]:
        reward_calculator = SBSRewardCalculator(config=config.get("reward", {}))
        trainer.set_reward_calculator(reward_calculator)
    
    # 创建交易结果跟踪器（用于强化学习训练）
    if args.mode == "reinforcement":
        trade_tracker = TradeResultTracker(config=config.get("trade", {}))
        trainer.set_trade_tracker(trade_tracker)
    
    # 设置保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get("model", {}).get("name", "sbs_model")
    save_path = os.path.join(project_root, "models", f"{model_name}_{args.mode}_{timestamp}")
    
    # 开始训练
    logger.info(f"开始训练，模式: {args.mode}")
    start_time = time.time()
    
    if args.mode == "standard":
        trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=trainer_config.get("epochs", 10),
            batch_size=trainer_config.get("batch_size", 32),
            save_path=save_path
        )
    
    elif args.mode == "self_supervised":
        trainer.train_self_supervised(
            train_data=train_data,
            val_data=val_data,
            epochs=trainer_config.get("epochs", 10),
            batch_size=trainer_config.get("batch_size", 32),
            save_path=save_path
        )
    
    elif args.mode == "reinforcement":
        trainer.train_with_reinforcement(
            train_data=train_data,
            val_data=val_data,
            epochs=trainer_config.get("epochs", 10),
            batch_size=trainer_config.get("batch_size", 32),
            save_path=save_path
        )
    
    elif args.mode == "active":
        trainer.train_with_active_learning(
            train_data=train_data,
            val_data=val_data,
            epochs=trainer_config.get("epochs", 10),
            batch_size=trainer_config.get("batch_size", 32),
            save_path=save_path
        )
    
    training_time = time.time() - start_time
    logger.info(f"训练完成，耗时: {training_time:.2f}秒")
    logger.info(f"模型已保存到: {save_path}")
    
    return save_path

def evaluate_model(model, save_path, config, args, logger):
    """评估训练后的模型"""
    # 如果禁用验证评估，直接返回
    if args.no_validation:
        logger.info("验证评估已禁用，跳过评估步骤")
        return
    
    logger.info("开始模型评估...")
    
    # 加载模型（确保使用最新的权重）
    model.load(os.path.join(save_path, "best_model.pth"))
    
    # 加载测试数据
    from src.data.data_loader import DataLoader
    data_loader = DataLoader(config.get("data", {}))
    test_data = data_loader.load_test_data()
    
    logger.info(f"测试数据加载完成，样本数: {len(test_data)}")
    
    # 创建评估器
    from src.self_supervised.metrics.evaluator import SBSEvaluator
    evaluator = SBSEvaluator(config=config.get("evaluation", {}))
    
    # 进行评估
    evaluation_results = evaluator.evaluate(model, test_data)
    
    # 保存评估结果
    results_file = os.path.join(save_path, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"评估完成，结果已保存到: {results_file}")
    
    # 打印主要指标
    logger.info("主要评估指标:")
    for key, value in evaluation_results.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
    
    return evaluation_results

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置训练环境
    config, logger = setup_training_environment(args)
    
    try:
        # 加载模型
        model = load_model(config, args, logger)
        
        # 准备数据
        train_data, val_data = prepare_data(config, args, logger)
        
        # 训练模型
        save_path = train_model(model, train_data, val_data, config, args, logger)
        
        # 评估模型
        evaluation_results = evaluate_model(model, save_path, config, args, logger)
        
        logger.info("训练任务完成")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 