import os
import yaml
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime
import random
import numpy as np
import sys

# 添加项目根目录到PATH以便导入src模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sbs_model import SBSModel
from models.sbs_data import SBSDataModule
from src.self_supervised.utils.sbs_optimizer import SBSOptimizer  # 导入SBSOptimizer类


def setup_logging(config: Dict[str, Any]) -> None:
    """
    设置日志记录

    参数:
        config: 配置字典
    """
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=config['logging']['level'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件

    参数:
        config_path: 配置文件路径

    返回:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {e}")


def create_loggers(config: Dict[str, Any], trial_number: int) -> list:
    """
    创建日志记录器

    参数:
        config: 配置字典
        trial_number: 试验编号

    返回:
        日志记录器列表
    """
    loggers = []
    
    # TensorBoard logger
    tensorboard = TensorBoardLogger(
        save_dir=config['paths']['log_dir'],
        name=f'trial_{trial_number}',
        version=datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    loggers.append(tensorboard)
    
    # Weights & Biases logger
    if config['tracking']['wandb']['project']:
        wandb = WandbLogger(
            project=config['tracking']['wandb']['project'],
            name=f'trial_{trial_number}',
            config=config
        )
        loggers.append(wandb)
    
    return loggers


def create_callbacks(config: Dict[str, Any], trial: optuna.Trial, trial_dir: Path) -> list:
    """
    创建回调函数

    参数:
        config: 配置字典
        trial: Optuna trial 对象
        trial_dir: 试验目录

    返回:
        回调函数列表
    """
    callbacks = []
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=trial_dir,
        monitor='val_acc',
        mode='max',
        save_top_k=config['training']['checkpointing']['save_top_k'],
        filename='best-{epoch:02d}-{val_acc:.4f}'
    )
    callbacks.append(checkpoint_callback)
    
    # 早停
    early_stopping = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta'],
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Optuna 剪枝
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_acc')
    callbacks.append(pruning_callback)
    
    return callbacks


def objective(trial: optuna.Trial, config: Dict[str, Any]) -> float:
    """
    Optuna 优化目标函数

    参数:
        trial: Optuna trial 对象
        config: 配置字典

    返回:
        验证集准确率
    """
    try:
        # 定义超参数搜索空间
        model_params = {
            'input_size': config.get('model', {}).get('input_size', 6),  # 默认使用配置中的值
            'hidden_size': trial.suggest_int('hidden_size', 64, 256),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }
        
        data_params = {
            'batch_size': trial.suggest_int('batch_size', 16, 128),
            'sequence_length': trial.suggest_int('sequence_length', 30, 120)
        }
        
        # 创建试验目录
        trial_dir = Path(config['paths']['model_dir']) / f'trial_{trial.number}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建数据模块
        try:
            data_module = SBSDataModule(
                data_dir=config['paths']['data_dir'],
                **data_params
            )
        except Exception as e:
            logging.error(f"创建数据模块失败: {e}")
            raise
        
        # 创建模型
        try:
            model = SBSModel(**model_params)
        except Exception as e:
            logging.error(f"创建模型失败: {e}")
            raise
        
        # 创建日志记录器
        loggers = create_loggers(config, trial.number)
        
        # 创建回调函数
        callbacks = create_callbacks(config, trial, trial_dir)
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=config['training']['epochs'],
            accelerator='auto',
            devices='auto',
            logger=loggers,
            callbacks=callbacks,
            deterministic=True,
            log_every_n_steps=config['logging']['log_every_n_steps']
        )
        
        # 训练模型
        trainer.fit(model, data_module)
        
        # 返回最佳验证准确率
        if 'val_acc' in trainer.callback_metrics:
            return trainer.callback_metrics['val_acc'].item()
        else:
            logging.warning("未找到验证准确率指标，返回0.0")
            return 0.0
        
    except Exception as e:
        logging.error(f"试验 {trial.number} 失败: {e}")
        raise optuna.exceptions.TrialPruned()


def save_study_results(study: optuna.Study, config: Dict[str, Any]) -> None:
    """
    保存优化结果

    参数:
        study: Optuna study 对象
        config: 配置字典
    """
    log_dir = Path(config['paths']['log_dir'])
    
    # 保存参数重要性分析
    try:
        importances = optuna.importance.get_param_importances(study)
        with open(log_dir / 'parameter_importance.txt', 'w') as f:
            for param, importance in importances.items():
                f.write(f"{param}: {importance}\n")
    except Exception as e:
        logging.warning(f"保存参数重要性分析失败: {e}")
    
    # 保存所有试验结果
    study.trials_dataframe().to_csv(
        log_dir / 'optimization_results.csv',
        index=False
    )


def set_random_seeds(seed: int) -> None:
    """
    设置所有随机种子，确保可重复性

    参数:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"已设置随机种子: {seed}")


def ensure_directories(paths: Dict[str, str]) -> None:
    """
    确保所有必要的目录都存在

    参数:
        paths: 包含路径的字典
    """
    for name, path in paths.items():
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            logging.info(f"已创建目录: {path}")
        except Exception as e:
            logging.error(f"创建目录 {path} 失败: {e}")
            raise


def main():
    """
    主函数
    """
    try:
        # 加载配置
        config = load_config('config/training_config.yaml')
        
        # 设置日志记录
        setup_logging(config)
        
        # 设置随机种子以确保可重复性
        seed = config.get('seed', 42)
        set_random_seeds(seed)
        
        # 创建必要的目录
        ensure_directories(config['paths'])
        
        # 根据命令行参数确定运行模式
        import argparse
        parser = argparse.ArgumentParser(description='SBS训练和优化脚本')
        parser.add_argument('--mode', type=str, choices=['train', 'optimize', 'test'], 
                            default='train', help='运行模式: train, optimize 或 test')
        args = parser.parse_args()
        
        # 创建数据模块
        data_module = SBSDataModule(
            data_dir=config['paths']['data_dir'],
            batch_size=config['training']['dataloader']['batch_size'],
            sequence_length=config['training']['dataloader']['sequence_length'],
            train_ratio=config['training']['dataloader']['train_ratio'],
            val_ratio=config['training']['dataloader']['val_ratio'],
            num_workers=config['training']['dataloader']['num_workers']
        )
        
        # 准备数据
        data_module.prepare_data()
        data_module.setup()
        
        # 根据运行模式执行不同操作
        if args.mode == 'optimize':
            logging.info("开始超参数优化...")
            # 使用SBSOptimizer进行优化
            optimizer = SBSOptimizer(config)
            best_params = optimizer.optimize(
                data_module.train_dataloader(), 
                data_module.val_dataloader()
            )
            
            # 使用最佳参数训练最终模型
            logging.info(f"使用最佳参数进行最终训练: {best_params}")
            final_model = optimizer.train_with_best_params(
                data_module.train_dataloader(),
                data_module.val_dataloader(),
                data_module.test_dataloader()
            )
            
            # 测试最终模型
            logging.info("最终模型训练和评估完成。")
            
        elif args.mode == 'train':
            # 原始训练代码保持不变
            logging.info("开始常规训练...")
            
            # 创建 Optuna study
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(),
                study_name=config['optuna']['study_name']
            )
            
            # 运行优化
            study.optimize(
                lambda trial: objective(trial, config),
                n_trials=config['optuna']['n_trials'],
                timeout=config['optuna']['timeout_seconds'],
                n_jobs=1
            )
            
            # 打印优化结果
            logging.info(f"最佳准确率: {study.best_value:.4f}")
            logging.info(f"最佳参数: {study.best_params}")
            
            # 保存优化结果
            save_study_results(study, config)
            
            # 使用最佳参数训练最终模型
            logging.info("使用最佳参数训练最终模型...")
            best_model_params = {
                'input_size': config.get('model', {}).get('input_size', 6),  # 默认使用配置中的值
                'hidden_size': study.best_params['hidden_size'],
                'num_layers': study.best_params['num_layers'],
                'dropout': study.best_params['dropout'],
                'learning_rate': study.best_params['learning_rate']
            }
            
            best_data_params = {
                'batch_size': study.best_params['batch_size'],
                'sequence_length': study.best_params['sequence_length']
            }
            
            # 创建最终模型目录
            final_model_dir = Path(config['paths']['model_dir']) / 'final_model'
            final_model_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建数据模块
            data_module = SBSDataModule(
                data_dir=config['paths']['data_dir'],
                **best_data_params
            )
            
            # 创建模型
            final_model = SBSModel(**best_model_params)
            
            # 创建最终模型的检查点回调
            final_checkpoint = ModelCheckpoint(
                dirpath=final_model_dir,
                filename='best-model',
                monitor='val_acc',
                mode='max',
                save_top_k=1
            )
            
            # 创建训练器
            final_trainer = pl.Trainer(
                max_epochs=config['training']['final_epochs'],
                accelerator='auto',
                devices='auto',
                logger=create_loggers(config, 'final'),
                callbacks=[
                    final_checkpoint,
                    EarlyStopping(
                        monitor='val_acc',
                        mode='max',
                        patience=config['training']['early_stopping']['patience'],
                        min_delta=config['training']['early_stopping']['min_delta'],
                        verbose=True
                    ),
                    LearningRateMonitor(logging_interval='step')
                ],
                deterministic=True,
                log_every_n_steps=config['logging']['log_every_n_steps']
            )
            
            # 训练最终模型
            final_trainer.fit(final_model, data_module)
            
            # 测试最终模型
            test_result = final_trainer.test(final_model, data_module)
            logging.info(f"最终模型测试结果: {test_result}")
            
            # 保存最终模型
            final_model_path = final_model_dir / 'sbs_model.pt'
            torch.save(final_model.state_dict(), final_model_path)
            logging.info(f"最终模型已保存到: {final_model_path}")
            
            logging.info("训练完成！")
            
        elif args.mode == 'test':
            logging.info("开始测试模式...")
            # 加载已训练的模型并进行测试
            model_path = config.get('paths', {}).get('model_dir', 'models/checkpoints') + '/final_model/best-model.ckpt'
            if os.path.exists(model_path):
                model = SBSModel.load_from_checkpoint(model_path)
                
                # 创建测试训练器
                test_trainer = pl.Trainer(
                    accelerator='auto',
                    devices='auto',
                    logger=False
                )
                
                # 运行测试
                test_result = test_trainer.test(model, data_module)
                logging.info(f"测试结果: {test_result}")
            else:
                logging.error(f"模型文件不存在: {model_path}")
        
        logging.info("执行完成！")
        
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 