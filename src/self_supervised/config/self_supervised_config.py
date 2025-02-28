#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自监督学习模块的配置文件
"""

SELF_SUPERVISED_CONFIG = {
    # 模型配置
    'model': {
        'image_size': 224,           # 输入图像大小
        'sequence_length': 10,        # 序列长度
        'hidden_size': 256,           # 隐藏层大小
        'num_heads': 4,               # 注意力头数量
        'num_layers': 4,              # Transformer层数量
        'dropout_rate': 0.1,          # Dropout比率
        'activation': 'gelu',         # 激活函数
    },
    
    # 训练配置
    'training': {
        'num_epochs': 50,             # 训练轮数
        'batch_size': 128,            # 批次大小
        'effective_batch_size': 256,   # 增加有效批次大小
        'learning_rate': 2e-4,        # 学习率
        'weight_decay': 1e-5,         # 权重衰减
        'lr_scheduler': 'cosine',     # 学习率调度器类型
        'warmup_steps': 1000,         # 预热步数
        'validate_every': 1,          # 每多少轮验证一次
        'early_stopping_patience': 5, # 早停耐心值
        'gradient_clip_val': 1.0,     # 梯度裁剪值
        'gradient_accumulation_steps': 2,  # 梯度累积
        'num_workers': 8,            # 数据加载器工作进程数
        'prefetch_factor': 3,        # 数据预取因子
        'pin_memory': True,          # 启用内存锁定
    },
    
    # 数据配置
    'data': {
        'train_ratio': 0.8,           # 训练集比例
        'val_ratio': 0.1,             # 验证集比例
        'test_ratio': 0.1,            # 测试集比例
        'augmentation': True,         # 是否使用数据增强
        'normalize': True,            # 是否归一化数据
        'shuffle': True,              # 是否打乱数据
        'num_workers': 4,             # 数据加载器工作进程数
    },
    
    # 信号跟踪配置
    'signal_tracker': {
        'tracking_window': 10,        # 跟踪窗口大小（天数）
        'min_confidence': 0.7,        # 最小置信度
        'success_threshold': 0.5,     # 成功阈值（相对于目标价格）
        'failure_threshold': 1.0,     # 失败阈值（相对于止损）
    },
    
    # 奖励机制配置
    'reward_mechanism': {
        'success_reward': 1.0,        # 成功信号奖励
        'failure_penalty': -0.5,      # 失败信号惩罚
        'rr_bonus_threshold': 2.0,    # 风险回报比奖励阈值
        'rr_bonus_value': 0.5,        # 风险回报比奖励值
        'confidence_weight': 0.3,     # 置信度权重
        'sample_weight_factor': 2.0,  # 样本权重因子
        'curriculum_thresholds': [0.3, 0.5, 0.7],  # 课程学习阈值
        'pseudo_label_confidence': 0.9,  # 伪标签置信度阈值
    },
    
    # 日志配置
    'logging': {
        'log_dir': 'logs/self_supervised',  # 日志目录
        'log_level': 'INFO',                # 日志级别
        'log_to_file': True,                # 是否记录到文件
        'log_to_console': True,             # 是否记录到控制台
    },
    
    # 保存配置
    'saving': {
        'save_dir': 'models/self_supervised',  # 保存目录
        'save_every': 5,                       # 每多少轮保存一次
        'save_best': True,                     # 是否保存最佳模型
        'save_last': True,                     # 是否保存最后一个模型
        'max_to_keep': 3,                      # 最多保存多少个检查点
    },
    
    # 阶段配置
    'stages': {
        # 阶段1：序列识别
        '1': {
            'loss_weights': {
                'sequence': 1.0,
                'market_structure': 0.0,
                'signal': 0.0,
                'price': 0.0,
            },
            'learning_rate': 1e-4,
            'num_epochs': 30,
        },
        # 阶段2：市场结构分析
        '2': {
            'loss_weights': {
                'sequence': 0.3,
                'market_structure': 1.0,
                'signal': 0.0,
                'price': 0.0,
            },
            'learning_rate': 5e-5,
            'num_epochs': 30,
        },
        # 阶段3：交易信号生成
        '3': {
            'loss_weights': {
                'sequence': 0.1,
                'market_structure': 0.3,
                'signal': 1.0,
                'price': 0.5,
            },
            'learning_rate': 1e-5,
            'num_epochs': 50,
        },
    },
    
    'sbs_collection_dir': '/home/easyai/桌面/sbs_system/sbs_sequences',
    'discord_webhook': None,
    
    # 优化内存配置
    'memory_optimization': {
        'max_memory_usage': 80,  # 最大内存使用率(%)
        'clear_cache_frequency': 50,  # 清理缓存频率(步数)
        'enable_amp': True,  # 启用自动混合精度
        'enable_gradient_checkpointing': True,  # 启用梯度检查点
    }
} 