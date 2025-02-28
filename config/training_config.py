"""
训练配置文件
包含所有训练相关的配置参数
"""

# GPU 相关设置
GPU_SETTINGS = {
    'batch_size': 32,
    'gradient_accumulation_steps': 4,
    'mixed_precision': 'fp16',
    'max_seq_length': 512,
    'memory_efficient_attention': True,
    'gradient_checkpointing': True
}

# 训练阶段配置
TRAINING_STAGES = {
    'visual_pretrain': {
        'description': '视觉特征预训练阶段',
        'epochs': 10,
        'learning_rate': 1e-4,
        'batch_size': 64,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'focus': 'visual_features',
        'validation_frequency': 100
    },
    'sequence_recognition': {
        'description': 'SBS序列识别阶段',
        'epochs': 5,
        'learning_rate': 5e-5,
        'batch_size': 32,
        'warmup_steps': 500,
        'weight_decay': 0.02,
        'focus': 'sequence_patterns',
        'validation_frequency': 50
    },
    'sequence_finetuning': {
        'description': 'SBS序列微调阶段',
        'epochs': 3,
        'learning_rate': 2e-5,
        'batch_size': 16,
        'warmup_steps': 200,
        'weight_decay': 0.03,
        'focus': 'sequence_understanding',
        'validation_frequency': 25
    }
}

# 基准验证配置
BASELINE_VALIDATION = {
    'period': '1d',
    'metrics': ['accuracy', 'loss', 'precision', 'recall', 'f1'],
    'threshold': {
        'accuracy': 0.8,
        'loss': 0.5,
        'precision': 0.75,
        'recall': 0.75,
        'f1': 0.8
    }
}

# 验证配置
VALIDATION = {
    'frequency': 100,  # 每训练多少步验证一次
    'metrics_threshold': {
        'loss': 0.5,
        'accuracy': 0.8,
        'precision': 0.75,
        'recall': 0.75,
        'f1': 0.8
    },
    'early_stopping': {
        'patience': 3,
        'min_delta': 0.001
    }
}

# 模型配置
MODEL_CONFIG = {
    'model_type': 'transformer',
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'intermediate_size': 3072,
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'visual_encoder': {
        'type': 'resnet50',
        'pretrained': True,
        'freeze_layers': ['layer1', 'layer2']
    }
}

# 数据配置
DATA_CONFIG = {
    'train_test_split': 0.8,
    'validation_split': 0.1,
    'sequence_length': 128,
    'stride': 64,
    'random_seed': 42,
    'augmentation': {
        'enabled': True,
        'methods': ['random_crop', 'color_jitter', 'random_flip'],
        'probability': 0.5
    }
}

# 优化器配置
OPTIMIZER_CONFIG = {
    'type': 'AdamW',
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    'weight_decay': 0.01,
    'learning_rate': {
        'initial': 1e-4,
        'min': 1e-6,
        'scheduler': 'linear_warmup_cosine_decay'
    },
    'gradient_clipping': {
        'enabled': True,
        'max_norm': 1.0
    }
}

# 训练过程配置
TRAINING_PROCESS = {
    'max_steps': 100000,
    'save_steps': 1000,
    'logging_steps': 100,
    'eval_steps': 1000,
    'gradient_accumulation_steps': 4,
    'mixed_precision': 'fp16',
    'checkpointing': {
        'enabled': True,
        'save_best_only': True,
        'metric': 'val_loss'
    }
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_dir': 'logs',
    'tensorboard': {
        'enabled': True,
        'log_dir': 'logs/tensorboard',
        'update_freq': 'epoch'
    }
}

# 添加渲染配置
RENDER_SETTINGS = {
    'width': 1920,
    'height': 1080,
    'cache_dir': 'data/cache/images',
    'add_indicators': True,
    'window_size': 100,
    'step_size': 20,
    'use_gpu': True,
    'prerender': True,
    'cache_size_limit': 10000,  # 最大缓存图片数量
    'colors': {
        'up': [1.0, 0.2, 0.2],  # 红色
        'down': [0.2, 1.0, 0.2],  # 绿色
        'grid': [0.2, 0.2, 0.2],  # 网格颜色
        'background': [0.1, 0.1, 0.1]  # 背景颜色
    }
} 