#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SBS自定义训练器模块
为SBS交易系统提供定制的训练逻辑和功能
"""

import torch
import logging
import numpy as np
from pathlib import Path
from transformers import Trainer
import torch.nn.functional as F


class SBSTrainer(Trainer):
    """
    SBS自定义训练器，继承自Huggingface的Trainer
    添加了特定于SBS任务的训练功能和评估度量
    """
    
    def __init__(self, *args, **kwargs):
        """
        初始化SBS训练器
        
        参数继承自Trainer类，可以包括:
            model: 模型
            args: 训练参数
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            tokenizer: 分词器
        """
        super().__init__(*args, **kwargs)
        logging.info("初始化SBS自定义训练器")
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算训练损失
        
        参数:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回模型输出
            
        返回:
            loss: 损失值
            outputs: (可选) 模型输出
        """
        # 确保标签正确设置
        if 'labels' not in inputs:
            inputs['labels'] = inputs['input_ids'].clone()
        
        # 获取模型输出
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 记录详细的损失信息以便调试
        if self.state.global_step % 50 == 0:
            logging.info(f"步骤 {self.state.global_step}: 损失 = {loss.item():.4f}")
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        评估模型性能
        
        参数:
            eval_dataset: 评估数据集
            ignore_keys: 要忽略的键
            metric_key_prefix: 指标键前缀
            
        返回:
            metrics: 评估指标
        """
        # 调用父类的评估方法
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        logging.info(f"评估结果: {metrics}")
        
        return metrics
    
    def save_model(self, output_dir=None, _internal_call=False):
        """
        保存模型
        
        参数:
            output_dir: 输出目录
            _internal_call: 是否为内部调用
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"保存模型到 {output_dir}")
        
        # 调用父类的保存方法
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        
        # 保存训练配置
        self._save_training_config(output_dir)
        
    def _save_training_config(self, output_dir):
        """
        保存训练配置信息
        
        参数:
            output_dir: 输出目录
        """
        config = {
            "train_batch_size": self.args.train_batch_size,
            "eval_batch_size": self.args.eval_batch_size,
            "learning_rate": self.args.learning_rate,
            "num_train_epochs": self.args.num_train_epochs,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
        }
        
        import json
        config_path = Path(output_dir) / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            

class SBSActiveLearner:
    """
    SBS主动学习器
    用于实施主动学习策略，选择最具信息价值的样本进行标注
    """
    
    def __init__(self, model, dataset, processor, device='cuda', batch_size=8):
        """
        初始化主动学习器
        
        参数:
            model: LLaVA模型
            dataset: SBS主动学习数据集
            processor: LLaVA处理器
            device: 设备（'cuda'或'cpu'）
            batch_size: 批处理大小
        """
        self.model = model
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.batch_size = batch_size
        
        # 将模型设置为评估模式
        self.model.eval()
        
    def select_samples(self, num_samples=10, strategy='uncertainty'):
        """
        选择样本进行标注
        
        参数:
            num_samples: 要选择的样本数量
            strategy: 选择策略，'uncertainty'或'random'
            
        返回:
            selected_indices: 选择的样本索引
        """
        # 获取未标记的样本索引
        unlabeled_indices = self.dataset.get_unlabeled_indices()
        
        if len(unlabeled_indices) <= num_samples:
            return unlabeled_indices
        
        if strategy == 'random':
            # 随机选择样本
            selected_indices = np.random.choice(
                unlabeled_indices, 
                size=num_samples, 
                replace=False
            ).tolist()
        
        elif strategy == 'uncertainty':
            # 基于不确定性选择样本
            uncertainties = self._compute_uncertainties(unlabeled_indices)
            
            # 选择不确定性最高的样本
            selected_indices = [unlabeled_indices[i] for i in np.argsort(uncertainties)[-num_samples:]]
            
        else:
            raise ValueError(f"不支持的选择策略: {strategy}")
        
        return selected_indices
    
    def _compute_uncertainties(self, indices):
        """
        计算样本的不确定性
        
        参数:
            indices: 样本索引列表
            
        返回:
            uncertainties: 不确定性得分列表
        """
        uncertainties = []
        
        # 创建数据加载器
        from torch.utils.data import DataLoader, Subset
        subset = Subset(self.dataset, indices)
        dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                # 将数据移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 获取模型输出
                outputs = self.model(**batch)
                logits = outputs.logits
                
                # 计算每个样本的不确定性
                # 这里使用预测熵作为不确定性度量
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                
                # 将不确定性添加到列表
                batch_uncertainties = entropy.mean(dim=1).cpu().numpy()
                uncertainties.extend(batch_uncertainties.tolist())
        
        return uncertainties
    
    def update_model(self, new_model_path):
        """
        更新模型
        
        参数:
            new_model_path: 新模型路径
        """
        from transformers import AutoModelForCausalLM
        
        logging.info(f"加载更新的模型: {new_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            new_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval() 