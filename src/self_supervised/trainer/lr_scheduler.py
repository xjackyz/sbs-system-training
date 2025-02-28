import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=1e-3,
                 min_lr=1e-6, warmup_steps=0, gamma=1., last_epoch=-1):
        """
        初始化Cosine Annealing with Warm Restarts学习率调度器
        
        Args:
            optimizer: 优化器
            first_cycle_steps: 第一个周期的步数
            cycle_mult: 周期长度的乘数
            max_lr: 最大学习率
            min_lr: 最小学习率
            warmup_steps: 预热步数
            gamma: 学习率衰减因子
            last_epoch: 上一轮的epoch
        """
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
        
    def init_lr(self):
        """初始化学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
    
    def get_lr(self):
        """计算当前学习率"""
        if self.step_in_cycle == -1:
            return [self.min_lr for _ in self.base_lrs]
            
        # 预热阶段
        if self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - self.min_lr) * self.step_in_cycle / self.warmup_steps + self.min_lr
                    for _ in self.base_lrs]
            
        # Cosine退火阶段
        progress = (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
        return [self.min_lr + 0.5 * (self.max_lr - self.min_lr) * 
                (1 + math.cos(math.pi * progress)) for _ in self.base_lrs]
    
    def step(self, epoch=None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            
            # 检查是否需要开始新的周期
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
                self.max_lr = self.max_lr * self.gamma
        else:
            self.step_in_cycle = epoch
            
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
    def get_last_lr(self):
        """获取最后一次的学习率"""
        return self.get_lr()

def create_scheduler(optimizer, config):
    """
    创建学习率调度器
    
    Args:
        optimizer: PyTorch优化器
        config: 调度器配置字典
    """
    return CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=config['params']['T_0'],
        cycle_mult=config['params']['T_mult'],
        max_lr=config['params']['max_lr'],
        min_lr=config['params']['min_lr'],
        warmup_steps=config['warmup']['steps'],
        gamma=config['decay']['factor']
    ) 