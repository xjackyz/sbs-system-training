#!/usr/bin/env python3
"""GPU显存清理脚本"""

import os
import sys
import torch
import subprocess
import psutil
import time
import logging

def setup_logger():
    """设置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def kill_python_processes():
    """终止所有Python进程"""
    killed = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                # 不要终止清理脚本自身
                if proc.pid != os.getpid():
                    proc.kill()
                    killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed

def clear_gpu_memory():
    """清理GPU显存"""
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        return
        
    try:
        # 清理PyTorch缓存
        torch.cuda.empty_cache()
        
        # 获取GPU信息
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            # 使用torch.device创建正确的设备对象
            device = torch.device(f'cuda:{i}')
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.reset_accumulated_memory_stats(device)
    except Exception as e:
        logger.warning(f"清理GPU显存时出错: {e}")

def get_gpu_memory_info():
    """获取GPU显存信息"""
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        return "GPU不可用"
        
    info = []
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        try:
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(device)
            mem_used = torch.cuda.memory_allocated(device) / 1024**2
            mem_total = props.total_memory / 1024**2
            info.append(f"GPU {i} ({props.name}): 使用 {mem_used:.1f}MB / 总计 {mem_total:.1f}MB")
        except Exception as e:
            info.append(f"GPU {i}: 获取信息失败 - {e}")
    return "\n".join(info)

def main():
    """主函数"""
    logger = setup_logger()
    
    # 显示初始状态
    logger.info("初始GPU状态:")
    logger.info(get_gpu_memory_info())
    
    # 终止Python进程
    killed_pids = kill_python_processes()
    if killed_pids:
        logger.info(f"已终止的Python进程: {killed_pids}")
        # 等待进程完全终止
        time.sleep(2)
    
    # 清理GPU显存
    clear_gpu_memory()
    
    # 显示清理后状态
    logger.info("\n清理后GPU状态:")
    logger.info(get_gpu_memory_info())

if __name__ == "__main__":
    main() 