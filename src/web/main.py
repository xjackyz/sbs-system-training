from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import psutil
import GPUtil
import time
from datetime import datetime
from typing import List, Dict, Optional
import os
import json

app = FastAPI(title="SBS Trading System API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 数据模型
class SystemStatus(BaseModel):
    training_system: bool
    trading_analysis: bool
    monitoring_system: bool
    discord_bot: bool

class ResourceUsage(BaseModel):
    gpu_usage: float
    memory_usage: float
    cpu_usage: float
    storage_usage: float

class TrainingMetrics(BaseModel):
    current_epoch: int
    total_epochs: int
    loss: float
    accuracy: float
    training_time: str

class Alert(BaseModel):
    time: str
    message: str
    level: str  # success, info, warning, error

# API路由
@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status():
    """获取系统各组件的运行状态"""
    return {
        "training_system": True,
        "trading_analysis": True,
        "monitoring_system": True,
        "discord_bot": True
    }

@app.get("/api/system/resources", response_model=ResourceUsage)
async def get_resource_usage():
    """获取系统资源使用情况"""
    # GPU使用率
    try:
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0
    except:
        gpu_usage = 0
    
    # CPU使用率
    cpu_usage = psutil.cpu_percent()
    
    # 内存使用率
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    
    # 存储使用率
    disk = psutil.disk_usage('/')
    storage_usage = disk.percent
    
    return {
        "gpu_usage": gpu_usage,
        "memory_usage": memory_usage,
        "cpu_usage": cpu_usage,
        "storage_usage": storage_usage
    }

@app.get("/api/training/metrics", response_model=TrainingMetrics)
async def get_training_metrics():
    """获取训练指标"""
    # 这里应该从训练系统获取实际数据
    return {
        "current_epoch": 23,
        "total_epochs": 100,
        "loss": 0.0234,
        "accuracy": 94.5,
        "training_time": "12h 34m"
    }

@app.get("/api/alerts/recent", response_model=List[Alert])
async def get_recent_alerts():
    """获取最近的告警信息"""
    return [
        {
            "time": "10:23",
            "message": "GPU温度超过阈值(85°C)",
            "level": "warning"
        },
        {
            "time": "09:45",
            "message": "新的模型检查点已保存",
            "level": "info"
        },
        {
            "time": "09:30",
            "message": "训练阶段1完成",
            "level": "success"
        }
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 