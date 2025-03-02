from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import psutil
import GPUtil
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Set
import os
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'web_api.log'))
    ]
)
logger = logging.getLogger("sbs_web_api")

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

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"新的WebSocket连接, 当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket连接断开, 当前连接数: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"广播消息失败: {str(e)}")
                disconnected.add(connection)
        
        # 清理断开的连接
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

# API路由
@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status():
    """获取系统各组件的运行状态"""
    try:
        # 这里应该从实际系统获取状态
        status = {
            "training_system": True,
            "trading_analysis": True,
            "monitoring_system": True,
            "discord_bot": True
        }
        return status
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")

@app.get("/api/system/resources", response_model=ResourceUsage)
async def get_resource_usage():
    """获取系统资源使用情况"""
    try:
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
    except Exception as e:
        logger.error(f"获取资源使用情况失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取资源使用情况失败: {str(e)}")

@app.get("/api/training/metrics", response_model=TrainingMetrics)
async def get_training_metrics():
    """获取训练指标"""
    try:
        # 这里应该从训练系统获取实际数据
        return {
            "current_epoch": 23,
            "total_epochs": 100,
            "loss": 0.0234,
            "accuracy": 94.5,
            "training_time": "12h 34m"
        }
    except Exception as e:
        logger.error(f"获取训练指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取训练指标失败: {str(e)}")

@app.get("/api/alerts/recent", response_model=List[Alert])
async def get_recent_alerts():
    """获取最近的告警信息"""
    try:
        # 这里应该从日志或数据库中获取实际数据
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
    except Exception as e:
        logger.error(f"获取告警信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取告警信息失败: {str(e)}")

# WebSocket路由
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 客户端发送的消息(如果有)
            data = await websocket.receive_text()
            # 记录接收到的消息
            logger.debug(f"接收到WebSocket消息: {data}")
            
            # 简单的回复确认收到
            await websocket.send_json({"status": "received", "message": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket错误: {str(e)}")
        manager.disconnect(websocket)

# 后台任务:定期广播资源使用情况
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(periodic_resource_broadcast())

async def periodic_resource_broadcast():
    """周期性广播系统资源使用情况"""
    while True:
        try:
            # 获取资源数据
            resources = await get_resource_usage()
            # 广播给所有连接
            await manager.broadcast({
                "type": "resources",
                "data": resources.dict()
            })
            await asyncio.sleep(5)  # 5秒更新一次
        except Exception as e:
            logger.error(f"资源广播任务错误: {str(e)}")
            await asyncio.sleep(10)  # 发生错误时等待10秒

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 