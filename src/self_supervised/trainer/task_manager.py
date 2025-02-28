from typing import List, Dict, Optional
from datetime import datetime
import json
import os
from enum import Enum
import threading
import queue
import logging

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

class TrainingTask:
    def __init__(self, task_id: str, config: Dict):
        self.task_id = task_id
        self.config = config
        self.status = TaskStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.current_epoch = 0
        self.metrics = {}
        self.error = None

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "config": self.config,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_epoch": self.current_epoch,
            "metrics": self.metrics,
            "error": str(self.error) if self.error else None
        }

class TaskManager:
    def __init__(self, save_dir: str = "tasks"):
        self.save_dir = save_dir
        self.tasks: Dict[str, TrainingTask] = {}
        self.task_queue = queue.PriorityQueue()
        self.current_task: Optional[TrainingTask] = None
        self.lock = threading.Lock()
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载已有任务
        self._load_tasks()
        
    def _load_tasks(self):
        """加载已保存的任务"""
        try:
            task_file = os.path.join(self.save_dir, "tasks.json")
            if os.path.exists(task_file):
                with open(task_file, "r") as f:
                    tasks_data = json.load(f)
                    for task_data in tasks_data:
                        task = TrainingTask(task_data["task_id"], task_data["config"])
                        task.status = TaskStatus(task_data["status"])
                        if task_data["start_time"]:
                            task.start_time = datetime.fromisoformat(task_data["start_time"])
                        if task_data["end_time"]:
                            task.end_time = datetime.fromisoformat(task_data["end_time"])
                        task.current_epoch = task_data["current_epoch"]
                        task.metrics = task_data["metrics"]
                        task.error = task_data["error"]
                        self.tasks[task.task_id] = task
        except Exception as e:
            logging.error(f"加载任务失败: {e}")

    def _save_tasks(self):
        """保存任务到文件"""
        try:
            task_file = os.path.join(self.save_dir, "tasks.json")
            with open(task_file, "w") as f:
                tasks_data = [task.to_dict() for task in self.tasks.values()]
                json.dump(tasks_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"保存任务失败: {e}")

    def add_task(self, config: Dict, priority: int = 0) -> str:
        """添加新训练任务"""
        with self.lock:
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            task = TrainingTask(task_id, config)
            self.tasks[task_id] = task
            self.task_queue.put((priority, task))
            self._save_tasks()
            return task_id

    def get_next_task(self) -> Optional[TrainingTask]:
        """获取下一个待执行的任务"""
        try:
            _, task = self.task_queue.get_nowait()
            self.current_task = task
            return task
        except queue.Empty:
            return None

    def update_task_status(self, task_id: str, status: TaskStatus, 
                          metrics: Dict = None, error: str = None):
        """更新任务状态"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                if status == TaskStatus.RUNNING and not task.start_time:
                    task.start_time = datetime.now()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    task.end_time = datetime.now()
                if metrics:
                    task.metrics.update(metrics)
                if error:
                    task.error = error
                self._save_tasks()

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        return None

    def get_all_tasks(self) -> List[Dict]:
        """获取所有任务"""
        return [task.to_dict() for task in self.tasks.values()]

    def interrupt_task(self, task_id: str):
        """中断任务"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.INTERRUPTED
                task.end_time = datetime.now()
                self._save_tasks()

    def resume_task(self, task_id: str, priority: int = 0):
        """恢复中断的任务"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == TaskStatus.INTERRUPTED:
                    task.status = TaskStatus.PENDING
                    self.task_queue.put((priority, task))
                    self._save_tasks()

    def clean_completed_tasks(self, keep_days: int = 7):
        """清理已完成的任务"""
        with self.lock:
            current_time = datetime.now()
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    if task.end_time:
                        days_diff = (current_time - task.end_time).days
                        if days_diff > keep_days:
                            to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]
            
            self._save_tasks() 