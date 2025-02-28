import json
import logging
from typing import Dict
from datetime import datetime

class LogAggregator:
    """日志聚合器"""
    def __init__(self):
        self.logs = []
        self.logger = logging.getLogger('log_aggregator')

    def aggregate(self, log_type: str, message: str, metadata: Dict = None):
        """聚合日志"""
        self.logs.append({
            'type': log_type,
            'message': message,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        })
        self.logger.info(f"日志已聚合: {log_type} - {message}")

    def export_logs(self, format: str = 'json') -> str:
        """导出日志"""
        if format == 'json':
            return json.dumps(self.logs, indent=2)
        return str(self.logs) 