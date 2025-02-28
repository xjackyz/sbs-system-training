"""
Label Studio集成模块
实现与Label Studio的交互和数据同步
"""

import requests
import json
from typing import Dict, List, Optional
from ..utils.logger import setup_logger

logger = setup_logger('label_studio_integration')

class LabelStudioIntegration:
    """Label Studio集成类"""
    
    def __init__(self, config: Dict):
        """
        初始化Label Studio集成
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.base_url = config['label_studio_url']
        self.api_key = config['label_studio_api_key']
        self.headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        }
        
    def create_project(self, name: str, description: str) -> Dict:
        """
        创建标注项目
        
        Args:
            name: 项目名称
            description: 项目描述
            
        Returns:
            项目信息
        """
        try:
            data = {
                'title': name,
                'description': description,
                'label_config': self._get_label_config()
            }
            
            response = requests.post(
                f'{self.base_url}/api/projects',
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"创建项目失败: {str(e)}")
            raise
            
    def import_tasks(self, project_id: int, tasks: List[Dict]) -> Dict:
        """
        导入标注任务
        
        Args:
            project_id: 项目ID
            tasks: 任务列表
            
        Returns:
            导入结果
        """
        try:
            response = requests.post(
                f'{self.base_url}/api/projects/{project_id}/import',
                headers=self.headers,
                json=tasks
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"导入任务失败: {str(e)}")
            raise
            
    def export_annotations(self, project_id: int) -> List[Dict]:
        """
        导出标注结果
        
        Args:
            project_id: 项目ID
            
        Returns:
            标注结果列表
        """
        try:
            response = requests.get(
                f'{self.base_url}/api/projects/{project_id}/export',
                headers=self.headers
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"导出标注结果失败: {str(e)}")
            raise
            
    def sync_predictions(self, project_id: int, predictions: List[Dict]) -> Dict:
        """
        同步模型预测结果
        
        Args:
            project_id: 项目ID
            predictions: 预测结果列表
            
        Returns:
            同步结果
        """
        try:
            response = requests.post(
                f'{self.base_url}/api/projects/{project_id}/predictions',
                headers=self.headers,
                json=predictions
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"同步预测结果失败: {str(e)}")
            raise
            
    def get_project_stats(self, project_id: int) -> Dict:
        """
        获取项目统计信息
        
        Args:
            project_id: 项目ID
            
        Returns:
            统计信息
        """
        try:
            response = requests.get(
                f'{self.base_url}/api/projects/{project_id}/statistics',
                headers=self.headers
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"获取项目统计信息失败: {str(e)}")
            raise
            
    def _get_label_config(self) -> str:
        """
        获取标注配置
        
        Returns:
            标注配置XML
        """
        return """
        <View>
          <Image name="image" value="$image"/>
          <KeyPointLabels name="points" toName="image">
            <Label value="point1" background="red"/>
            <Label value="point2" background="blue"/>
            <Label value="point3" background="green"/>
          </KeyPointLabels>
        </View>
        """ 