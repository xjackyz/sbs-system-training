#!/usr/bin/env python
"""
Label Studio服务
用于与Label Studio API交互，管理标注项目和任务
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from ...utils.logger import setup_logger

logger = setup_logger('label_studio_service')

class LabelStudioService:
    """
    Label Studio服务类，用于与Label Studio API交互
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化Label Studio服务
        
        参数:
            config: 配置字典，包含API密钥和服务器URL
        """
        self.config = config or {}
        self.base_url = self.config.get('label_studio_url', 'http://localhost:8080/api')
        self.api_key = self.config.get('label_studio_api_key', '')
        
        if not self.api_key:
            logger.warning("未提供Label Studio API密钥，某些功能可能无法使用")
            
        self.headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        }
        
    def test_connection(self) -> bool:
        """
        测试与Label Studio服务器的连接
        
        返回:
            连接成功返回True，否则返回False
        """
        try:
            response = requests.get(f"{self.base_url}/projects", headers=self.headers)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"连接Label Studio失败: {e}")
            return False
            
    def get_projects(self) -> List[Dict]:
        """
        获取所有项目列表
        
        返回:
            项目列表
        """
        try:
            response = requests.get(f"{self.base_url}/projects", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取项目列表失败: {e}")
            return []
            
    def create_project(self, name: str, description: str = "", label_config: str = None) -> Optional[Dict]:
        """
        创建新项目
        
        参数:
            name: 项目名称
            description: 项目描述
            label_config: 标签配置XML
            
        返回:
            创建的项目信息
        """
        # 默认的标签配置，用于SBS序列标注
        if label_config is None:
            label_config = """
            <View>
              <Image name="image" value="$image" zoom="true" />
              <RectangleLabels name="sbs_points" toName="image">
                <Label value="point1" background="#FF0000" />
                <Label value="point2" background="#00FF00" />
                <Label value="point3" background="#0000FF" />
                <Label value="point4" background="#FFFF00" />
                <Label value="point5" background="#FF00FF" />
              </RectangleLabels>
              <Choices name="direction" toName="image">
                <Choice value="bullish" />
                <Choice value="bearish" />
              </Choices>
              <TextArea name="comments" toName="image" 
                placeholder="Enter any comments here..." 
                maxSubmissions="1" />
            </View>
            """
            
        data = {
            "title": name,
            "description": description,
            "label_config": label_config
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/projects", 
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            logger.info(f"成功创建项目: {name}")
            return response.json()
        except Exception as e:
            logger.error(f"创建项目失败: {e}")
            return None
            
    def get_project(self, project_id: int) -> Optional[Dict]:
        """
        获取项目详情
        
        参数:
            project_id: 项目ID
            
        返回:
            项目详情
        """
        try:
            response = requests.get(
                f"{self.base_url}/projects/{project_id}", 
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取项目详情失败 (ID: {project_id}): {e}")
            return None
            
    def delete_project(self, project_id: int) -> bool:
        """
        删除项目
        
        参数:
            project_id: 项目ID
            
        返回:
            成功返回True，否则返回False
        """
        try:
            response = requests.delete(
                f"{self.base_url}/projects/{project_id}", 
                headers=self.headers
            )
            response.raise_for_status()
            logger.info(f"成功删除项目 (ID: {project_id})")
            return True
        except Exception as e:
            logger.error(f"删除项目失败 (ID: {project_id}): {e}")
            return False
            
    def import_tasks(self, project_id: int, tasks: List[Dict]) -> bool:
        """
        导入任务到项目
        
        参数:
            project_id: 项目ID
            tasks: 任务列表
            
        返回:
            成功返回True，否则返回False
        """
        try:
            response = requests.post(
                f"{self.base_url}/projects/{project_id}/import", 
                headers=self.headers,
                json=tasks
            )
            response.raise_for_status()
            logger.info(f"成功导入{len(tasks)}个任务到项目 (ID: {project_id})")
            return True
        except Exception as e:
            logger.error(f"导入任务失败 (项目ID: {project_id}): {e}")
            return False
            
    def get_tasks(self, project_id: int) -> List[Dict]:
        """
        获取项目任务列表
        
        参数:
            project_id: 项目ID
            
        返回:
            任务列表
        """
        try:
            response = requests.get(
                f"{self.base_url}/projects/{project_id}/tasks", 
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取任务列表失败 (项目ID: {project_id}): {e}")
            return []
            
    def get_annotations(self, project_id: int, task_id: int = None) -> List[Dict]:
        """
        获取标注结果
        
        参数:
            project_id: 项目ID
            task_id: 任务ID，如果提供则获取特定任务的标注
            
        返回:
            标注结果列表
        """
        url = f"{self.base_url}/projects/{project_id}/annotations"
        if task_id:
            url = f"{self.base_url}/tasks/{task_id}/annotations"
            
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取标注结果失败: {e}")
            return []
            
    def export_annotations(self, project_id: int, export_format: str = 'JSON') -> Optional[Dict]:
        """
        导出项目标注结果
        
        参数:
            project_id: 项目ID
            export_format: 导出格式，默认为JSON
            
        返回:
            标注结果数据
        """
        try:
            response = requests.get(
                f"{self.base_url}/projects/{project_id}/export?exportType={export_format}", 
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"导出标注结果失败 (项目ID: {project_id}): {e}")
            return None
            
    def parse_sbs_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """
        解析SBS序列标注结果
        
        参数:
            annotations: 标注结果列表
            
        返回:
            解析后的SBS序列数据
        """
        parsed_results = []
        
        for item in annotations:
            if 'annotations' not in item or not item['annotations']:
                continue
                
            task_id = item.get('id')
            image_url = None
            if 'data' in item and 'image' in item['data']:
                image_url = item['data']['image']
                
            annotation = item['annotations'][0]  # 使用第一个标注
            result = annotation.get('result', [])
            
            sbs_points = {}
            direction = None
            comments = None
            
            for r in result:
                if r.get('type') == 'rectanglelabels':
                    label = r.get('value', {}).get('rectanglelabels', [])[0]
                    if label in ['point1', 'point2', 'point3', 'point4', 'point5']:
                        # 保存点位的坐标
                        x = r.get('value', {}).get('x')
                        y = r.get('value', {}).get('y')
                        width = r.get('value', {}).get('width')
                        height = r.get('value', {}).get('height')
                        
                        sbs_points[label] = {
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height
                        }
                elif r.get('type') == 'choices':
                    direction = r.get('value', {}).get('choices', [])[0]
                elif r.get('type') == 'textarea':
                    comments = r.get('value', {}).get('text', [])
            
            parsed_results.append({
                'task_id': task_id,
                'image_url': image_url,
                'sbs_points': sbs_points,
                'direction': direction,
                'comments': comments
            })
            
        return parsed_results
        
    def select_uncertain_samples(self, model_predictions: List[Dict], count: int = 10) -> List[int]:
        """
        从模型预测中选择最不确定的样本进行人工标注
        
        参数:
            model_predictions: 模型预测结果
            count: 选择的样本数量
            
        返回:
            选择的样本ID列表
        """
        # 按照置信度排序
        sorted_predictions = sorted(
            model_predictions, 
            key=lambda x: x.get('confidence', 1.0)
        )
        
        # 返回置信度最低的样本ID
        return [p.get('id') for p in sorted_predictions[:count]] 