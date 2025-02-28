"""
Label Studio适配器
用于集成Label Studio进行SBS序列标注
"""

import os
import json
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from ..utils.logger import setup_logger

logger = setup_logger('label_studio_adapter')

class LabelStudioAdapter:
    """Label Studio适配器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Label Studio适配器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.api_url = config.get('label_studio_url', 'http://localhost:8080')
        self.api_key = config.get('label_studio_api_key')
        
        if not self.api_key:
            raise ValueError("Label Studio API密钥未设置")
            
        # 设置API请求头
        self.headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # 创建输出目录
        self.output_dir = Path(config.get('output_dir', 'data/labeled'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_project(self, name: str, description: str) -> int:
        """
        创建标注项目
        
        Args:
            name: 项目名称
            description: 项目描述
            
        Returns:
            项目ID
        """
        # 项目标注配置
        label_config = '''
        <View>
          <Header value="SBS序列标注"/>
          <Image name="kline" value="$image"/>
          <Labels name="points" toName="kline">
            <Label value="point1" background="#F7B32B" title="第一次回调"/>
            <Label value="point2" background="#00BCD4" title="极值点"/>
            <Label value="point3" background="#E91E63" title="流动性获取"/>
            <Label value="point4" background="#FF9800" title="确认点"/>
            <Label value="point5" background="#4CAF50" title="目标点"/>
          </Labels>
          <TextArea name="notes" toName="kline" 
                    placeholder="添加标注说明..." 
                    maxSubmissions="1"/>
          <Choices name="confidence" toName="kline" required="true">
            <Choice value="high" title="高置信度"/>
            <Choice value="medium" title="中等置信度"/>
            <Choice value="low" title="低置信度"/>
          </Choices>
        </View>
        '''
        
        try:
            # 创建项目
            response = requests.post(
                f'{self.api_url}/api/projects',
                headers=self.headers,
                json={
                    'title': name,
                    'description': description,
                    'label_config': label_config
                }
            )
            response.raise_for_status()
            
            project_id = response.json()['id']
            logger.info(f"已创建项目: {name} (ID: {project_id})")
            return project_id
            
        except requests.exceptions.RequestException as e:
            logger.error(f"创建项目失败: {str(e)}")
            raise
            
    def import_tasks(self, 
                    project_id: int,
                    predictions: List[Dict],
                    kline_images: Dict[str, str]) -> None:
        """
        导入标注任务
        
        Args:
            project_id: 项目ID
            predictions: 模型预测结果列表
            kline_images: K线图像路径字典
        """
        try:
            tasks = []
            for pred_id, prediction in enumerate(predictions):
                if str(pred_id) not in kline_images:
                    continue
                    
                # 构建任务数据
                task = {
                    'data': {
                        'image': kline_images[str(pred_id)],
                        'predictions': prediction['points'],
                        'confidence_scores': prediction['confidence_scores']
                    },
                    'predictions': [{
                        'model_version': 'v1',
                        'result': self._format_predictions(prediction)
                    }]
                }
                tasks.append(task)
                
            # 批量导入任务
            response = requests.post(
                f'{self.api_url}/api/projects/{project_id}/import',
                headers=self.headers,
                json={
                    'tasks': tasks
                }
            )
            response.raise_for_status()
            
            logger.info(f"已导入 {len(tasks)} 个标注任务")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"导入任务失败: {str(e)}")
            raise
            
    def import_uncertain_tasks(self, project_id: int, uncertain_samples: List[Dict], kline_images: Dict[str, str]) -> None:
        """
        导入不确定性样本任务
        
        Args:
            project_id: 项目ID
            uncertain_samples: 不确定性样本列表
            kline_images: K线图像路径字典
        """
        tasks = []
        for sample in uncertain_samples:
            task = {
                'data': {
                    'image': kline_images[sample['id']],
                    'predictions': sample['predictions'],
                },
                'predictions': [{
                    'model_version': 'v1',
                    'result': self._format_predictions(sample)
                }]
            }
            tasks.append(task)

        # 批量导入任务
        response = requests.post(
            f'{self.api_url}/api/projects/{project_id}/import',
            headers=self.headers,
            json={'tasks': tasks}
        )
        response.raise_for_status()
        logger.info(f"已导入 {len(tasks)} 个不确定性样本任务")
            
    def export_annotations(self, 
                         project_id: int,
                         output_path: Optional[str] = None) -> List[Dict]:
        """
        导出标注结果
        
        Args:
            project_id: 项目ID
            output_path: 输出文件路径
            
        Returns:
            标注结果列表
        """
        try:
            # 获取标注结果
            response = requests.get(
                f'{self.api_url}/api/projects/{project_id}/export',
                headers=self.headers
            )
            response.raise_for_status()
            
            annotations = response.json()
            
            # 处理标注结果
            processed_annotations = []
            for item in annotations:
                processed = {
                    'task_id': item['id'],
                    'points': self._extract_points(item['annotations'][0]),
                    'confidence': item['annotations'][0]['result'].get('confidence'),
                    'notes': item['annotations'][0]['result'].get('notes', ''),
                    'timestamp': datetime.now().isoformat()
                }
                processed_annotations.append(processed)
                
            # 保存结果
            if output_path:
                output_file = self.output_dir / output_path
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_annotations, f, ensure_ascii=False, indent=2)
                logger.info(f"标注结果已保存至: {output_file}")
                
            return processed_annotations
            
        except requests.exceptions.RequestException as e:
            logger.error(f"导出标注结果失败: {str(e)}")
            raise
            
    def _format_predictions(self, prediction: Dict) -> List[Dict]:
        """
        格式化预测结果为Label Studio格式
        
        Args:
            prediction: 预测结果
            
        Returns:
            Label Studio格式的预测结果
        """
        results = []
        for point_name, point_idx in prediction['points'].items():
            if point_idx is not None:
                results.append({
                    'type': 'labels',
                    'value': {
                        'labels': [point_name],
                        'start': point_idx,
                        'end': point_idx,
                        'score': prediction['confidence_scores'][point_name]
                    }
                })
        return results
        
    def _extract_points(self, annotation: Dict) -> Dict[str, int]:
        """
        从标注中提取点位信息
        
        Args:
            annotation: 标注数据
            
        Returns:
            点位字典
        """
        points = {}
        for result in annotation['result']:
            if result['type'] == 'labels':
                point_name = result['value']['labels'][0]
                points[point_name] = result['value']['start']
        return points
        
    def get_project_stats(self, project_id: int) -> Dict[str, Any]:
        """
        获取项目统计信息
        
        Args:
            project_id: 项目ID
            
        Returns:
            统计信息字典
        """
        try:
            response = requests.get(
                f'{self.api_url}/api/projects/{project_id}',
                headers=self.headers
            )
            response.raise_for_status()
            
            project = response.json()
            return {
                'total_tasks': project['task_number'],
                'completed_tasks': project['finished_task_number'],
                'total_predictions': project['total_predictions_number'],
                'total_annotations': project['total_annotations_number']
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"获取项目统计信息失败: {str(e)}")
            raise 