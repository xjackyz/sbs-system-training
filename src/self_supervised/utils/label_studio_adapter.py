"""
Label Studio适配器
用于连接Label Studio标注平台，处理数据导入和结果导出
"""

import os
import json
import time
import requests
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger('label_studio_adapter')

class LabelStudioAdapter:
    """Label Studio适配器类，用于SBS序列标注任务"""
    
    def __init__(self, config: Dict):
        """
        初始化Label Studio适配器
        
        Args:
            config: 配置信息
        """
        self.config = config
        self.api_url = config.get('api_url', 'http://localhost:8080/api')
        self.api_key = config.get('api_key', '')
        self.project_id = config.get('project_id')
        self.logger = setup_logger('label_studio_adapter')
        
        # 标签配置
        self.label_config = self._get_default_label_config()
        
        # API请求头
        self.headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        }
        
    def create_project(self, name: str = 'SBS序列标注', description: str = 'SBS交易序列标注任务') -> int:
        """
        创建标注项目
        
        Args:
            name: 项目名称
            description: 项目描述
            
        Returns:
            项目ID
        """
        try:
            # 准备项目数据
            project_data = {
                'title': name,
                'description': description,
                'label_config': self.label_config
            }
            
            # 发送创建请求
            response = requests.post(
                f'{self.api_url}/projects',
                headers=self.headers,
                json=project_data
            )
            
            # 检查响应
            if response.status_code == 201:
                project_info = response.json()
                self.project_id = project_info['id']
                logger.info(f"已创建项目: {name} (ID: {self.project_id})")
                return self.project_id
            else:
                logger.error(f"创建项目失败: {response.status_code} - {response.text}")
                raise Exception(f"创建项目失败: {response.text}")
                
        except Exception as e:
            logger.error(f"创建项目异常: {str(e)}")
            raise
            
    def import_tasks(self, 
                    tasks_data: List[Dict], 
                    include_predictions: bool = True,
                    replace_duplicates: bool = True) -> List[int]:
        """
        导入标注任务
        
        Args:
            tasks_data: 任务数据列表
            include_predictions: 是否包含预测结果
            replace_duplicates: 是否替换重复任务
            
        Returns:
            导入的任务ID列表
        """
        try:
            if not self.project_id:
                raise ValueError("未设置项目ID，请先创建或设置项目")
                
            # 准备任务数据
            tasks = []
            for task in tasks_data:
                task_entry = {
                    'data': task['data']
                }
                
                # 如果包含预测结果
                if include_predictions and 'predictions' in task:
                    task_entry['predictions'] = task['predictions']
                
                tasks.append(task_entry)
                
            # 设置请求参数
            params = {}
            if replace_duplicates:
                params['return_task_ids'] = 'true'
                
            # 发送导入请求
            response = requests.post(
                f'{self.api_url}/projects/{self.project_id}/import',
                headers=self.headers,
                json=tasks,
                params=params
            )
            
            # 检查响应
            if response.status_code in [201, 200]:
                result = response.json()
                imported_count = result.get('task_count', 0)
                task_ids = result.get('task_ids', [])
                logger.info(f"已导入 {imported_count} 个任务")
                return task_ids
            else:
                logger.error(f"导入任务失败: {response.status_code} - {response.text}")
                raise Exception(f"导入任务失败: {response.text}")
                
        except Exception as e:
            logger.error(f"导入任务异常: {str(e)}")
            raise
            
    def import_uncertain_tasks(self, 
                              klines_data: List[Dict], 
                              model_predictions: List[Dict],
                              confidence_thresholds: Dict[str, float] = None,
                              max_tasks: int = 100) -> List[int]:
        """
        导入不确定的任务
        
        Args:
            klines_data: K线数据列表
            model_predictions: 模型预测结果
            confidence_thresholds: 置信度阈值
            max_tasks: 最大任务数量
            
        Returns:
            导入的任务ID列表
        """
        try:
            # 默认置信度阈值
            default_thresholds = {
                'sequence_status': 0.7,
                'point1': 0.75,
                'point2': 0.8,
                'point3': 0.8, 
                'point4': 0.75,
                'point5': 0.7,
                'trade_direction': 0.8
            }
            
            # 使用用户提供的阈值或默认值
            thresholds = confidence_thresholds or default_thresholds
            
            # 筛选出不确定的样本
            uncertain_samples = []
            for i, (kline, prediction) in enumerate(zip(klines_data, model_predictions)):
                # 检查序列状态置信度
                if prediction.get('sequence_status', {}).get('confidence', 1.0) < thresholds['sequence_status']:
                    uncertain_samples.append((kline, prediction, i))
                    continue
                    
                # 只有当序列被识别为活跃时，才检查点位置信度
                if prediction.get('sequence_status', {}).get('is_active', False):
                    # 检查各点位置信度
                    points = prediction.get('points', {})
                    for point_name in ['point1', 'point2', 'point3', 'point4', 'point5']:
                        if point_name in points and points[point_name].get('confidence', 1.0) < thresholds.get(point_name, 0.75):
                            uncertain_samples.append((kline, prediction, i))
                            break
                            
                    # 检查交易方向置信度
                    if prediction.get('trade_setup', {}).get('confidence', 1.0) < thresholds.get('trade_direction', 0.8):
                        uncertain_samples.append((kline, prediction, i))
                
            # 限制任务数量
            uncertain_samples = uncertain_samples[:max_tasks]
            
            # 准备标注任务
            tasks_data = []
            for kline, prediction, idx in uncertain_samples:
                # 创建预标注数据
                pre_annotations = self._create_annotations_from_prediction(prediction)
                
                # 创建任务
                task = {
                    'data': {
                        'kline_data': json.dumps(kline),
                        'chart_id': f"chart_{idx}",
                        'timestamp': kline.get('timestamp', ''),
                        'symbol': kline.get('symbol', ''),
                        'timeframe': kline.get('timeframe', ''),
                    },
                    'predictions': [{
                        'model_version': prediction.get('model_version', 'v1.0'),
                        'score': prediction.get('sequence_status', {}).get('confidence', 0.5),
                        'result': pre_annotations
                    }]
                }
                tasks_data.append(task)
                
            # 导入任务
            return self.import_tasks(tasks_data, include_predictions=True)
            
        except Exception as e:
            logger.error(f"导入不确定任务异常: {str(e)}")
            raise
            
    def _create_annotations_from_prediction(self, prediction: Dict) -> List[Dict]:
        """
        从预测结果创建标注数据
        
        Args:
            prediction: 预测结果
            
        Returns:
            标注数据列表
        """
        annotations = []
        
        # 标注序列状态
        if 'sequence_status' in prediction:
            annotations.append({
                'from_name': 'sequence_status',
                'to_name': 'kline_chart',
                'type': 'choices',
                'value': {
                    'choices': [prediction['sequence_status']['label']]
                }
            })
        
        # 标注点位
        if 'points' in prediction:
            points = prediction['points']
            for point_name in ['point1', 'point2', 'point3', 'point4', 'point5']:
                if point_name in points and points[point_name] is not None:
                    point_index = points[point_name]
                    # 标注点位
                    annotations.append({
                        'from_name': 'sbs_points',
                        'to_name': 'kline_chart',
                        'type': 'rectanglelabels',
                        'value': {
                            'rectanglelabels': [point_name],
                            'x': point_index / 100.0,  # 归一化坐标
                            'y': 0.5,
                            'width': 0.01,
                            'height': 0.1
                        }
                    })
        
        # 标注交易方向
        if 'trade_setup' in prediction and 'direction' in prediction['trade_setup']:
            annotations.append({
                'from_name': 'trade_direction',
                'to_name': 'kline_chart',
                'type': 'choices',
                'value': {
                    'choices': [prediction['trade_setup']['direction']]
                }
            })
            
        return annotations
        
    def export_annotations(self, 
                          project_id: Optional[int] = None, 
                          task_ids: Optional[List[int]] = None) -> List[Dict]:
        """
        导出标注结果
        
        Args:
            project_id: 项目ID，如果为None则使用当前项目
            task_ids: 任务ID列表，如果为None则导出所有任务
            
        Returns:
            标注结果列表
        """
        try:
            project_id = project_id or self.project_id
            if not project_id:
                raise ValueError("未设置项目ID")
                
            # 构建请求参数
            params = {}
            if task_ids:
                params['ids'] = ','.join(map(str, task_ids))
                
            # 发送导出请求
            response = requests.get(
                f'{self.api_url}/projects/{project_id}/export',
                headers=self.headers,
                params=params
            )
            
            # 检查响应
            if response.status_code == 200:
                annotations = response.json()
                logger.info(f"已导出 {len(annotations)} 条标注记录")
                return annotations
            else:
                logger.error(f"导出标注失败: {response.status_code} - {response.text}")
                raise Exception(f"导出标注失败: {response.text}")
                
        except Exception as e:
            logger.error(f"导出标注异常: {str(e)}")
            raise
            
    def get_completed_tasks(self, project_id: Optional[int] = None) -> List[Dict]:
        """
        获取已完成的任务
        
        Args:
            project_id: 项目ID，如果为None则使用当前项目
            
        Returns:
            已完成任务列表
        """
        try:
            project_id = project_id or self.project_id
            if not project_id:
                raise ValueError("未设置项目ID")
                
            # 发送请求
            response = requests.get(
                f'{self.api_url}/projects/{project_id}/tasks',
                headers=self.headers,
                params={'filter': '{"completed":true}'}
            )
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                tasks = result.get('tasks', [])
                logger.info(f"已获取 {len(tasks)} 个已完成任务")
                return tasks
            else:
                logger.error(f"获取已完成任务失败: {response.status_code} - {response.text}")
                raise Exception(f"获取已完成任务失败: {response.text}")
                
        except Exception as e:
            logger.error(f"获取已完成任务异常: {str(e)}")
            raise
            
    def parse_annotations(self, annotations: List[Dict]) -> Dict[str, Dict]:
        """
        解析标注结果
        
        Args:
            annotations: 标注结果列表
            
        Returns:
            解析后的标注数据
        """
        try:
            parsed_data = {}
            
            for annotation in annotations:
                task_id = annotation.get('id')
                
                # 提取K线数据
                kline_data = annotation.get('data', {}).get('kline_data', '{}')
                kline = json.loads(kline_data) if isinstance(kline_data, str) else kline_data
                
                # 提取标注结果
                completions = annotation.get('annotations', [])
                if not completions:
                    continue
                    
                # 以第一个完成的标注为准
                completion = completions[0]
                result = completion.get('result', [])
                
                # 解析结果
                sequence_status = None
                points = {}
                trade_direction = None
                
                for item in result:
                    item_type = item.get('type')
                    from_name = item.get('from_name')
                    
                    if from_name == 'sequence_status' and item_type == 'choices':
                        choices = item.get('value', {}).get('choices', [])
                        if choices:
                            sequence_status = choices[0]
                            
                    elif from_name == 'sbs_points' and item_type == 'rectanglelabels':
                        rect_labels = item.get('value', {}).get('rectanglelabels', [])
                        if rect_labels:
                            point_name = rect_labels[0]
                            # 从相对位置转换为K线索引
                            x_position = item.get('value', {}).get('x', 0)
                            # 假设kline长度为100
                            kline_length = len(kline.get('open', [])) if isinstance(kline, dict) else 100
                            point_idx = int(x_position * kline_length)
                            points[point_name] = point_idx
                            
                    elif from_name == 'trade_direction' and item_type == 'choices':
                        choices = item.get('value', {}).get('choices', [])
                        if choices:
                            trade_direction = choices[0]
                            
                # 组织标注数据
                parsed_data[task_id] = {
                    'sample_id': task_id,
                    'sequence_status': {
                        'label': sequence_status,
                        'is_active': sequence_status != "未形成" if sequence_status else False
                    },
                    'points': points,
                    'trade_setup': {
                        'direction': trade_direction
                    },
                    'kline': kline
                }
                
            return parsed_data
            
        except Exception as e:
            logger.error(f"解析标注结果异常: {str(e)}")
            raise
            
    def _get_default_label_config(self) -> str:
        """
        获取默认标签配置
        
        Returns:
            XML格式的标签配置
        """
        return """
        <View>
            <Header value="SBS序列标注" />
            <TimeSeriesLabels name="kline_chart" valueType="json" value="$kline_data" timeColumn="time" format="yyyy-MM-dd'T'HH:mm:ss.SSS'Z'">
                <View style="display: flex; gap: 1em; justify-content: center; align-items: center;">
                    <Labels name="sbs_points" toName="kline_chart">
                        <Label value="point1" background="#FFA39E" />
                        <Label value="point2" background="#D4380D" />
                        <Label value="point3" background="#FFC069" />
                        <Label value="point4" background="#AD8B00" />
                        <Label value="point5" background="#D3F261" />
                    </Labels>
                </View>
                <View style="margin-top: 1em">
                    <Header value="序列状态" />
                    <Choices name="sequence_status" toName="kline_chart" showInLine="true">
                        <Choice value="未形成" />
                        <Choice value="形成中" />
                        <Choice value="已完成" />
                    </Choices>
                </View>
                <View style="margin-top: 1em">
                    <Header value="交易方向" />
                    <Choices name="trade_direction" toName="kline_chart" showInLine="true">
                        <Choice value="多" />
                        <Choice value="空" />
                        <Choice value="无信号" />
                    </Choices>
                </View>
            </TimeSeriesLabels>
            <View style="margin-top: 1em">
                <TextArea name="comments" toName="kline_chart" placeholder="添加备注..." rows="2" editable="true" maxSubmissions="1" />
            </View>
        </View>
        """
        
    def generate_preannotated_task(self, 
                                  kline: Dict, 
                                  prediction: Dict, 
                                  task_id: Optional[str] = None) -> Dict:
        """
        生成预标注任务
        
        Args:
            kline: K线数据
            prediction: 预测结果
            task_id: 任务ID（可选）
            
        Returns:
            预标注任务数据
        """
        # 生成任务ID
        task_id = task_id or f"task_{int(time.time() * 1000)}"
        
        # 创建预标注数据
        pre_annotations = self._create_annotations_from_prediction(prediction)
        
        # 创建任务
        task = {
            'data': {
                'kline_data': json.dumps(kline),
                'chart_id': task_id,
                'timestamp': kline.get('timestamp', ''),
                'symbol': kline.get('symbol', ''),
                'timeframe': kline.get('timeframe', ''),
            },
            'predictions': [{
                'model_version': prediction.get('model_version', 'v1.0'),
                'score': prediction.get('sequence_status', {}).get('confidence', 0.5),
                'result': pre_annotations
            }]
        }
        
        return task 