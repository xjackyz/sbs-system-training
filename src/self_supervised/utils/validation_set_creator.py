import os
import json
import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ValidationSetCreator:
    """验证集创建器
    
    用于根据日期范围或特定规则创建自监督学习的验证集
    """
    
    def __init__(self, base_data_path: str, output_dir: str = "data/validation"):
        """
        初始化验证集创建器
        
        Args:
            base_data_path: 原始数据文件路径
            output_dir: 验证集输出目录
        """
        self.base_data_path = base_data_path
        self.output_dir = output_dir
        self.logger = logging.getLogger("validation_set_creator")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def create_date_range_validation(self, start_date: str, end_date: str, 
                                     validation_name: str = None) -> Dict:
        """
        根据日期范围创建验证集
        
        Args:
            start_date: 开始日期 (格式: 'YYYY-MM-DD')
            end_date: 结束日期 (格式: 'YYYY-MM-DD')
            validation_name: 验证集名称 (不指定则使用日期范围)
            
        Returns:
            验证集信息
        """
        # 设置验证集名称
        if validation_name is None:
            validation_name = f"validation_{start_date}_to_{end_date}"
        
        self.logger.info(f"开始创建日期范围验证集 {validation_name}: {start_date} 到 {end_date}")
        
        # 加载原始数据
        df = self._load_data(self.base_data_path)
        
        # 确保时间列被正确解析
        if 'date' in df.columns:
            time_col = 'date'
        elif 'timestamp' in df.columns:
            time_col = 'timestamp'
        else:
            # 尝试查找日期时间列
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                time_col = date_cols[0]
            else:
                raise ValueError("数据中没有找到日期时间列")
        
        # 转换为日期时间格式
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # 筛选指定日期范围的数据
        mask = (df[time_col] >= start_date) & (df[time_col] <= end_date)
        validation_df = df[mask].copy()
        
        # 检查是否有足够的数据
        if len(validation_df) == 0:
            self.logger.warning(f"在指定日期范围 {start_date} 到 {end_date} 内没有找到数据")
            return {"status": "error", "message": "指定日期范围内没有数据"}
        
        # 创建验证集目录
        validation_dir = os.path.join(self.output_dir, validation_name)
        os.makedirs(validation_dir, exist_ok=True)
        
        # 保存验证集数据
        validation_path = os.path.join(validation_dir, "validation_data.csv")
        validation_df.to_csv(validation_path, index=False)
        
        # 创建验证集描述
        validation_info = {
            "name": validation_name,
            "start_date": start_date,
            "end_date": end_date,
            "data_points": len(validation_df),
            "data_columns": list(validation_df.columns),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": validation_path
        }
        
        # 保存验证集描述
        info_path = os.path.join(validation_dir, "validation_info.json")
        with open(info_path, 'w') as f:
            json.dump(validation_info, f, indent=4)
        
        # 生成验证集统计信息
        self._generate_validation_stats(validation_df, time_col, validation_dir)
        
        self.logger.info(f"验证集 {validation_name} 创建完成，包含 {len(validation_df)} 个数据点")
        
        return validation_info
    
    def create_recent_period_validation(self, period_days: int = 7, 
                                        end_date: str = None) -> Dict:
        """
        创建最近一段时间的验证集
        
        Args:
            period_days: 验证集时间长度（天）
            end_date: 结束日期，默认为数据集最后一天
            
        Returns:
            验证集信息
        """
        # 加载原始数据
        df = self._load_data(self.base_data_path)
        
        # 确保时间列被正确解析
        if 'date' in df.columns:
            time_col = 'date'
        elif 'timestamp' in df.columns:
            time_col = 'timestamp'
        else:
            # 尝试查找日期时间列
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                time_col = date_cols[0]
            else:
                raise ValueError("数据中没有找到日期时间列")
        
        # 转换为日期时间格式
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # 确定结束日期
        if end_date is None:
            end_date = df[time_col].max()
        else:
            end_date = pd.to_datetime(end_date)
        
        # 计算开始日期
        start_date = end_date - timedelta(days=period_days)
        
        # 创建验证集名称
        validation_name = f"recent_{period_days}days_{end_date.strftime('%Y%m%d')}"
        
        # 调用日期范围验证集创建方法
        return self.create_date_range_validation(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            validation_name=validation_name
        )
    
    def create_last_week_of_year_validation(self, year: int = 2024) -> Dict:
        """
        创建指定年份最后一周的验证集
        
        Args:
            year: 年份
            
        Returns:
            验证集信息
        """
        # 计算该年最后一天
        end_date = datetime(year, 12, 31)
        
        # 计算最后一周的开始日期 (最后7天)
        start_date = end_date - timedelta(days=6)
        
        # 创建验证集名称
        validation_name = f"last_week_of_{year}"
        
        # 调用日期范围验证集创建方法
        return self.create_date_range_validation(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            validation_name=validation_name
        )
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            数据DataFrame
        """
        file_ext = os.path.splitext(data_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(data_path)
        elif file_ext in ['.parquet', '.pq']:
            return pd.read_parquet(data_path)
        elif file_ext in ['.h5', '.hdf5']:
            return pd.read_hdf(data_path)
        else:
            raise ValueError(f"不支持的数据文件格式: {file_ext}")
    
    def _generate_validation_stats(self, df: pd.DataFrame, time_col: str, output_dir: str):
        """
        生成验证集统计信息
        
        Args:
            df: 验证集数据
            time_col: 时间列名
            output_dir: 输出目录
        """
        # 创建时间分布图
        plt.figure(figsize=(12, 6))
        df[time_col].dt.date.value_counts().sort_index().plot(kind='bar')
        plt.title('验证集数据时间分布')
        plt.xlabel('日期')
        plt.ylabel('数据点数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_distribution.png'))
        plt.close()
        
        # 如果有价格列，创建价格走势图
        price_cols = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower()]
        if price_cols:
            plt.figure(figsize=(12, 6))
            df.set_index(time_col)[price_cols[0]].plot()
            plt.title(f'验证集 {price_cols[0]} 走势')
            plt.xlabel('时间')
            plt.ylabel(price_cols[0])
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'price_trend.png'))
            plt.close()
        
        # 创建基本统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[numeric_cols].describe()
        
        # 保存统计信息
        stats.to_csv(os.path.join(output_dir, 'statistics.csv'))
        
        # 创建统计信息可视化
        fig, axes = plt.subplots(len(numeric_cols[:5]), 1, figsize=(10, 3*len(numeric_cols[:5])))
        for i, col in enumerate(numeric_cols[:5]):
            if len(numeric_cols) > 1:
                ax = axes[i]
            else:
                ax = axes
            df[col].hist(bins=50, ax=ax)
            ax.set_title(f'{col} 分布')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'numeric_distributions.png'))
        plt.close()
    
    def list_available_validations(self) -> List[Dict]:
        """
        列出所有可用的验证集
        
        Returns:
            验证集信息列表
        """
        validations = []
        
        # 检查验证集目录是否存在
        if not os.path.exists(self.output_dir):
            return validations
        
        # 遍历验证集目录
        for val_dir in os.listdir(self.output_dir):
            info_path = os.path.join(self.output_dir, val_dir, "validation_info.json")
            
            # 检查验证集信息文件是否存在
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    validation_info = json.load(f)
                    validations.append(validation_info)
        
        return validations
    
    def get_validation_split(self, validation_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取训练和验证数据分割
        
        根据验证集名称获取分割后的训练集和验证集
        
        Args:
            validation_name: 验证集名称
            
        Returns:
            (训练数据, 验证数据)
        """
        # 加载原始数据
        df = self._load_data(self.base_data_path)
        
        # 加载验证集信息
        info_path = os.path.join(self.output_dir, validation_name, "validation_info.json")
        if not os.path.exists(info_path):
            raise ValueError(f"验证集 {validation_name} 不存在")
            
        with open(info_path, 'r') as f:
            validation_info = json.load(f)
        
        start_date = validation_info["start_date"]
        end_date = validation_info["end_date"]
        
        # 确保时间列被正确解析
        if 'date' in df.columns:
            time_col = 'date'
        elif 'timestamp' in df.columns:
            time_col = 'timestamp'
        else:
            # 尝试查找日期时间列
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                time_col = date_cols[0]
            else:
                raise ValueError("数据中没有找到日期时间列")
        
        # 转换为日期时间格式
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # 分割数据
        validation_mask = (df[time_col] >= start_date) & (df[time_col] <= end_date)
        validation_df = df[validation_mask].copy()
        train_df = df[~validation_mask].copy()
        
        return train_df, validation_df

    def _create_annotated_images(self, seq_dir: Path, seq_info: Dict[str, Any]):
        """为序列创建标注版本的图像
        
        Args:
            seq_dir: 序列目录
            seq_info: 序列信息
        """
        # 获取序列中的所有图像
        image_files = sorted(
            [f for f in seq_dir.glob('*.png') if 'annotated' not in f.name]
        )
        
        if not image_files:
            logger.warning(f"序列目录中没有图像文件: {seq_dir}")
            return
        
        # 获取序列标签
        sequence_label = seq_info.get('label', 'neutral')
        signal_result = seq_info.get('signal_result', 'unknown')
        confidence = seq_info.get('confidence', 0.0)
        price_change = seq_info.get('price_change', 0.0)
        
        # 为每个图像创建标注版本
        for idx, img_file in enumerate(image_files):
            try:
                # 打开图像
                image = Image.open(img_file)
                draw = ImageDraw.Draw(image)
                
                # 尝试加载字体
                try:
                    font = ImageFont.truetype("simhei.ttf", 20)
                except:
                    # 如果找不到中文字体，使用默认字体
                    font = ImageFont.load_default()
                
                # 添加序列信息
                info_text = [
                    f"序列ID: {seq_dir.name}",
                    f"图像索引: {idx+1}/{len(image_files)}",
                    f"标签: {self.label_mapping.get(sequence_label, sequence_label)}",
                    f"置信度: {confidence:.2f}",
                    f"价格变化: {price_change:.2f}%",
                    f"结果: {'成功' if signal_result == 'success' else '失败' if signal_result == 'failure' else '未知'}"
                ]
                
                # 绘制半透明背景
                draw.rectangle(
                    [(10, 10), (280, 150)], 
                    fill=(0, 0, 0, 128)
                )
                
                # 绘制文本
                y_text = 20
                for text in info_text:
                    draw.text((20, y_text), text, fill=(255, 255, 255), font=font)
                    y_text += 25
                
                # 保存标注版本
                annotated_path = seq_dir / f"{img_file.stem}_annotated{img_file.suffix}"
                image.save(annotated_path)
                
            except Exception as e:
                logger.error(f"为图像 {img_file} 创建标注版本时出错: {e}")
    
    def _update_validation_info(self, seq_id: str, seq_info: Dict[str, Any]):
        """更新验证集信息
        
        Args:
            seq_id: 序列ID
            seq_info: 序列信息
        """
        # 获取序列标签
        sequence_label = seq_info.get('label', 'neutral')
        
        # 更新统计信息
        self.validation_info['total_sequences'] += 1
        self.validation_info[f'{sequence_label}_sequences'] += 1
        
        # 添加序列信息
        self.validation_info['sequences'].append({
            'id': seq_id,
            'label': sequence_label,
            'confidence': seq_info.get('confidence', 0.0),
            'signal_result': seq_info.get('signal_result', 'unknown'),
            'price_change': seq_info.get('price_change', 0.0),
            'created_at': seq_info.get('created_at', '')
        })
    
    def generate_validation_summary(self) -> Dict[str, Any]:
        """生成验证集摘要
        
        Returns:
            验证集摘要信息
        """
        summary = {
            'total_sequences': self.validation_info['total_sequences'],
            'distribution': {
                'bullish': self.validation_info['bullish_sequences'],
                'bearish': self.validation_info['bearish_sequences'],
                'neutral': self.validation_info['neutral_sequences']
            },
            'success_rate': 0.0,
            'avg_confidence': 0.0
        }
        
        # 计算成功率
        sequences = self.validation_info.get('sequences', [])
        if sequences:
            success_count = sum(1 for seq in sequences if seq.get('signal_result') == 'success')
            summary['success_rate'] = success_count / len(sequences)
            
            # 计算平均置信度
            avg_confidence = sum(seq.get('confidence', 0.0) for seq in sequences) / len(sequences)
            summary['avg_confidence'] = avg_confidence
        
        return summary
    
    def create_distribution_chart(self, save_path: Optional[str] = None) -> str:
        """创建验证集分布图表
        
        Args:
            save_path: 保存路径，如果为None，则使用默认路径
            
        Returns:
            图表保存路径
        """
        if save_path is None:
            save_path = str(self.validation_dir / 'validation_distribution.png')
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 准备数据
        labels = ['看涨', '看跌', '中性']
        sizes = [
            self.validation_info['bullish_sequences'],
            self.validation_info['bearish_sequences'],
            self.validation_info['neutral_sequences']
        ]
        colors = ['green', 'red', 'gray']
        
        # 绘制饼图
        plt.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%', 
            startangle=90,
            shadow=True
        )
        plt.axis('equal')  # 保持饼图为圆形
        plt.title('验证集标签分布')
        
        # 保存图表
        plt.savefig(save_path)
        plt.close()
        
        return save_path 