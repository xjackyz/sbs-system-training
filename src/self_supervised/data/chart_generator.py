import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any

class ChartGenerator:
    """
    图表生成器类，用于从数据生成图表。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化图表生成器。
        
        Args:
            config: 包含图表生成配置的字典。
        """
        self.output_dir = config['output']['dir']
        self.chart_style = config['chart']
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_chart(self, df: pd.DataFrame, output_path: str, title: str, indicators: List[str] = None):
        """
        生成图表并保存到指定路径。
        
        Args:
            df: 包含图表数据的 DataFrame。
            output_path: 图表保存路径。
            title: 图表标题。
            indicators: 要添加的技术指标。
        """
        plt.figure(figsize=(self.chart_style['width'], self.chart_style['height']))
        plt.title(title)
        plt.plot(df['date'], df['price'], label='价格')
        
        # 添加技术指标
        if indicators:
            for indicator in indicators:
                if indicator == 'sma20':
                    df['sma20'] = df['price'].rolling(window=20).mean()
                    plt.plot(df['date'], df['sma20'], label='SMA 20')
                elif indicator == 'sma200':
                    df['sma200'] = df['price'].rolling(window=200).mean()
                    plt.plot(df['date'], df['sma200'], label='SMA 200')

        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        return output_path

    def generate_sequence_charts(self, csv_path: str, window_size: int, stride: int, output_dir: str, indicators: List[str]):
        """
        从 CSV 文件生成多个图表。
        
        Args:
            csv_path: CSV 文件路径。
            window_size: 每个图表的窗口大小。
            stride: 滑动窗口步长。
            output_dir: 输出目录。
            indicators: 要添加的技术指标。
        """
        df = pd.read_csv(csv_path)
        chart_paths = []
        for start in range(0, len(df) - window_size + 1, stride):
            end = start + window_size
            window_df = df.iloc[start:end]
            output_path = os.path.join(output_dir, f'chart_{start}.png')
            chart_path = self.generate_chart(window_df, output_path, f'图表 {start}')
            chart_paths.append(chart_path)
        return chart_paths 