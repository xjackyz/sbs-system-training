import pandas as pd
import os
from datetime import datetime, timedelta

def extract_last_week_data():
    """从12月数据中提取最后一周的数据作为验证集"""
    # 读取12月数据
    df = pd.read_csv('data/NQ1!_202412_1m.csv')
    
    # 确保时间列格式正确
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 获取最后一周的数据
    last_day = df['datetime'].max()
    start_date = last_day - timedelta(days=7)
    
    # 筛选最后一周的数据
    last_week_data = df[df['datetime'] >= start_date].copy()
    
    # 确保输出目录存在
    os.makedirs('data/validation', exist_ok=True)
    
    # 保存验证集
    output_path = 'data/validation/last_week_2024.csv'
    last_week_data.to_csv(output_path, index=False)
    print(f"验证集已保存至: {output_path}")
    print(f"数据范围: {last_week_data['datetime'].min()} 到 {last_week_data['datetime'].max()}")
    print(f"总行数: {len(last_week_data)}")

if __name__ == "__main__":
    extract_last_week_data() 