import numpy as np
from PIL import Image
import os

# 创建目录
os.makedirs('temp', exist_ok=True)

# 创建空白图像 (800x600)
img = np.ones((800, 600, 3), dtype=np.uint8) * 255

# 添加网格线
for i in range(0, 800, 50):
    img[i:i+2, :] = [0, 0, 0]
    
for i in range(0, 600, 50):
    img[:, i:i+2] = [0, 0, 0]

# 添加一些"蜡烛图"模拟K线
for i in range(100, 550, 40):  # 限制在宽度范围内
    # K线的高低点
    high = np.random.randint(100, 500)
    low = np.random.randint(100, high)
    
    # 开盘和收盘价
    open_price = np.random.randint(low, high)
    close_price = np.random.randint(low, high)
    
    # 画蜡烛线
    img[low:high, i:i+10] = [0, 0, 0]  # 影线
    
    if close_price > open_price:
        # 阳线 - 绿色
        img[open_price:close_price, i-5:i+15] = [100, 200, 100]
    else:
        # 阴线 - 红色
        img[close_price:open_price, i-5:i+15] = [200, 100, 100]

# 添加简化的SMA线（灰色）
for i in range(100, 550):
    y_pos = 300 + 50 * np.sin(i/50)  # 简单的波浪线
    y_pos = int(max(100, min(700, y_pos)))  # 确保在图像范围内
    img[y_pos-2:y_pos+2, i-2:i+2] = [180, 180, 180]

# 保存图像
Image.fromarray(img).save('temp/test_chart.png')
print("已创建测试图表: temp/test_chart.png") 