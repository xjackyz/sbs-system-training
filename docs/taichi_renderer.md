# TaichiKLineRenderer 类实现步骤

## 1. 初始化
- **`__init__` 方法**：接收 `width` 和 `height` 参数，初始化渲染缓冲区和基本设置。

## 2. 缓冲区操作
- **`clear_buffers` 方法**：清空渲染缓冲区。
- **`draw_line` 方法**：在缓冲区中绘制直线。

## 3. K线绘制
- **`draw_candlestick` 方法**：绘制单个K线，包括开盘、收盘、最高和最低价格。
- **`render_klines` 方法**：渲染整个K线图，返回渲染后的图像数组。

## 4. 网格和指标
- **`draw_grid` 方法**：绘制价格网格。
- **`add_technical_indicators` 方法**：添加技术指标，如移动平均线等。
- **`_draw_indicator_line` 方法**：绘制指标线。

## 5. 图像保存
- **`save_image` 方法**：将渲染结果保存为图片文件。 