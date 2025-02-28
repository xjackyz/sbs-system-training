# SBS系统API参考

本文档提供SBS交易分析系统的API接口说明。

## API概述

SBS系统提供REST API接口，允许您以编程方式访问系统功能。API使用JSON格式进行数据交换，并使用标准HTTP方法和状态码。

基本URL: `http://localhost:5000/api/v1`

## 认证

API使用令牌认证机制。在每个请求的头部包含`Authorization`字段：

```
Authorization: Bearer YOUR_API_TOKEN
```

您可以在系统配置文件中设置或生成API令牌。

## 端点参考

### 图表分析

#### 分析图表图像

```
POST /analyze/chart
```

**请求参数**

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| image | File | 是 | 要分析的图表图像文件 |
| mode | String | 否 | 分析模式: "full", "quick" (默认: "full") |
| require_signals | Boolean | 否 | 是否必须返回信号 (默认: true) |

**请求示例**

```bash
curl -X POST "http://localhost:5000/api/v1/analyze/chart" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -F "image=@/path/to/chart.png" \
  -F "mode=full" \
  -F "require_signals=true"
```

**响应示例**

```json
{
  "status": "success",
  "analysis": {
    "trend": "上升",
    "support_levels": [
      {
        "price": 42500,
        "strength": "强"
      },
      {
        "price": 41200,
        "strength": "中"
      }
    ],
    "resistance_levels": [
      {
        "price": 44300,
        "strength": "强"
      }
    ],
    "signals": [
      {
        "type": "BUY",
        "confidence": 0.85,
        "description": "价格突破上升三角形形态，伴随成交量增加",
        "entry": 43200,
        "stop_loss": 41500,
        "take_profit": 46000
      }
    ],
    "indicators": {
      "macd": "bullish",
      "rsi": 63,
      "ma_alignment": "bullish"
    }
  },
  "processing_time": 2.34,
  "request_id": "f8e7d6c5-b4a3-42f1-9e8d-7c6b5a4f3d2e"
}
```

### 信号管理

#### 获取历史信号

```
GET /signals
```

**请求参数**

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| from_date | String | 否 | 开始日期 (YYYY-MM-DD) |
| to_date | String | 否 | 结束日期 (YYYY-MM-DD) |
| limit | Integer | 否 | 返回结果数量限制 (默认: 50) |
| offset | Integer | 否 | 分页偏移量 (默认: 0) |
| signal_type | String | 否 | 信号类型过滤 ("BUY", "SELL", "HOLD") |

**请求示例**

```bash
curl -X GET "http://localhost:5000/api/v1/signals?from_date=2023-01-01&limit=10&signal_type=BUY" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**响应示例**

```json
{
  "status": "success",
  "signals": [
    {
      "id": "abc123",
      "timestamp": "2023-01-15T08:30:45Z",
      "type": "BUY",
      "confidence": 0.92,
      "description": "价格突破上升通道上轨",
      "entry": 43500,
      "stop_loss": 42100,
      "take_profit": 46000,
      "chart_id": "chart_20230115_btc"
    },
    // 更多信号...
  ],
  "total": 45,
  "limit": 10,
  "offset": 0
}
```

### 系统管理

#### 获取系统状态

```
GET /system/status
```

**请求示例**

```bash
curl -X GET "http://localhost:5000/api/v1/system/status" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**响应示例**

```json
{
  "status": "success",
  "system_status": {
    "version": "1.0.0",
    "uptime": "5d 12h 34m",
    "cpu_usage": 23.5,
    "memory_usage": 4.2,
    "gpu_usage": 45.6,
    "active_tasks": 2,
    "queue_size": 0,
    "last_error": null
  },
  "components_status": {
    "image_processor": "healthy",
    "llava_model": "healthy",
    "signal_generator": "healthy",
    "notification_system": "healthy",
    "database": "healthy"
  }
}
```

## 错误处理

API使用标准HTTP状态码表示请求状态：

- 200: 成功
- 400: 无效请求
- 401: 未授权
- 403: 禁止访问
- 404: 资源不存在
- 500: 服务器错误

错误响应格式：

```json
{
  "status": "error",
  "error": {
    "code": "invalid_image",
    "message": "提供的图像无法识别为有效的交易图表"
  },
  "request_id": "a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6"
}
```

## 速率限制

API具有速率限制，以防止过度使用：

- 基础限制: 60次请求/小时
- 分析端点限制: 10次请求/小时

超过限制将返回429状态码。

## 进一步阅读

- [API使用示例](examples.md)
- [Webhook集成](webhooks.md)
- [API客户端库](client-libraries.md) 