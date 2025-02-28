# LabelStudioAdapter 类实现步骤

## 1. 初始化
- **`__init__` 方法**：接收 `config` 参数，初始化 Label Studio 适配器。

## 2. 项目管理
- **`create_project` 方法**：创建新的标注项目，返回项目ID。

## 3. 数据导入
- **`import_tasks` 方法**：导入任务和预测结果。
- **`import_uncertain_tasks` 方法**：导入不确定性样本。

## 4. 数据导出
- **`export_annotations` 方法**：导出标注结果。
- **`_format_predictions` 方法**：格式化预测结果。
- **`_extract_points` 方法**：提取标注点位信息。

## 5. 统计信息
- **`get_project_stats` 方法**：获取项目统计信息。 