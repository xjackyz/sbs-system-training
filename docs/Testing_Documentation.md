# 测试文档

## 运行测试
要运行项目的测试，请确保您已安装所有依赖项，并在项目根目录下执行以下命令：
```bash
pytest tests/ --maxfail=1 --disable-warnings -q
```

## 测试框架
本项目使用 [pytest](https://docs.pytest.org/en/stable/) 作为测试框架。pytest 提供了简单易用的测试功能，支持测试用例的组织和执行。

### 测试覆盖率
要检查测试覆盖率，请安装 `pytest-cov` 插件，并使用以下命令运行测试：
```bash
pytest --cov=src tests/
```
这将生成测试覆盖率报告，显示哪些代码行被测试覆盖。

## 编写测试
在 `tests/` 目录中，您可以找到所有测试用例。每个测试文件应以 `test_` 开头，并包含以 `test_` 开头的测试函数。

### 示例测试用例
```python
import pytest
from src.self_supervised.trainer.self_supervised_trainer import DataCollector

def test_data_collector_initialization():
    config = CollectorConfig(...)
    collector = DataCollector(config)
    assert collector is not None
``` 