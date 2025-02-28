# SBS系统开发指南

本文档为希望参与SBS交易分析系统开发的开发者提供指导。

## 开发环境设置

### 推荐开发环境

- Python 3.10+
- 代码编辑器：VS Code、PyCharm或Vim
- Git客户端

### 开发环境设置步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/sbs_system.git
cd sbs_system
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows
```

3. 安装开发依赖
```bash
pip install -r requirements-dev.txt
```

4. 设置pre-commit钩子
```bash
pre-commit install
```

## 项目结构

主要目录和文件的作用：

- `src/`: 主要源代码
  - `utils/`: 工具函数
  - `image/`: 图像处理相关代码
  - `analysis/`: 分析模块
  - `bot/`: 机器人相关代码
  - `notification/`: 通知系统
  - `self_supervised/`: 自监督学习相关代码
- `app/`: Web应用相关代码
- `config/`: 配置文件
- `tests/`: 测试代码
- `scripts/`: 实用脚本
- `docs/`: 文档

## 开发流程

### 分支策略

我们采用基于功能分支的开发方式：

1. `main`: 主分支，包含稳定版本代码
2. `develop`: 开发分支，包含最新开发代码
3. 功能分支：从`develop`分支创建，命名格式为`feature/feature-name`
4. 修复分支：从`main`分支创建，命名格式为`hotfix/issue-name`

### 开发新功能

1. 创建新分支
```bash
git checkout develop
git pull
git checkout -b feature/your-feature-name
```

2. 开发功能

3. 运行测试
```bash
pytest
```

4. 提交代码
```bash
git add .
git commit -m "feat: add new feature"
```

5. 推送分支
```bash
git push origin feature/your-feature-name
```

6. 创建Pull Request到`develop`分支

### 代码风格

我们使用以下代码风格规范：

- Python: PEP 8
- 使用Black格式化代码
- 使用isort排序导入
- 使用flake8检查代码质量

可以运行以下命令格式化代码：
```bash
black .
isort .
flake8
```

## 测试

### 测试框架

我们使用pytest进行单元测试和集成测试。

### 运行测试

运行所有测试：
```bash
pytest
```

运行特定测试：
```bash
pytest tests/unit/test_specific_module.py
```

生成覆盖率报告：
```bash
pytest --cov=src tests/
```

### 测试规范

1. 所有功能必须包含测试
2. 单元测试应覆盖边缘情况
3. 集成测试应验证模块间交互

## 文档

### 文档规范

- 所有公共API必须有文档字符串
- 复杂函数应说明参数和返回值
- 使用Google风格的文档字符串

示例：
```python
def analyze_image(image_path: str, mode: str = "full") -> dict:
    """分析交易图表图像并返回结果。
    
    Args:
        image_path: 图表图像的路径
        mode: 分析模式，可选值为"full"或"quick"
        
    Returns:
        包含分析结果的字典
        
    Raises:
        ValueError: 当图像无法加载或识别时
    """
    # 函数实现
```

### 生成文档

我们使用Sphinx生成文档：
```bash
cd docs
make html
```

## 贡献指南

### 提交Pull Request

1. 确保代码符合风格规范
2. 确保所有测试通过
3. 更新相关文档
4. 创建清晰的PR描述，包括以下内容：
   - 实现的功能或修复的问题
   - 实现方式
   - 测试方法
   - 任何注意事项

### 代码评审流程

1. 至少需要一位维护者批准
2. 所有自动化测试必须通过
3. 代码必须符合风格规范
4. 文档必须更新

## 版本发布

### 版本号规范

我们使用语义版本控制(Semantic Versioning)：
- 主版本号：不兼容的API变更
- 次版本号：向后兼容的功能新增
- 修订号：向后兼容的问题修复

### 发布流程

1. 更新版本号和CHANGELOG.md
2. 从`develop`分支创建发布分支
3. 执行最终测试
4. 合并到`main`分支
5. 创建版本标签
6. 发布到PyPI（如适用）

## 持续集成/持续部署

我们使用GitHub Actions进行CI/CD：

- 每次提交会运行lint和测试
- 合并到`develop`分支会部署到测试环境
- 合并到`main`分支会部署到生产环境

## 联系与支持

如有任何问题，请通过以下方式联系：

- 创建GitHub Issue
- 发送邮件到项目维护者 