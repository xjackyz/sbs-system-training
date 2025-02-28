# 使用python:3.9-slim作为基础
FROM python:3.9-slim
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir "torch>=2.0.0,<2.2.0"
RUN pip install --no-cache-dir packaging
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "scripts/train.py"]
