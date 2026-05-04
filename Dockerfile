# 使用官方 Python 轻量级镜像，建议使用 3.11 或以上版本
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，防止 python 缓存以及输出缓冲
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

# 安装系统依赖（如需编译部分包可能需要 gcc，如果不需要可精简）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY app/ ./app/
# 如果需要启动脚本或其他目录也可以一起复制
# COPY scripts/ ./scripts/

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动命令 (请确保与 main.py 和 config 中的配置一致)
# 使用 uvicorn 启动 app.main:app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
