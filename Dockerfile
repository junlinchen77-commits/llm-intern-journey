# syntax=docker/dockerfile:1

# ---------- 基础镜像 ----------
FROM python:3.11-slim

# 环境变量：
#   PYTHONDONTWRITEBYTECODE —— 不生成 .pyc，保持镜像干净
#   PYTHONUNBUFFERED        —— 日志实时输出，否则会被缓冲导致看不到日志
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---------- 依赖层（独立且靠前，便于缓存复用）----------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 代码层 ----------
COPY src/ ./src/

# ---------- 运行时 ----------
EXPOSE 8000

# 以非 root 用户运行，降低容器逃逸后的影响面
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]