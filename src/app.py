"""FastAPI 应用入口。

把本地模块（token_estimator）暴露为 HTTP 接口。
"""

from fastapi import FastAPI

from src.models import HealthResponse, TokenCountResponse
from src.token_estimator import estimate_tokens

app = FastAPI(title="LLM Intern Journey API", version="0.1.0")


@app.get("/health")
def health_check() -> HealthResponse:
    """健康检查接口，用于确认服务是否存活。"""
    return HealthResponse(status="ok")


@app.get("/token-count")
def token_count(text: str) -> TokenCountResponse:
    """估算给定文本的 token 数量。

    Args:
        text: 查询参数中传入的待估算文本。

    Returns:
        TokenCountResponse: 含 tokens 字段的响应模型。
    """
    return TokenCountResponse(tokens=estimate_tokens(text))