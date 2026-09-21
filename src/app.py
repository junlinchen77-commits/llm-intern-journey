"""FastAPI 应用入口。

把本地模块（token_estimator）暴露为 HTTP 接口。
"""

from typing import Annotated

from fastapi import FastAPI
from pydantic import Field

from src.models import HealthResponse, TokenCountResponse
from src.token_estimator import estimate_tokens

app = FastAPI(title="LLM Intern Journey API", version="0.1.0")


@app.get("/health")
def health_check() -> HealthResponse:
    """健康检查接口，用于确认服务是否存活。"""
    return HealthResponse(status="ok")

"""@app.get("/token-count")
def token_count(text: str) -> TokenCountResponse:
    return {"token": estimate_tokens(text)}      # ← 故意返回 dict，不是模型"""

@app.get("/token-count")
def token_count(
    text: Annotated[
        str,
        Field(
            min_length=1,
                        # 上限 1 万字符：约 3000-5000 token，足以覆盖单次文本估算；
            # 且不超出常见 URL 长度限制（浏览器/Nginx/CDN 通常 8KB 以内）。
            # 需要处理更长文本时应改用 POST + 请求体。
            max_length=10_000,
            description="待估算的文本。空白字符会被忽略。",
            examples=["你好世界"],
        ),
    ],
) -> TokenCountResponse:
    """估算给定文本的 token 数量。

    Args:
        text: 查询参数中传入的待估算文本。

    Returns:
        TokenCountResponse: 含 tokens 字段的响应模型。
    """
    return TokenCountResponse(tokens=estimate_tokens(text.strip()))
