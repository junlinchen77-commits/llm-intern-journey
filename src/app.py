

"""FastAPI 应用入口。

把本地模块（token_estimator、llm_client）暴露为 HTTP 接口。
"""

from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import Field

from src.exceptions import LLMCallError, LLMError, LLMResponseError
from src.llm_client import LLMClient
from src.models import (
    ChatRequest,
    ChatResponse,
    ChatUsage,
    HealthResponse,
    TokenCountResponse,
)
from src.token_estimator import estimate_tokens

app = FastAPI(title="LLM Intern Journey API", version="0.2.0")



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


@app.post("/chat")
def chat(payload: ChatRequest) -> ChatResponse:
    """调用大模型生成回复。

    错误映射遵循「客户端错误 vs 服务端错误」的区分：
      - 重试耗尽的临时故障（限流、上游 5xx）→ 503，调用方可稍后重试
      - 请求被模型服务商拒绝（4xx）→ 502，重试无意义
      - 响应结构异常 → 502，属上游契约问题
    """
    client = LLMClient()
    try:
        result = client.chat(
            payload.message,
            system=payload.system,
            temperature=payload.temperature,
        )
    except LLMCallError as exc:
        if exc.retryable:
            raise HTTPException(
                status_code=503,
                detail=f"模型服务暂时不可用（已尝试 {exc.attempts} 次），请稍后重试",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"模型服务拒绝了请求（HTTP {exc.status_code}）",
        ) from exc
    except LLMResponseError as exc:
        raise HTTPException(status_code=502, detail="模型响应格式异常") from exc
    except LLMError as exc:
        raise HTTPException(status_code=500, detail="模型调用发生内部错误") from exc
    finally:
        client.close()

    return ChatResponse(
        reply=result.content,
        model=result.model,
        usage=ChatUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        attempts=result.attempts,
        request_id=result.request_id,
    )