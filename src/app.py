"""FastAPI 应用入口。

把本地模块（token_estimator、llm_client）暴露为 HTTP 接口。
"""

import json
import logging
from functools import lru_cache
from typing import Annotated, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import Field

from src.exceptions import (
    LLMCallError,
    LLMError,
    LLMResponseError,
    LLMStreamInterruptedError,
)
from src.config import get_settings
from src.llm_client import LLMClient
from src.models import (
    ChatRequest,
    ChatResponse,
    ChatUsage,
    HealthResponse,
    TokenCountResponse,
)
from src.token_estimator import estimate_tokens

logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Intern Journey API", version="0.3.0")

# /token-count 的输入上限，取自配置 token_count_max_length。
# 上限受常见 URL 长度限制约束（浏览器/Nginx/CDN 通常 8KB 以内）；
# 需要处理更长文本时应改用 POST + 请求体。
TOKEN_COUNT_MAX_LENGTH = get_settings().token_count_max_length


@lru_cache
def get_llm_client() -> LLMClient:
    """返回进程内共享的 LLM 客户端。

    用单例而非每请求新建：httpx 的连接池需要复用，否则每次请求都要
    重新做 TLS 握手（约 100-300ms）。lru_cache 保证只初始化一次，
    且不依赖 FastAPI 的 lifespan，测试中无需进入应用上下文。
    """
    return LLMClient()


@app.get("/health")
def health_check() -> HealthResponse:
    """健康检查接口，用于确认服务是否存活。"""
    return HealthResponse(status="ok")


@app.get("/token-count")
def token_count(
    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=TOKEN_COUNT_MAX_LENGTH,
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


def _build_messages(payload: ChatRequest) -> list[dict[str, str]]:
    """把请求中的 system / history / message 拼成消息列表。

    顺序遵循 OpenAI 兼容约定：system 在最前，历史居中，本次用户消息最后。
    """
    messages: list[dict[str, str]] = []
    if payload.system:
        messages.append({"role": "system", "content": payload.system})
    messages.extend({"role": item.role, "content": item.content} for item in payload.history)
    messages.append({"role": "user", "content": payload.message})
    return messages


def _to_http_exception(exc: LLMError) -> HTTPException:
    """把 LLM 层异常映射为 HTTP 状态码。

    映射原则：让调用方能判断「该不该重试」。
      - 503：上游临时故障且重试已耗尽，稍后重试可能成功
      - 502：上游明确拒绝或响应异常，重试无意义
      - 500：未预期的内部错误
    """
    if isinstance(exc, LLMCallError):
        if exc.retryable:
            return HTTPException(
                status_code=503,
                detail=f"模型服务暂时不可用（已尝试 {exc.attempts} 次），请稍后重试",
            )
        return HTTPException(
            status_code=502,
            detail=f"模型服务拒绝了请求（HTTP {exc.status_code}）",
        )
    if isinstance(exc, LLMResponseError):
        return HTTPException(status_code=502, detail="模型响应格式异常")
    return HTTPException(status_code=500, detail="模型调用发生内部错误")


@app.post("/chat")
def chat(payload: ChatRequest) -> ChatResponse:
    """调用大模型生成回复。

    多轮对话请把之前的往来放进 history，服务端不保存会话状态。
    """
    client = get_llm_client()
    try:
        result = client.chat_messages(
            _build_messages(payload),
            temperature=payload.temperature,
        )
    except LLMError as exc:
        raise _to_http_exception(exc) from exc

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


def _sse(payload: dict[str, object]) -> str:
    """把字典编码为一条 SSE 事件。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """流式调用大模型，以 SSE 逐块返回文本增量。

    事件格式（每条以 data: 开头，双换行分隔）：
      - {"delta": "文本", "model": "..."}  增量片段，需累积拼接
      - {"usage": {...}, "is_final": true} 结束标志，含 token 统计
      - {"error": "...", "code": "..."}    错误（流中途出错时）

    用 POST 而非 EventSource 的原因：需要发送结构化请求体。
    浏览器端可用 fetch + ReadableStream 消费，或直接用支持 SSE 的客户端。
    """
    client = get_llm_client()

    def event_generator() -> Iterator[str]:
        try:
            for chunk in client.chat_stream(
                _build_messages(payload),
                temperature=payload.temperature,
            ):
                if chunk.is_final:
                    final: dict[str, object] = {"is_final": True, "model": chunk.model}
                    if chunk.usage is not None:
                        final["usage"] = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }
                    yield _sse(final)
                    break
                yield _sse({"delta": chunk.delta, "model": chunk.model})
        except LLMStreamInterruptedError as exc:
            # 流已中断，无法通过状态码表达（响应头早已发出），只能作为事件下发。
            logger.warning("流式响应中断: %s", exc)
            yield _sse({"error": "生成过程中断，请重新发起请求", "code": "stream_interrupted"})
        except LLMError as exc:
            logger.warning("流式调用失败: %s", exc)
            yield _sse({"error": "模型调用失败", "code": "llm_error"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，否则流会被攒着一起发
        },
    )
