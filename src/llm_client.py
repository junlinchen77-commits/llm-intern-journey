"""LLM 调用客户端。

设计目标：把不可靠的外部 API 调用封装成对上层而言稳定的接口。
分四层，从下往上：

  ① 传输层   httpx2 发请求、解析 JSON
  ② 错误映射 把 httpx2 的异常翻译成本项目的 LLMCallError
  ③ 重试层   对可恢复错误做指数退避重试，尊重 Retry-After
  ④ 业务层   chat() 返回结构化结果，上层无需认识 httpx2

分层的目的：httpx2 只在本文件出现。换用其他 SDK 时，
只需修改本文件内部实现，所有调用方代码不变。
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx2

from src.config import Settings, get_settings
from src.exceptions import LLMCallError, LLMResponseError

logger = logging.getLogger(__name__)

# 可重试的 HTTP 状态码。
# 5xx 是服务端临时故障；429 虽然属于 4xx（客户端侧），
# 但语义是「请求过于频繁，稍后重试」，属于可恢复错误。
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# 退避基数（秒）。第 n 次失败后的等待 = base * 2^(n-1) + 抖动
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 20.0


@dataclass(frozen=True)
class TokenUsage:
    """一次调用的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_api(cls, raw: dict[str, Any] | None) -> "TokenUsage":
        """从 API 响应的 usage 字段构造。缺字段时按 0 处理。"""
        raw = raw or {}
        return cls(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            total_tokens=int(raw.get("total_tokens") or 0),
        )


@dataclass(frozen=True)
class ChatResult:
    """一次对话调用的结果。

    Attributes:
        content: 模型返回的文本。
        model: 服务端实际使用的模型名。注意这可能与请求的模型名不同——
            服务商常用别名路由，成本核算与效果对比应以本字段为准。
        usage: token 用量。
        attempts: 实际发起的请求次数（含首次）。
        request_id: 本次调用使用的幂等键。
    """

    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    attempts: int = 1
    request_id: str = ""


class LLMClient:
    """OpenAI 兼容接口的客户端。

    用法::

        client = LLMClient()
        result = client.chat("你好")
        print(result.content, result.usage.total_tokens)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = httpx2.Client(
            base_url=self._settings.llm_base_url.rstrip("/"),
            timeout=httpx2.Timeout(
                connect=5.0,
                read=float(self._settings.llm_timeout_seconds),
                write=10.0,
                pool=5.0,
            ),
            headers={
                "Authorization": f"Bearer {self._settings.llm_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
             trust_env=self._settings.llm_trust_env, 
        )

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_retries: int | None = None,
        idempotency_key: str | None = None,
    ) -> ChatResult:
        """发送一条用户消息并返回模型回复。

        Args:
            message: 用户消息内容。
            system: 可选的 system 提示。
            temperature: 采样温度，0 表示尽量确定性输出。
            max_retries: **额外**重试次数。总请求次数 = 1 + max_retries。
                例如 max_retries=3 时最多发起 4 次请求。
                为 None 时取配置中的 llm_max_retries。
            idempotency_key: 幂等键。为 None 时自动生成。
                重试时复用同一个键，使服务端能识别重复请求，
                避免「已执行但响应丢失」导致的重复计费或重复副作用。

        Returns:
            ChatResult: 含回复文本、实际模型名、token 用量与尝试次数。

        Raises:
            LLMCallError: 调用失败（含重试耗尽）。异常上的 retryable
                表明是否属于可恢复错误。
            LLMResponseError: 连通但响应结构不符合预期。
        """
        if not message or not message.strip():
            raise LLMCallError("message 不能为空", retryable=False)

        retries = self._settings.llm_max_retries if max_retries is None else max_retries
        if retries < 0:
            raise LLMCallError("max_retries 不能为负数", retryable=False)

        total_attempts = 1 + retries
        request_id = idempotency_key or str(uuid.uuid4())

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }

        last_error: LLMCallError | None = None
        last_exc: Exception | None = None

        for attempt in range(1, total_attempts + 1):
            # 每轮循环独立处理，避免跨分支共享未定义变量。
            response: httpx2.Response | None = None
            try:
                response = self._client.post(
                    "/chat/completions",
                    json=payload,
                    headers={"Idempotency-Key": request_id},
                )
            except (httpx2.TimeoutException, httpx2.TransportError) as exc:
                # 网络层失败：请求可能已到达服务端，复用同一幂等键重试是安全的。
                # 记录原始异常，便于重试耗尽后通过 raise from 保留因果链。
                last_exc = exc
                last_error = LLMCallError(
                    f"网络请求失败: {type(exc).__name__}",
                    attempts=attempt,
                    retryable=True,
                )
                logger.warning(
                    "LLM 网络层失败 attempt=%d/%d type=%s",
                    attempt, total_attempts, type(exc).__name__,
                )
            else:
                if response.status_code in RETRYABLE_STATUS_CODES:
                    last_error = LLMCallError(
                        f"服务端可恢复错误: HTTP {response.status_code}",
                        status_code=response.status_code,
                        attempts=attempt,
                        retryable=True,
                    )
                    logger.warning(
                        "LLM 可恢复错误 attempt=%d/%d status=%d",
                        attempt, total_attempts, response.status_code,
                    )
                elif response.status_code >= 400:
                    # 4xx（除 429）：请求本身有问题，重试无意义，立即失败。
                    raise LLMCallError(
                        f"请求被拒绝: HTTP {response.status_code} - {response.text[:200]}",
                        status_code=response.status_code,
                        attempts=attempt,
                        retryable=False,
                    )
                else:
                    return self._parse_response(
                        response, attempts=attempt, request_id=request_id
                    )

            if attempt < total_attempts:
                delay = self._compute_delay(attempt, response=response)
                logger.info("LLM 调用将在 %.2fs 后重试", delay)
                time.sleep(delay)

        assert last_error is not None
        if last_exc is not None:
            raise last_error from last_exc
        raise last_error

    def close(self) -> None:
        """关闭底层连接池。"""
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_delay(attempt: int, *, response: httpx2.Response | None) -> float:
        """计算重试等待时间。

        优先尊重服务端返回的 Retry-After 头；否则用指数退避加随机抖动。
        抖动的作用：多个客户端同时被限流时，避免它们在同一时刻齐步重试
        （惊群效应），否则会再次触发限流。
        """
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), _BACKOFF_MAX_SECONDS)
                except ValueError:
                    logger.debug("无法解析 Retry-After: %r", retry_after)

        backoff = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        jitter = random.uniform(0, backoff * 0.3)
        return min(backoff + jitter, _BACKOFF_MAX_SECONDS)

    @staticmethod
    def _parse_response(
        response: httpx2.Response,
        *,
        attempts: int,
        request_id: str,
    ) -> ChatResult:
        """把 HTTP 响应解析成 ChatResult。

        连通但结构异常时抛 LLMResponseError——这类问题重试无用，
        因为说明请求方式或对方接口变了。
        """
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                f"响应不是合法 JSON: {response.text[:200]}"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"响应结构不符合 OpenAI 兼容格式: {str(data)[:200]}"
            ) from exc

        return ChatResult(
            content=content or "",
            model=str(data.get("model") or ""),
            usage=TokenUsage.from_api(data.get("usage")),
            attempts=attempts,
            request_id=request_id,
        )
