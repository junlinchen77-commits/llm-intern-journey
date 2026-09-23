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

import json
import logging
import random
import time
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import httpx2

from src.config import Settings, get_settings
from src.exceptions import LLMCallError, LLMResponseError, LLMStreamInterruptedError

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


@dataclass(frozen=True)
class ChatStreamChunk:
    """流式输出中的一个片段。

    Attributes:
        delta: 本次新增的文本增量。调用方应累积拼接，而非替换。
        model: 服务端实际使用的模型名，通常只在首个片段中非空。
        usage: token 用量。仅在流结束时（若服务端开启统计）出现一次。
        is_final: 是否为流结束的标志片段。
    """

    delta: str = ""
    model: str = ""
    usage: TokenUsage | None = None
    is_final: bool = False


# 消息列表类型：符合 OpenAI 兼容格式的 role/content 字典列表
Messages = list[dict[str, str]]


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
        """发送单条用户消息并返回模型回复。

        这是 chat_messages() 的便捷包装，适用于无历史上下文的单轮调用。
        多轮对话请直接使用 chat_messages()，由调用方传入完整历史。

        Args:
            message: 用户消息内容。
            system: 可选的 system 提示。
            temperature: 采样温度，0 表示尽量确定性输出。
            max_retries: **额外**重试次数。总请求次数 = 1 + max_retries。
                例如 max_retries=3 时最多发起 4 次请求。
                为 None 时取配置中的 llm_max_retries。传 0 表示不重试。
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
        messages = self.build_messages(message, system=system)
        return self.chat_messages(
            messages,
            temperature=temperature,
            max_retries=max_retries,
            idempotency_key=idempotency_key,
        )

    def chat_messages(
        self,
        messages: Messages,
        *,
        temperature: float = 0.0,
        max_retries: int | None = None,
        idempotency_key: str | None = None,
    ) -> ChatResult:
        """按完整消息列表调用模型，用于多轮对话。

        多轮对话由调用方维护完整历史并每次全量传入——服务端不保存会话状态，
        这样任意实例都能处理任意请求，便于水平扩展。

        Args:
            messages: 消息列表，每项形如 {"role": "user", "content": "..."}。
                role 取值：system / user / assistant。
            temperature: 采样温度。
            max_retries: 额外重试次数，语义同 chat()。
            idempotency_key: 幂等键，语义同 chat()。

        Returns:
            ChatResult: 调用结果。

        Raises:
            LLMCallError: messages 为空/非法，或调用失败（含重试耗尽）。
            LLMResponseError: 响应结构不符合预期。
        """
        payload = self._build_payload(messages, temperature=temperature, stream=False)
        response = self._execute(payload, max_retries=max_retries, idempotency_key=idempotency_key)
        return self._parse_response(
            response,
            attempts=response.extensions.get("llm_attempts", 1),
            request_id=response.extensions.get("llm_request_id", ""),
        )

    def chat_stream(
        self,
        messages: Messages,
        *,
        temperature: float = 0.0,
        max_retries: int | None = None,
        idempotency_key: str | None = None,
    ) -> Iterator[ChatStreamChunk]:
        """按消息列表流式调用模型，逐块产出文本增量。

        Args:
            messages: 消息列表，语义同 chat_messages()。
            temperature: 采样温度。
            max_retries: 额外重试次数。仅在建立连接阶段生效——
                一旦开始接收数据流就不重试，因为部分输出已经产生，
                重试会导致重复内容。此时抛 LLMStreamInterruptedError。
            idempotency_key: 幂等键，语义同 chat()。

        Yields:
            ChatStreamChunk: 文本增量片段，最后一块的 is_final 为 True。

        Raises:
            LLMCallError: 建流阶段失败（含重试耗尽）。
            LLMStreamInterruptedError: 流中途断开。
        """
        payload = self._build_payload(messages, temperature=temperature, stream=True)
        # 流式场景必须让 stream() 上下文覆盖整个读取过程：
        # httpx 的 stream() 是 @contextmanager，退出时会 close() 掉 response。
        # 若提前退出上下文再交给调用方迭代，响应会被关闭并抛 StreamClosed。
        #
        # 注意：_execute_stream 返回的是已 __enter__ 过的上下文管理器，
        # 这里不能再对其使用 with（重复 __enter__ 会破坏 _GeneratorContextManager）。
        stream_cm, response = self._execute_stream(
            payload,
            max_retries=max_retries,
            idempotency_key=idempotency_key,
        )
        try:
            yield from self._iter_stream(response)
        finally:
            stream_cm.__exit__(None, None, None)

    @staticmethod
    def build_messages(message: str, *, system: str | None = None) -> Messages:
        """把单条消息（可选 system）组装成消息列表。"""
        messages: Messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        return messages

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: Messages,
        *,
        temperature: float,
        stream: bool,
    ) -> dict[str, Any]:
        """校验消息并组装请求体。"""
        if not messages:
            raise LLMCallError("messages 不能为空", retryable=False)

        for index, item in enumerate(messages):
            role = item.get("role")
            if role not in ("system", "user", "assistant"):
                raise LLMCallError(
                    f"messages[{index}].role 非法: {role!r}，应为 system/user/assistant",
                    retryable=False,
                )
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise LLMCallError(
                    f"messages[{index}].content 不能为空",
                    retryable=False,
                )

        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if stream:
            payload["stream"] = True
            # 部分服务商需要显式开启用量统计才会在流末尾返回 usage
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _resolve_retries(self, max_retries: int | None) -> int:
        """确定额外重试次数。

        显式传入 0 表示不重试，必须与「未传入(None)」区分开——
        因此这里用 is None 判断而不是 or。
        """
        if max_retries is None:
            return self._settings.llm_max_retries
        if max_retries < 0:
            raise LLMCallError("max_retries 不能为负数", retryable=False)
        return max_retries

    def _execute(
        self,
        payload: dict[str, Any],
        *,
        max_retries: int | None,
        idempotency_key: str | None,
    ) -> httpx2.Response:
        """执行非流式请求并对可恢复错误重试。

        成功时返回 Response。为让调用方拿到 attempts 与 request_id，
        把它们写入 response.extensions。

        Raises:
            LLMCallError: 重试耗尽或遇到不可重试错误。
        """
        retries = self._resolve_retries(max_retries)
        total_attempts = 1 + retries
        request_id = idempotency_key or str(uuid.uuid4())

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
                    response.extensions["llm_attempts"] = attempt
                    response.extensions["llm_request_id"] = request_id
                    return response

            if attempt < total_attempts:
                delay = self._compute_delay(attempt, response=response)
                logger.info("LLM 调用将在 %.2fs 后重试", delay)
                time.sleep(delay)

        assert last_error is not None
        if last_exc is not None:
            raise last_error from last_exc
        raise last_error

    def _execute_stream(
        self,
        payload: dict[str, Any],
        *,
        max_retries: int | None,
        idempotency_key: str | None,
    ) -> tuple[AbstractContextManager[httpx2.Response], httpx2.Response]:
        """流式请求的执行器，保持连接打开直到调用方读完。

        返回 (上下文管理器, 已进入上下文的 Response) 二元组。
        调用方负责在读完数据后调用上下文的 __exit__ 释放连接。

        与 _execute 的差别：不能直接返回 Response，因为 httpx 的 stream()
        是 @contextmanager，退出即关闭响应，必须由调用方在上下文内完成迭代。

        重试只发生在「建流阶段」——此时尚无任何内容产出。一旦拿到 2xx
        响应，后续读取若中断则不再重试（重试会导致重复内容）。
        """
        retries = self._resolve_retries(max_retries)
        total_attempts = 1 + retries
        request_id = idempotency_key or str(uuid.uuid4())

        last_error: LLMCallError | None = None
        last_exc: Exception | None = None

        for attempt in range(1, total_attempts + 1):
            stream_cm: AbstractContextManager[httpx2.Response] | None = None
            response: httpx2.Response | None = None
            try:
                stream_cm = self._client.stream(
                    "POST",
                    "/chat/completions",
                    json=payload,
                    headers={"Idempotency-Key": request_id},
                )
                response = stream_cm.__enter__()
            except (httpx2.TimeoutException, httpx2.TransportError) as exc:
                last_exc = exc
                last_error = LLMCallError(
                    f"网络请求失败: {type(exc).__name__}",
                    attempts=attempt,
                    retryable=True,
                )
                logger.warning(
                    "LLM 流式建连失败 attempt=%d/%d type=%s",
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
                        "LLM 流式可恢复错误 attempt=%d/%d status=%d",
                        attempt, total_attempts, response.status_code,
                    )
                    stream_cm.__exit__(None, None, None)
                elif response.status_code >= 400:
                    preview = response.read().decode("utf-8", errors="replace")[:200]
                    stream_cm.__exit__(None, None, None)
                    raise LLMCallError(
                        f"请求被拒绝: HTTP {response.status_code} - {preview}",
                        status_code=response.status_code,
                        attempts=attempt,
                        retryable=False,
                    )
                else:
                    return stream_cm, response

            if attempt < total_attempts:
                delay = self._compute_delay(attempt, response=None)
                logger.info("LLM 流式调用将在 %.2fs 后重试", delay)
                time.sleep(delay)

        assert last_error is not None
        if last_exc is not None:
            raise last_error from last_exc
        raise last_error

    def _iter_stream(self, response: httpx2.Response) -> Iterator[ChatStreamChunk]:
        """解析 OpenAI 兼容的 SSE 数据流。

        格式为每行 data: {...}，以 data: [DONE] 结束。
        一旦开始接收数据就不再重试：部分内容已经产出，重试会导致重复。
        """
        model = ""
        usage: TokenUsage | None = None
        try:
            for line in response.iter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    logger.debug("无法解析流式片段，已跳过: %r", data[:200])
                    continue
                if event.get("model"):
                    model = str(event["model"])
                if event.get("usage"):
                    usage = TokenUsage.from_api(event["usage"])
                for choice in event.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield ChatStreamChunk(delta=delta, model=model)
        except (httpx2.TimeoutException, httpx2.TransportError) as exc:
            raise LLMStreamInterruptedError(
                f"流式响应中断: {type(exc).__name__}"
            ) from exc

        yield ChatStreamChunk(model=model, usage=usage, is_final=True)

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

    def close(self) -> None:
        """关闭底层连接池。"""
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
