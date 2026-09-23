"""LLMClient 单元测试。

用 mock transport 构造失败与流式场景，不依赖真实网络与 API 额度，
因此结果完全确定，可在 CI 中运行。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx2

from src.exceptions import LLMCallError, LLMResponseError
from src.llm_client import ChatStreamChunk, LLMClient, TokenUsage

MOCK_SUCCESS_BODY = {
    "model": "mock-model",
    "choices": [{"message": {"role": "assistant", "content": "成功"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


def _sse_body(
    deltas: list[str],
    *,
    usage: dict | None = None,
    model: str = "mock-model",
) -> bytes:
    """构造 OpenAI 兼容的 SSE 响应体。"""
    lines: list[str] = []
    for delta in deltas:
        event = {
            "model": model,
            "choices": [{"index": 0, "delta": {"content": delta}}],
        }
        lines.append(f"data: {json.dumps(event)}\n\n")
    if usage is not None:
        # 用量通常作为独立事件出现在末尾，choices 为空
        lines.append(
            f"data: {json.dumps({'model': model, 'choices': [], 'usage': usage})}\n\n"
        )
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


class ScriptedTransport(httpx2.BaseTransport):
    """按脚本响应的传输层：前 fail_times 次失败，之后成功。

    Args:
        fail_times: 前多少次请求失败。
        status_on_fail: 非 None 时返回该状态码；None 时抛 ConnectError。
        body: 成功时返回的 JSON。
        sse_body: 非 None 时成功响应改为带该字节流（用于流式测试）。
    """

    def __init__(
        self,
        fail_times: int,
        *,
        status_on_fail: int | None = None,
        body: dict | None = None,
        sse_body: bytes | None = None,
    ) -> None:
        self.fail_times = fail_times
        self.status_on_fail = status_on_fail
        self.body = body if body is not None else MOCK_SUCCESS_BODY
        self.sse_body = sse_body
        self.call_count = 0
        self.seen_keys: list[str] = []
        self.seen_payloads: list[dict] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.call_count += 1
        self.seen_keys.append(request.headers.get("Idempotency-Key", ""))
        if request.content:
            try:
                self.seen_payloads.append(json.loads(request.content))
            except ValueError:
                pass

        if self.call_count <= self.fail_times:
            if self.status_on_fail is not None:
                return httpx2.Response(
                    self.status_on_fail, text="upstream error", request=request
                )
            raise httpx2.ConnectError("simulated failure", request=request)

        if self.sse_body is not None:
            return httpx2.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                stream=httpx2.ByteStream(self.sse_body),
                request=request,
            )
        return httpx2.Response(200, json=self.body, request=request)


def make_client(transport: httpx2.BaseTransport) -> LLMClient:
    """构造注入了 mock transport 的客户端。

    通过替换内部 client 避免真实网络；这是测试专用做法，
    生产代码不应访问私有属性。
    """
    client = LLMClient()
    client._client = httpx2.Client(
        base_url="http://mock.local/v1",
        transport=transport,
        headers={"Authorization": "Bearer mock"},
    )
    return client


# ----------------------------------------------------------------------
# token 用量解析
# ----------------------------------------------------------------------

def test_token_usage_from_api_normal():
    usage = TokenUsage.from_api(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    assert usage == TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_token_usage_from_api_missing_fields_defaults_to_zero():
    """字段缺失时应降级为 0，而不是抛异常——API 可能不返回 usage。"""
    assert TokenUsage.from_api(None) == TokenUsage(0, 0, 0)
    assert TokenUsage.from_api({}) == TokenUsage(0, 0, 0)


# ----------------------------------------------------------------------
# 正常路径
# ----------------------------------------------------------------------

def test_chat_success_parses_content_and_usage():
    transport = ScriptedTransport(fail_times=0)
    client = make_client(transport)
    result = client.chat("你好")

    assert result.content == "成功"
    assert result.model == "mock-model"
    assert result.usage.total_tokens == 4
    assert result.attempts == 1
    assert transport.call_count == 1


def test_chat_recovers_after_transient_network_failure():
    """前 2 次网络异常，第 3 次成功——应返回结果而非抛错。"""
    transport = ScriptedTransport(fail_times=2)
    client = make_client(transport)
    result = client.chat("你好", max_retries=3)

    assert result.content == "成功"
    assert result.attempts == 3
    assert transport.call_count == 3


# ----------------------------------------------------------------------
# 参数校验
# ----------------------------------------------------------------------

def test_empty_message_rejected_without_any_request():
    """空消息应在发请求之前被拒绝，transport 不应被调用。"""
    transport = ScriptedTransport(fail_times=0)
    client = make_client(transport)

    try:
        client.chat("   ")
        raise AssertionError("应当抛出 LLMCallError")
    except LLMCallError as exc:
        assert exc.retryable is False
        assert exc.attempts == 1

    assert transport.call_count == 0


def test_negative_max_retries_rejected():
    transport = ScriptedTransport(fail_times=0)
    client = make_client(transport)
    try:
        client.chat("你好", max_retries=-1)
        raise AssertionError("应当抛出 LLMCallError")
    except LLMCallError as exc:
        assert exc.retryable is False
    assert transport.call_count == 0


def test_invalid_role_rejected_without_request():
    """非法 role 应在发请求前被拦下，错误信息需指出下标。"""
    transport = ScriptedTransport(fail_times=0)
    client = make_client(transport)
    try:
        client.chat_messages([{"role": "robot", "content": "你好"}])
        raise AssertionError("应当抛出 LLMCallError")
    except LLMCallError as exc:
        assert "messages[0].role" in str(exc)
        assert exc.retryable is False
    assert transport.call_count == 0


def test_empty_messages_rejected():
    transport = ScriptedTransport(fail_times=0)
    client = make_client(transport)
    try:
        client.chat_messages([])
        raise AssertionError("应当抛出 LLMCallError")
    except LLMCallError as exc:
        assert exc.retryable is False
    assert transport.call_count == 0


# ----------------------------------------------------------------------
# 重试语义
# ----------------------------------------------------------------------

def test_total_attempts_equals_one_plus_max_retries():
    """max_retries 是「额外重试次数」：总请求数 = 1 + max_retries。"""
    transport = ScriptedTransport(fail_times=99)
    client = make_client(transport)
    try:
        client.chat("你好", max_retries=2)
        raise AssertionError("应当抛出 LLMCallError")
    except LLMCallError as exc:
        assert exc.attempts == 3
    assert transport.call_count == 3


def test_zero_max_retries_means_no_retry():
    """max_retries=0 应被尊重为「不重试」，而不是回退到配置值。

    这条锁住 _resolve_retries 里的 is None 判断：
    若有人改成 `max_retries or 配置值`，0 会被替换成配置值，此测试即失败。
    """
    transport = ScriptedTransport(fail_times=99, status_on_fail=500)
    client = make_client(transport)
    try:
        client.chat("你好", max_retries=0)
        raise AssertionError("应当抛出 LLMCallError")
    except LLMCallError as exc:
        assert exc.attempts == 1
    assert transport.call_count == 1


def test_client_error_is_not_retried():
    """400 属于请求本身有问题，重试无意义——应只调用一次。"""
    transport = ScriptedTransport(fail_times=99, status_on_fail=400)
    client = make_client(transport)
    try:
        client.chat("你好", max_retries=3)
        raise AssertionError("应当抛出 LLMCallError")
    except LLMCallError as exc:
        assert exc.retryable is False
        assert exc.status_code == 400
        assert exc.attempts == 1
    assert transport.call_count == 1


def test_rate_limit_is_retried():
    """429 虽然属于 4xx，但语义是「稍后重试」——必须重试。"""
    transport = ScriptedTransport(fail_times=99, status_on_fail=429)
    client = make_client(transport)
    try:
        client.chat("你好", max_retries=2)
        raise AssertionError("应当抛出 LLMCallError")
    except LLMCallError as exc:
        assert exc.status_code == 429
        assert exc.retryable is True
        assert exc.attempts == 3
    assert transport.call_count == 3


def test_network_error_preserves_cause_and_has_no_status_code():
    transport = ScriptedTransport(fail_times=99)
    client = make_client(transport)
    try:
        client.chat("你好", max_retries=1)
        raise AssertionError("应当抛出 LLMCallError")
    except LLMCallError as exc:
        assert exc.status_code is None
        assert exc.retryable is True
        assert isinstance(exc.__cause__, httpx2.ConnectError)


# ----------------------------------------------------------------------
# 响应解析失败
# ----------------------------------------------------------------------

def test_malformed_response_raises_response_error():
    """能连通但结构不对，应抛 LLMResponseError 而不是 LLMCallError。"""
    transport = ScriptedTransport(fail_times=0, body={"unexpected": "shape"})
    client = make_client(transport)
    try:
        client.chat("你好")
        raise AssertionError("应当抛出 LLMResponseError")
    except LLMResponseError:
        pass


# ----------------------------------------------------------------------
# 幂等键
# ----------------------------------------------------------------------

def test_idempotency_key_is_reused_across_retries():
    """重试必须复用同一幂等键，否则去重完全失效。"""
    transport = ScriptedTransport(fail_times=2)
    client = make_client(transport)
    result = client.chat("你好", max_retries=3)

    assert result.attempts == 3
    assert len(transport.seen_keys) == 3
    assert len(set(transport.seen_keys)) == 1, (
        f"重试时应复用同一键，实际={transport.seen_keys}"
    )
    assert transport.seen_keys[0] != ""


def test_explicit_idempotency_key_is_used():
    transport = ScriptedTransport(fail_times=0)
    client = make_client(transport)
    result = client.chat("你好", idempotency_key="order-42")
    assert result.request_id == "order-42"
    assert transport.seen_keys == ["order-42"]


# ----------------------------------------------------------------------
# 多轮对话
# ----------------------------------------------------------------------

def test_build_messages_puts_system_first():
    messages = LLMClient.build_messages("问题", system="你是助手")
    assert messages == [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "问题"},
    ]


def test_build_messages_without_system():
    assert LLMClient.build_messages("问题") == [{"role": "user", "content": "问题"}]


def test_chat_messages_sends_history_in_order():
    """历史消息应原样按顺序出现在请求体中，最后才是本次用户消息。"""
    transport = ScriptedTransport(fail_times=0)
    client = make_client(transport)
    client.chat_messages([
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
    ])

    sent = transport.seen_payloads[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[-1]["content"] == "第二问"


# ----------------------------------------------------------------------
# 流式输出
# ----------------------------------------------------------------------

def collect_stream(client: LLMClient, **kwargs) -> list[ChatStreamChunk]:
    return list(client.chat_stream([{"role": "user", "content": "你好"}], **kwargs))


def test_chat_stream_yields_deltas_in_order():
    transport = ScriptedTransport(
        fail_times=0,
        sse_body=_sse_body(
            ["你", "好", "世界"],
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        ),
    )
    client = make_client(transport)
    chunks = collect_stream(client)

    deltas = [c.delta for c in chunks if not c.is_final]
    assert "".join(deltas) == "你好世界"
    assert deltas == ["你", "好", "世界"], "应保持服务端返回的顺序"


def test_chat_stream_marks_final_chunk_with_usage():
    transport = ScriptedTransport(
        fail_times=0,
        sse_body=_sse_body(
            ["a"],
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        ),
    )
    client = make_client(transport)
    chunks = collect_stream(client)

    final = chunks[-1]
    assert final.is_final is True
    assert final.usage == TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5)
    assert final.delta == ""
    assert final.model == "mock-model"


def test_chat_stream_sets_stream_flag_in_payload():
    """流式请求必须在请求体里带 stream=true，否则服务端返回非流式响应。"""
    transport = ScriptedTransport(fail_times=0, sse_body=_sse_body(["a"]))
    client = make_client(transport)
    collect_stream(client)

    payload = transport.seen_payloads[0]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_chat_stream_ignores_done_marker():
    """data: [DONE] 是结束标记，不应被当成内容或导致解析错误。"""
    transport = ScriptedTransport(fail_times=0, sse_body=_sse_body(["x"]))
    client = make_client(transport)
    chunks = collect_stream(client)

    joined = "".join(c.delta for c in chunks)
    assert "DONE" not in joined
    assert joined == "x"


def test_chat_stream_tolerates_malformed_event():
    """单个事件 JSON 损坏时应跳过该事件，而不是中断整个流。"""
    raw = (
        'data: {"model":"m","choices":[{"delta":{"content":"A"}}]}\n\n'
        "data: {not valid json}\n\n"
        'data: {"model":"m","choices":[{"delta":{"content":"B"}}]}\n\n'
        "data: [DONE]\n\n"
    ).encode("utf-8")
    transport = ScriptedTransport(fail_times=0, sse_body=raw)
    client = make_client(transport)
    chunks = collect_stream(client)

    joined = "".join(c.delta for c in chunks)
    assert joined == "AB"


def test_chat_stream_retries_when_connect_fails_before_data():
    """建流阶段失败可以重试：此时还没有产出任何内容。"""
    transport = ScriptedTransport(fail_times=2, sse_body=_sse_body(["ok"]))
    client = make_client(transport)
    chunks = collect_stream(client, max_retries=3)

    assert "".join(c.delta for c in chunks) == "ok"
    assert transport.call_count == 3


if __name__ == "__main__":
    import time

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    start = time.time()
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    elapsed = time.time() - start
    print(f"\n{len(tests) - failed}/{len(tests)} passed in {elapsed:.1f}s")
    if failed:
        sys.exit(1)
