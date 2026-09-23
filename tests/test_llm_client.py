"""LLMClient 单元测试。

用 mock transport 构造失败场景，不依赖真实网络与 API 额度，
因此结果完全确定，可在 CI 中运行。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx2

from src.exceptions import LLMCallError, LLMResponseError
from src.llm_client import LLMClient, TokenUsage

MOCK_SUCCESS_BODY = {
    "model": "mock-model",
    "choices": [{"message": {"role": "assistant", "content": "成功"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


class ScriptedTransport(httpx2.BaseTransport):
    """按脚本响应的传输层：前 fail_times 次失败，之后成功。

    Args:
        fail_times: 前多少次请求失败。
        status_on_fail: 非 None 时返回该状态码；None 时抛 ConnectError。
        body: 成功时返回的 JSON。
    """

    def __init__(
        self,
        fail_times: int,
        *,
        status_on_fail: int | None = None,
        body: dict | None = None,
    ) -> None:
        self.fail_times = fail_times
        self.status_on_fail = status_on_fail
        self.body = body if body is not None else MOCK_SUCCESS_BODY
        self.call_count = 0

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            if self.status_on_fail is not None:
                return httpx2.Response(
                    self.status_on_fail, text="upstream error", request=request
                )
            raise httpx2.ConnectError("simulated failure", request=request)
        return httpx2.Response(200, json=self.body, request=request)


def make_client(transport: httpx2.BaseTransport, **kwargs) -> LLMClient:
    """构造注入了 mock transport 的客户端。

    通过替换内部 client 避免真实网络；这是测试专用做法，
    生产代码不应访问私有属性。
    """
    client = LLMClient()
    client._client.close()
    client._client = httpx2.Client(
        base_url="http://mock.local/v1",
        transport=transport,
        headers={"Authorization": "Bearer mock"},
        **kwargs,
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
    client.close()


def test_chat_recovers_after_transient_network_failure():
    """前 2 次网络异常，第 3 次成功——应返回结果而非抛错。"""
    transport = ScriptedTransport(fail_times=2)
    client = make_client(transport)
    result = client.chat("你好", max_retries=3)

    assert result.content == "成功"
    assert result.attempts == 3
    assert transport.call_count == 3
    client.close()


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
    client.close()


def test_negative_max_retries_rejected():
    transport = ScriptedTransport(fail_times=0)
    client = make_client(transport)
    try:
        client.chat("你好", max_retries=-1)
        raise AssertionError("应当抛出 LLMCallError")
    except LLMCallError as exc:
        assert exc.retryable is False
    assert transport.call_count == 0
    client.close()


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
    client.close()


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
    client.close()


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
    client.close()


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
    client.close()


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
    client.close()


# ----------------------------------------------------------------------
# 幂等键
# ----------------------------------------------------------------------

def test_idempotency_key_is_reused_across_retries():
    """重试必须复用同一幂等键，否则去重完全失效。"""
    seen_keys: list[str] = []

    class KeyCapturingTransport(httpx2.BaseTransport):
        def handle_request(self, request: httpx2.Request) -> httpx2.Response:
            seen_keys.append(request.headers.get("Idempotency-Key", ""))
            if len(seen_keys) < 3:
                raise httpx2.ConnectError("simulated", request=request)
            return httpx2.Response(200, json=MOCK_SUCCESS_BODY, request=request)

    transport = KeyCapturingTransport()
    client = make_client(transport)
    result = client.chat("你好", max_retries=3)

    assert result.attempts == 3
    assert len(seen_keys) == 3
    assert len(set(seen_keys)) == 1, f"重试时应复用同一键，实际={seen_keys}"
    assert seen_keys[0] != ""
    client.close()


def test_explicit_idempotency_key_is_used():
    seen_keys: list[str] = []

    class KeyCapturingTransport(httpx2.BaseTransport):
        def handle_request(self, request: httpx2.Request) -> httpx2.Response:
            seen_keys.append(request.headers.get("Idempotency-Key", ""))
            return httpx2.Response(200, json=MOCK_SUCCESS_BODY, request=request)

    client = make_client(KeyCapturingTransport())
    result = client.chat("你好", idempotency_key="order-42")
    assert result.request_id == "order-42"
    assert seen_keys == ["order-42"]
    client.close()


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