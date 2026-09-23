"""项目自定义异常。

业务代码只抛本模块定义的异常，不向上层泄露第三方库的异常类型。
这样替换底层实现（HTTP 库、模型供应商）时，上层调用方无需改动。
"""


class LLMError(Exception):
    """LLM 相关错误的基类。"""


class LLMConfigError(LLMError):
    """配置错误，例如缺少 API Key 或模型名非法。"""


class LLMCallError(LLMError):
    """调用 LLM 失败。

    Attributes:
        status_code: HTTP 状态码；网络层失败时为 None。
        attempts: 实际发起的请求次数（含首次）。
        retryable: 该错误是否属于可重试类型。
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        attempts: int = 1,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts
        self.retryable = retryable


class LLMResponseError(LLMError):
    """响应格式不符合预期（能连通但返回内容无法解析）。"""


class LLMStreamInterruptedError(LLMError):
    """流式响应中途断开。

    与 LLMCallError 的区别：此类错误发生在已经产出部分内容之后，
    重试会导致重复内容，因此不自动重试。由调用方决定如何处理，
    例如保留已收到的部分并提示用户重新发起请求。
    """
