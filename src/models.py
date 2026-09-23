"""API 请求与响应的数据模型。

用 Pydantic 模型替代裸 dict，使接口契约显式化：
类型校验、自动文档、示例值都由模型定义生成。
"""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(description="服务状态，正常时为 ok", examples=["ok"])


class TokenCountResponse(BaseModel):
    """token 计数响应。"""

    tokens: int = Field(description="估算出的 token 数量", examples=[4])

class ChatMessage(BaseModel):
    """一条对话消息。"""

    role: Literal["system", "user", "assistant"] = Field(
        description="消息角色。system 设定行为，user 是用户输入，assistant 是模型的历史回复。"
    )
    content: str = Field(
        min_length=1,
        max_length=8_000,
        description="消息内容。",
        examples=["你好"],
    )


class ChatRequest(BaseModel):
    """对话请求。

    多轮对话由调用方维护完整历史并每次全量传入，服务端不保存会话状态——
    这样任意实例都能处理任意请求，便于水平扩展。
    """

    message: str = Field(
        min_length=1,
        max_length=8_000,
        description="用户消息内容。",
        examples=["用一句话解释什么是 RAG"],
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=200,
        description="历史消息列表，按时间顺序排列，不含本次 message。"
        "服务端会拼成 [system?] + history + [user message] 后发给模型。",
    )
    system: str | None = Field(
        default=None,
        max_length=4_000,
        description="可选的 system 提示，用于设定模型角色。",
        examples=["你是一名网络安全工程师，回答要简洁准确。"],
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="采样温度。0 表示尽量确定性的输出。",
    )


class ChatUsage(BaseModel):
    """token 用量。"""

    prompt_tokens: int = Field(description="输入消耗的 token 数。")
    completion_tokens: int = Field(description="输出消耗的 token 数。")
    total_tokens: int = Field(
        description="总 token 数。与部分服务商的语义可能不同，"
        "不保证等于 prompt_tokens + completion_tokens。"
    )


class ChatResponse(BaseModel):
    """对话响应。"""

    reply: str = Field(description="模型返回的文本。")
    model: str = Field(
        description="服务端实际使用的模型名。可能与请求的模型名不同——"
        "服务商常用别名路由，计费与效果对比应以本字段为准。"
    )
    usage: ChatUsage = Field(description="本次调用的 token 用量。")
    attempts: int = Field(
        description="实际发起的请求次数（含首次）。大于 1 表示发生过重试。"
    )
    request_id: str = Field(
        description="本次调用的幂等键，可用于日志关联或向服务商提工单。"
    )