"""API 请求与响应的数据模型。

用 Pydantic 模型替代裸 dict，使接口契约显式化：
类型校验、自动文档、示例值都由模型定义生成。
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(description="服务状态，正常时为 ok", examples=["ok"])


class TokenCountResponse(BaseModel):
    """token 计数响应。"""

    tokens: int = Field(description="估算出的 token 数量", examples=[4])