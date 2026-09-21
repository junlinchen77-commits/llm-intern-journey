"""应用配置。

配置来源优先级（高到低）：
  1. 真实环境变量（部署环境、Docker、CI 注入）
  2. .env 文件（本地开发）

密钥绝不写入代码或提交到仓库：.env 已被 .gitignore 排除，
仓库中只保留 .env.example 作为配置说明。
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量与 .env 文件加载的应用配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM API ---
    llm_api_key: SecretStr = Field(
        description="LLM 服务商的 API 密钥。使用 SecretStr 避免日志意外打印明文。"
    )
    llm_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="OpenAI 兼容接口的基础地址。",
    )
    llm_model: str = Field(default="deepseek-chat", description="默认调用的模型名。")
    llm_timeout_seconds: int = Field(default=60, gt=0, description="单次请求超时（秒）。")
    llm_max_retries: int = Field(default=3, ge=0, description="失败重试次数上限。")

    # --- 应用 ---
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", description="运行环境标识。"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="日志级别。"
    )
    token_count_max_length: int = Field(
        default=10_000, gt=0, description="token 计数接口允许的最大输入字符数。"
    )


@lru_cache
def get_settings() -> Settings:
    """返回进程内单例配置对象。

    用 lru_cache 保证只解析一次 .env；测试中如需覆盖配置，
    可调用 get_settings.cache_clear() 后重新获取。
    """
    return Settings()