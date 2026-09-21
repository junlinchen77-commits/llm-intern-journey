"""API 层测试。

用 FastAPI 自带的 TestClient，无需真实启动 uvicorn 即可测试接口。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_token_count_chinese():
    response = client.get("/token-count", params={"text": "你好世界"})
    assert response.status_code == 200
    assert response.json() == {"tokens": 4}


def test_token_count_missing_param_returns_422():
    response = client.get("/token-count")
    assert response.status_code == 422


def test_token_count_empty_text_returns_422():
    """空字符串被 min_length=1 拒绝。"""
    response = client.get("/token-count", params={"text": ""})
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["type"] == "string_too_short"
    assert detail["loc"] == ["query", "text"]


def test_token_count_whitespace_only_returns_zero():
    """纯空白输入被 strip 后计为 0 个 token。"""
    response = client.get("/token-count", params={"text": "   "})
    assert response.status_code == 200
    assert response.json() == {"tokens": 0}


def test_token_count_text_at_max_length_is_accepted():
    """恰好等于上限的输入应被接受（边界内）。"""
    response = client.get("/token-count", params={"text": "a" * 10000})
    assert response.status_code == 200


def test_token_count_text_over_max_length_returns_422():
    """超过上限 1 个字符即被拒绝（边界外）。"""
    response = client.get("/token-count", params={"text": "a" * 10001})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "string_too_long"

if __name__ == "__main__":
    test_health_returns_ok()
    test_token_count_chinese()
    test_token_count_missing_param_returns_422()
    test_token_count_empty_text_returns_422()
    test_token_count_whitespace_only_returns_zero()
    test_token_count_text_at_max_length_is_accepted()
    test_token_count_text_over_max_length_returns_422()
    print("all api tests passed.")
    