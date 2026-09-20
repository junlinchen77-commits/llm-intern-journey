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


if __name__ == "__main__":
    test_health_returns_ok()
    test_token_count_chinese()
    test_token_count_missing_param_returns_422()
    print("all api tests passed.")