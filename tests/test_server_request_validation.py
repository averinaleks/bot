import os
import pytest

os.environ["CSRF_SECRET"] = "testsecret"

pytest.importorskip("transformers")
pytest.importorskip("fastapi_csrf_protect")

import server
from fastapi.testclient import TestClient
from fastapi_csrf_protect import CsrfProtect


HEADERS = {"Authorization": "Bearer testkey"}


def make_client(monkeypatch):
    def dummy_load_model():
        server.model_manager.tokenizer = object()
        server.model_manager.model = object()
    monkeypatch.setattr(server.model_manager, "load_model", dummy_load_model)
    monkeypatch.setenv("API_KEYS", "testkey")
    server.API_KEYS.clear()
    client = TestClient(server.app)
    csrf = CsrfProtect()
    token, signed = csrf.generate_csrf_tokens()
    client.cookies.set("fastapi-csrf-token", signed)
    headers = HEADERS | {"X-CSRF-Token": token}
    return client, headers


def test_chat_completions_validation(monkeypatch):
    client, headers = make_client(monkeypatch)
    with client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "temperature": 2.1},
            headers=headers,
        )
        assert resp.status_code == 422

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 513},
            headers=headers,
        )
        assert resp.status_code == 422


def test_completions_validation(monkeypatch):
    client, headers = make_client(monkeypatch)
    with client:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "temperature": -0.1},
            headers=headers,
        )
        assert resp.status_code == 422

        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 0},
            headers=headers,
        )
        assert resp.status_code == 422
