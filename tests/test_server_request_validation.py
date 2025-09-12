import os
import pytest
pytest.importorskip("transformers")
pytest.importorskip("httpx")
pytest.importorskip("fastapi_csrf_protect")

os.environ.setdefault("CSRF_SECRET", "testsecret")
try:
    import server
except RuntimeError:
    pytest.skip("fastapi_csrf_protect is required", allow_module_level=True)
from contextlib import contextmanager
from fastapi.testclient import TestClient
from fastapi_csrf_protect import CsrfProtect


HEADERS = {"Authorization": "Bearer testkey"}


@contextmanager
def make_client(monkeypatch):
    def dummy_load_model():
        server.model_manager.tokenizer = object()
        server.model_manager.model = object()

    monkeypatch.setattr(server.model_manager, "load_model", dummy_load_model)
    monkeypatch.setenv("API_KEYS", "testkey")
    original_keys = server.API_KEYS.copy()
    server.API_KEYS.clear()
    try:
        with TestClient(server.app) as client:
            csrf = CsrfProtect()
            token, signed = csrf.generate_csrf_tokens()
            client.cookies.set("fastapi-csrf-token", signed)
            headers = HEADERS | {"X-CSRF-Token": token}
            yield client, headers
    finally:
        server.API_KEYS.clear()
        server.API_KEYS.update(original_keys)


def test_chat_completions_validation(monkeypatch):
    with make_client(monkeypatch) as (client, headers):
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
    assert not server.API_KEYS


def test_completions_validation(monkeypatch):
    with make_client(monkeypatch) as (client, headers):
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
    assert not server.API_KEYS
