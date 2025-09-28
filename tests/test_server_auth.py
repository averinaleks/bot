import logging
import pytest

pytest.importorskip("transformers")
pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("fastapi_csrf_protect")

try:
    import server
except RuntimeError:
    pytest.skip("fastapi_csrf_protect is required", allow_module_level=True)
from contextlib import contextmanager
from fastapi.testclient import TestClient


@contextmanager
def make_client(monkeypatch, *, skip_csrf_validation: bool = False):
    def dummy_load_model():
        server.model_manager.tokenizer = object()
        server.model_manager.model = object()

    monkeypatch.setattr(server.model_manager, "load_model", dummy_load_model)
    monkeypatch.setenv("API_KEYS", "testkey")
    if skip_csrf_validation:
        async def _noop_validate(request):
            return None

        monkeypatch.setattr(server.csrf_protect, "validate_csrf", _noop_validate)
    original_keys = server.API_KEYS.copy()
    server.API_KEYS.clear()
    try:
        with TestClient(server.app) as client:
            yield client
    finally:
        server.API_KEYS.clear()
        server.API_KEYS.update(original_keys)


def test_completions_requires_key(monkeypatch, csrf_secret):
    with make_client(monkeypatch, skip_csrf_validation=True) as client:
        resp = client.post("/v1/completions", json={"prompt": "hi"})
        assert resp.status_code == 401
    assert not server.API_KEYS


def test_chat_completions_requires_key(monkeypatch, csrf_secret):
    with make_client(monkeypatch, skip_csrf_validation=True) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401
    assert not server.API_KEYS


def test_check_api_key_masks_sensitive_headers(monkeypatch, csrf_secret, caplog):
    with make_client(monkeypatch, skip_csrf_validation=True) as client:
        headers = {
            "Authorization": "Bearer secret-token",
            "Cookie": "session=abc",
            "X-API-Key": "top-secret",
        }
        with caplog.at_level(logging.WARNING):
            resp = client.post("/v1/completions", json={"prompt": "hi"}, headers=headers)
    assert resp.status_code == 401
    for secret in ("secret-token", "session=abc", "top-secret"):
        assert secret not in caplog.text
    assert caplog.text.count("***") >= 3
    assert not server.API_KEYS


def test_missing_csrf_token_logged(monkeypatch, csrf_secret, caplog):
    with make_client(monkeypatch) as client:
        headers = {"Authorization": "Bearer testkey"}
        with caplog.at_level(logging.WARNING):
            resp = client.post("/v1/completions", json={"prompt": "hi"}, headers=headers)
    assert resp.status_code == 403
    assert resp.json() == {"detail": "CSRF token missing or invalid"}
    assert "CSRF validation failed" in caplog.text
    assert "testkey" not in caplog.text
    assert "***" in caplog.text
    assert not server.API_KEYS
