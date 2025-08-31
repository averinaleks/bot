import os
import pytest

pytest.importorskip("transformers")

os.environ.setdefault("CSRF_SECRET", "testsecret")
import server
from contextlib import contextmanager
from fastapi.testclient import TestClient


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
        with TestClient(server.app, raise_server_exceptions=False) as client:
            yield client
    finally:
        server.API_KEYS.clear()
        server.API_KEYS.update(original_keys)


def test_unexpected_csrf_error_returns_500(monkeypatch):
    with make_client(monkeypatch) as client:

        def raise_runtime_error(request):
            raise RuntimeError("boom")

        monkeypatch.setattr(server.csrf_protect, "validate_csrf", raise_runtime_error)
        headers = {"Authorization": "Bearer testkey"}
        resp = client.post("/v1/completions", json={"prompt": "hi"}, headers=headers)
        assert resp.status_code == 500
        assert "Internal Server Error" in resp.text
    assert not server.API_KEYS
