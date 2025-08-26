import os
import pytest

os.environ["CSRF_SECRET"] = "testsecret"

pytest.importorskip("transformers")

import server
from fastapi.testclient import TestClient


def make_client(monkeypatch):
    def dummy_load_model():
        server.model_manager.tokenizer = object()
        server.model_manager.model = object()

    monkeypatch.setattr(server.model_manager, "load_model", dummy_load_model)
    monkeypatch.setenv("API_KEYS", "testkey")
    server.API_KEYS.clear()
    return TestClient(server.app, raise_server_exceptions=False)


def test_unexpected_csrf_error_returns_500(monkeypatch):
    with make_client(monkeypatch) as client:

        def raise_runtime_error(request):
            raise RuntimeError("boom")

        monkeypatch.setattr(server.csrf_protect, "validate_csrf", raise_runtime_error)
        headers = {"Authorization": "Bearer testkey"}
        resp = client.post("/v1/completions", json={"prompt": "hi"}, headers=headers)
        assert resp.status_code == 500
        assert "Internal Server Error" in resp.text
