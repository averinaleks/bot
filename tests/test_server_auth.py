import os
import logging
import pytest

os.environ["API_KEYS"] = "testkey"

pytest.importorskip("transformers")

import server
from fastapi.testclient import TestClient


def make_client(monkeypatch):
    def dummy_load_model():
        server.model_manager.tokenizer = object()
        server.model_manager.model = object()
    monkeypatch.setattr(server.model_manager, "load_model", dummy_load_model)
    return TestClient(server.app)


def test_completions_requires_key(monkeypatch):
    with make_client(monkeypatch) as client:
        resp = client.post("/v1/completions", json={"prompt": "hi"})
        assert resp.status_code == 401


def test_chat_completions_requires_key(monkeypatch):
    with make_client(monkeypatch) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401


def test_check_api_key_masks_sensitive_headers(monkeypatch, caplog):
    with make_client(monkeypatch) as client:
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
