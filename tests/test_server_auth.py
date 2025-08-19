import os
import pytest

os.environ["API_KEYS"] = "testkey"

pytest.importorskip("transformers")

import server
from fastapi.testclient import TestClient


def make_client():
    def dummy_load_model():
        server.tokenizer = object()
        server.model = object()
    server.load_model = dummy_load_model
    return TestClient(server.app)


def test_completions_requires_key():
    with make_client() as client:
        resp = client.post("/v1/completions", json={"prompt": "hi"})
        assert resp.status_code == 401


def test_chat_completions_requires_key():
    with make_client() as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401
