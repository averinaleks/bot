import concurrent.futures
import os
import secrets
import threading

import pytest
pytest.importorskip("transformers")
pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("fastapi_csrf_protect")

if "CSRF_SECRET" not in os.environ:
    os.environ["CSRF_SECRET"] = secrets.token_hex(32)
try:
    import server
except RuntimeError:
    pytest.skip("fastapi_csrf_protect is required", allow_module_level=True)
from contextlib import contextmanager
from fastapi.testclient import TestClient
from fastapi_csrf_protect import CsrfProtect


HEADERS = {"Authorization": "Bearer testkey"}


def _make_chat_payload() -> dict[str, object]:
    return {"messages": [{"role": "user", "content": "hi"}]}


def _make_completion_payload() -> dict[str, object]:
    return {"prompt": "hi"}


def _extract_chat_content(response) -> str:
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _extract_completion_text(response) -> str:
    data = response.json()
    return data["choices"][0]["text"]


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


@pytest.mark.parametrize(
    ("endpoint", "payload_factory", "extract_content"),
    (
        ("/v1/chat/completions", _make_chat_payload, _extract_chat_content),
        ("/v1/completions", _make_completion_payload, _extract_completion_text),
    ),
)
def test_model_busy_returns_429(monkeypatch, endpoint, payload_factory, extract_content):
    start_event = threading.Event()
    release_event = threading.Event()
    event_timeout = 5.0
    slow_text = "slow-response"

    def slow_generate_text(*args, **kwargs):
        start_event.set()
        if not release_event.wait(timeout=event_timeout):
            raise RuntimeError("Timed out waiting for release_event")
        return slow_text

    monkeypatch.setattr(server, "generate_text", slow_generate_text)

    with make_client(monkeypatch) as (client, headers):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(
                client.post,
                endpoint,
                json=payload_factory(),
                headers=headers,
            )
            if not start_event.wait(timeout=event_timeout):
                release_event.set()
                pytest.fail("Model generation did not start in time")
            try:
                busy_response = client.post(
                    endpoint,
                    json=payload_factory(),
                    headers=headers,
                )
            finally:
                release_event.set()
            assert busy_response.status_code == 429
            assert busy_response.json() == {"detail": "Model is busy"}
            first_response = first_future.result(timeout=event_timeout)
            assert first_response.status_code == 200
            assert extract_content(first_response) == slow_text
    assert not server.API_KEYS
