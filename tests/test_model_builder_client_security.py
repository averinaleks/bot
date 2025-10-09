import socket

import asyncio

import pytest

import model_builder_client


def _fake_resolve(hostname: str, addresses: set[str]):
    def _resolver(_hostname: str) -> set[str]:
        assert _hostname == hostname
        return addresses

    return _resolver


def test_prepare_endpoint_rejects_credentials(monkeypatch):
    monkeypatch.setattr(
        model_builder_client,
        "_resolve_hostname",
        _fake_resolve("localhost", {"127.0.0.1"}),
    )

    endpoint = model_builder_client._prepare_endpoint(  # type: ignore[attr-defined]
        "http://user:pass@localhost:8000",
        purpose="тест",
    )

    assert endpoint is None


def test_prepare_endpoint_allows_default_service_hosts(monkeypatch):
    monkeypatch.setattr(
        model_builder_client,
        "_resolve_hostname",
        _fake_resolve("model_builder", {"127.0.0.1"}),
    )

    endpoint = model_builder_client._prepare_endpoint(  # type: ignore[attr-defined]
        "http://model_builder:8001",
        purpose="тест",
    )

    assert endpoint is not None
    assert endpoint.hostname == "model_builder"


def test_prepare_endpoint_respects_allowlist_env(monkeypatch):
    monkeypatch.setenv("MODEL_BUILDER_ALLOWED_HOSTS", "trusted.local")
    def fake_getaddrinfo(host, *_args, **_kwargs):
        if host == "trusted.local":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", 0))]
        raise AssertionError(host)

    monkeypatch.setattr(model_builder_client.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(
        model_builder_client,
        "_resolve_hostname",
        _fake_resolve("trusted.local", {"10.0.0.8"}),
    )

    endpoint = model_builder_client._prepare_endpoint(  # type: ignore[attr-defined]
        "https://trusted.local:8443",
        purpose="тест",
    )

    assert endpoint is not None
    assert endpoint.hostname == "trusted.local"


def test_load_allowed_hosts_filters_public(monkeypatch, caplog):
    monkeypatch.setenv("MODEL_BUILDER_ALLOWED_HOSTS", "trusted.local")

    def fake_getaddrinfo(host, *_args, **_kwargs):
        if host == "trusted.local":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]
        raise AssertionError(host)

    monkeypatch.setattr(model_builder_client.socket, "getaddrinfo", fake_getaddrinfo)

    with caplog.at_level("WARNING"):
        hosts = model_builder_client._load_allowed_hosts()

    assert "trusted.local" not in hosts
    assert "Пропускаем" in caplog.text


def test_prepare_endpoint_rejects_unlisted_host(monkeypatch):
    monkeypatch.delenv("MODEL_BUILDER_ALLOWED_HOSTS", raising=False)
    monkeypatch.setattr(
        model_builder_client,
        "_resolve_hostname",
        _fake_resolve("example.com", {"203.0.113.10"}),
    )

    endpoint = model_builder_client._prepare_endpoint(  # type: ignore[attr-defined]
        "https://example.com/api",
        purpose="тест",
    )

    assert endpoint is None


class _DummyResponse:
    def __init__(self, *, headers=None, content=b"{}", parsed=None, json_exc=None):
        self.status_code = 200
        self.headers = headers or {}
        self._content = content
        self._parsed = parsed if parsed is not None else {"ohlcv": []}
        self._json_exc = json_exc

    @property
    def content(self):
        return self._content

    def json(self):
        if self._json_exc:
            raise self._json_exc
        return self._parsed


class _DummyAsyncClient:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, *_args, **_kwargs):
        return self._response


def test_fetch_training_data_rejects_unexpected_content_type(monkeypatch, caplog):
    endpoint = model_builder_client.ServiceEndpoint(
        scheme="http",
        base_url="http://model_builder:8001",
        hostname="model_builder",
        allowed_ips=frozenset({"127.0.0.1"}),
    )

    async def _allow(_endpoint):
        return True

    monkeypatch.setattr(model_builder_client, "_hostname_still_allowed", _allow)
    response = _DummyResponse(
        headers={"Content-Type": "text/html"},
        content=b"<html></html>",
        json_exc=AssertionError("json should not be called"),
    )
    monkeypatch.setattr(
        model_builder_client.httpx,
        "AsyncClient",
        lambda *a, **k: _DummyAsyncClient(response),
    )

    async def _run():
        return await model_builder_client._fetch_training_data_from_endpoint(  # type: ignore[attr-defined]
            endpoint,
            "BTCUSDT",
        )

    with caplog.at_level("ERROR"):
        features, labels = asyncio.run(_run())

    assert features == []
    assert labels == []
    assert "Content-Type" in caplog.text


def test_fetch_training_data_limits_payload_size(monkeypatch, caplog):
    endpoint = model_builder_client.ServiceEndpoint(
        scheme="http",
        base_url="http://model_builder:8001",
        hostname="model_builder",
        allowed_ips=frozenset({"127.0.0.1"}),
    )

    async def _allow(_endpoint):
        return True

    monkeypatch.setattr(model_builder_client, "_hostname_still_allowed", _allow)

    oversized = b"{" + b"0" * (model_builder_client._MAX_TRAINING_PAYLOAD_BYTES + 1) + b"}"
    response = _DummyResponse(
        headers={"Content-Type": "application/json"},
        content=oversized,
        json_exc=AssertionError("json should not be called"),
    )
    monkeypatch.setattr(
        model_builder_client.httpx,
        "AsyncClient",
        lambda *a, **k: _DummyAsyncClient(response),
    )

    async def _run():
        return await model_builder_client._fetch_training_data_from_endpoint(  # type: ignore[attr-defined]
            endpoint,
            "BTCUSDT",
        )

    with caplog.at_level("ERROR"):
        features, labels = asyncio.run(_run())

    assert features == []
    assert labels == []
    assert "exceeded" in caplog.text
