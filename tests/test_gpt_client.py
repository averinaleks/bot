import sys
import asyncio
import socket
import logging
import json

import pytest
import httpx

sys.modules.pop("tenacity", None)
import tenacity  # noqa: F401  # re-import after pop

from bot.gpt_client import (
    GPTClientError,
    GPTClientJSONError,
    GPTClientNetworkError,
    GPTClientResponseError,
    MAX_PROMPT_BYTES,
    MAX_RESPONSE_BYTES,
    _get_api_url_timeout,
    _validate_api_url,
    _parse_gpt_response,
    query_gpt,
    query_gpt_async,
    query_gpt_json_async,
)


class DummyStream:
    def __init__(self, content=b"content", headers=None):
        self.content = content
        self.headers = headers or {"Content-Type": "application/json"}

    def raise_for_status(self):
        pass

    def iter_bytes(self):
        yield self.content

    async def aiter_bytes(self):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


QUERIES = [
    (query_gpt, httpx.AsyncClient),
    (query_gpt_async, httpx.AsyncClient),
]


async def run_query(func, prompt):
    if asyncio.iscoroutinefunction(func):
        return await func(prompt)
    return await asyncio.to_thread(func, prompt)


def test_parse_gpt_response_success():
    content = json.dumps({"choices": [{"text": "ok"}]}).encode()
    assert _parse_gpt_response(content) == "ok"


def test_parse_gpt_response_invalid_json():
    with pytest.raises(GPTClientJSONError):
        _parse_gpt_response(b"not json")


def test_parse_gpt_response_missing_field():
    content = json.dumps({"foo": "bar"}).encode()
    with pytest.raises(GPTClientResponseError):
        _parse_gpt_response(content)

@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_network_error(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    def fake_stream(self, *args, **kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    with pytest.raises(GPTClientNetworkError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_non_json(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=b"not json")

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    with pytest.raises(GPTClientJSONError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_bad_content_type(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content, headers={"Content-Type": "text/plain"})

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    with pytest.raises(GPTClientResponseError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_missing_fields(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=json.dumps({"foo": "bar"}).encode())

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    with pytest.raises(GPTClientResponseError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_insecure_url(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "http://example.com")
    with pytest.raises(GPTClientError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
@pytest.mark.parametrize("ip", [
    "127.0.0.1",
    "10.0.0.1",
    "172.16.0.1",
    "192.168.1.1",
])
async def test_private_ip_allowed(monkeypatch, func, client_cls, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://{ip}")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    assert await run_query(func, "hi") == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_public_ip_blocked(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "http://8.8.8.8")
    with pytest.raises(GPTClientError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
@pytest.mark.parametrize("ip", ["::1", "fc00::1"])
async def test_private_ipv6_allowed(monkeypatch, func, client_cls, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://[{ip}]")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    assert await run_query(func, "hi") == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
@pytest.mark.parametrize("ip", [
    "2001:4860:4860::8888",
    "2606:4700:4700::1111",
])
async def test_public_ipv6_blocked(monkeypatch, func, client_cls, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://[{ip}]")
    with pytest.raises(GPTClientError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
@pytest.mark.parametrize("url", ["http://::1", "http://[::1%eth0]"])
async def test_invalid_ipv6(monkeypatch, func, client_cls, url):
    monkeypatch.setenv("GPT_OSS_API", url)
    with pytest.raises(GPTClientError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_invalid_url(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "bad-url")
    with pytest.raises(GPTClientError, match="Invalid GPT_OSS_API URL"):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_invalid_url_no_host(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://")
    with pytest.raises(GPTClientError, match="Invalid GPT_OSS_API URL"):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_no_env(monkeypatch, func, client_cls):
    monkeypatch.delenv("GPT_OSS_API", raising=False)
    with pytest.raises(GPTClientNetworkError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_retry_success(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    calls = {"count": 0}

    def fake_stream(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPError("boom")
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    sleep_path = "asyncio.sleep" if asyncio.iscoroutinefunction(func) else "time.sleep"
    if asyncio.iscoroutinefunction(func):
        async def no_sleep(*args, **kwargs):  # pragma: no cover - no sleep
            pass
        monkeypatch.setattr(sleep_path, no_sleep)
    else:
        monkeypatch.setattr(sleep_path, lambda *a, **k: None)
    assert await run_query(func, "hi") == "ok"
    assert calls["count"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_retry_failure(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    calls = {"count": 0}

    def fake_stream(self, *args, **kwargs):
        calls["count"] += 1
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    sleep_path = "asyncio.sleep" if asyncio.iscoroutinefunction(func) else "time.sleep"
    if asyncio.iscoroutinefunction(func):
        async def no_sleep(*args, **kwargs):  # pragma: no cover - no sleep
            pass
        monkeypatch.setattr(sleep_path, no_sleep)
    else:
        monkeypatch.setattr(sleep_path, lambda *a, **k: None)
    with pytest.raises(GPTClientNetworkError):
        await run_query(func, "hi")
    assert calls["count"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("func, _", QUERIES)
async def test_prompt_too_long(func, _):
    with pytest.raises(GPTClientError):
        await run_query(func, "я" * (MAX_PROMPT_BYTES // 2 + 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_response_too_long(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=b"x" * (MAX_RESPONSE_BYTES + 1))

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    with pytest.raises(GPTClientError):
        await run_query(func, "hi")


def test_validate_api_url_multiple_dns_results_public_blocked(monkeypatch):
    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        assert host == "foo.local"
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(GPTClientError):
        _validate_api_url("http://foo.local")


def test_query_gpt_private_fqdn_allowed(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://foo.local")
    called = {"value": False}

    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        called["value"] = True
        assert host == "foo.local"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0))]

    async def fake_async_getaddrinfo(
        self, host, port, family=0, type=0, proto=0, flags=0
    ):
        return fake_getaddrinfo(host, port, family, type, proto, flags)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(asyncio.AbstractEventLoop, "getaddrinfo", fake_async_getaddrinfo)

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    assert query_gpt("hi") == "ok"
    assert called["value"]


def test_query_gpt_dns_error(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://foo.local")
    with pytest.raises(GPTClientError):
        query_gpt("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_context(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    assert query_gpt("hi") == "ok"


def test_get_api_url_timeout_non_positive(monkeypatch, caplog):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    monkeypatch.setenv("GPT_OSS_TIMEOUT", "0")
    with caplog.at_level(logging.WARNING):
        _, timeout, _, _ = _get_api_url_timeout()
    assert timeout == 5.0
    assert "Non-positive GPT_OSS_TIMEOUT value" in caplog.text


def test_get_api_url_timeout_invalid(monkeypatch, caplog):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    monkeypatch.setenv("GPT_OSS_TIMEOUT", "abc")
    with caplog.at_level(logging.WARNING):
        _, timeout, _, _ = _get_api_url_timeout()
    assert timeout == 5.0
    assert "Invalid GPT_OSS_TIMEOUT value 'abc'; defaulting to 5.0" in caplog.text


@pytest.mark.asyncio
async def test_query_gpt_json_async(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": '{"signal": "buy"}'}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    result = await query_gpt_json_async("hi")
    assert result["signal"] == "buy"
