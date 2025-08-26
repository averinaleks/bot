import sys
import asyncio
import socket
import logging
import json

import pytest
import httpx

sys.modules.pop("tenacity", None)
import tenacity

from bot.gpt_client import (
    GPTClientError,
    GPTClientJSONError,
    GPTClientNetworkError,
    GPTClientResponseError,
    MAX_PROMPT_BYTES,
    MAX_RESPONSE_BYTES,
    _get_api_url_timeout,
    _validate_api_url,
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyAStream:
    def __init__(self, content=b"content", headers=None):
        self.content = content
        self.headers = headers or {"Content-Type": "application/json"}

    def raise_for_status(self):
        pass

    async def aiter_bytes(self):
        yield self.content

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


def test_query_gpt_network_error(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    def fake_stream(self, *args, **kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    with pytest.raises(GPTClientNetworkError):
        query_gpt("hi")


def test_query_gpt_non_json(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=b"not json")

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    with pytest.raises(GPTClientJSONError):
        query_gpt("hi")


def test_query_gpt_bad_content_type(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content, headers={"Content-Type": "text/plain"})

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    with pytest.raises(GPTClientResponseError):
        query_gpt("hi")


def test_query_gpt_missing_fields(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=json.dumps({"foo": "bar"}).encode())

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    with pytest.raises(GPTClientResponseError):
        query_gpt("hi")


def test_query_gpt_insecure_url(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://example.com")
    with pytest.raises(GPTClientError):
        query_gpt("hi")


def test_query_gpt_uppercase_scheme(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "HTTPS://example.com")

    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=json.dumps({"choices": [{"text": "ok"}]}).encode())

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    assert query_gpt("hi") == "ok"


def test_query_gpt_prompt_too_long():
    with pytest.raises(GPTClientError):
        query_gpt("я" * (MAX_PROMPT_BYTES // 2 + 1))


def test_query_gpt_response_too_long(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=b"x" * (MAX_RESPONSE_BYTES + 1))

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    with pytest.raises(GPTClientError):
        query_gpt("hi")


@pytest.mark.parametrize("ip", [
    "127.0.0.1",
    "10.0.0.1",
    "172.16.0.1",
    "192.168.1.1",
])
def test_query_gpt_private_ip_allowed(monkeypatch, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://{ip}")

    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=json.dumps({"choices": [{"text": "ok"}]}).encode())

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    assert query_gpt("hi") == "ok"


def test_query_gpt_public_ip_blocked(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://8.8.8.8")
    with pytest.raises(GPTClientError):
        query_gpt("hi")


@pytest.mark.parametrize("ip", [
    "::1",
    "fc00::1",
])
def test_query_gpt_private_ipv6_allowed(monkeypatch, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://[{ip}]")

    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=json.dumps({"choices": [{"text": "ok"}]}).encode())

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    assert query_gpt("hi") == "ok"


@pytest.mark.parametrize(
    "ip",
    [
        "2001:4860:4860::8888",
        "2606:4700:4700::1111",
    ],
)
def test_query_gpt_public_ipv6_blocked(monkeypatch, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://[{ip}]")
    with pytest.raises(GPTClientError):
        query_gpt("hi")


@pytest.mark.parametrize("url", [
    "http://::1",
    "http://[::1%eth0]",
])
def test_query_gpt_invalid_ipv6(monkeypatch, url):
    monkeypatch.setenv("GPT_OSS_API", url)
    with pytest.raises(GPTClientError):
        query_gpt("hi")


def test_query_gpt_private_fqdn_allowed(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://foo.local")

    called = {"value": False}

    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        called["value"] = True
        assert host == "foo.local"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=json.dumps({"choices": [{"text": "ok"}]}).encode())

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    assert query_gpt("hi") == "ok"
    assert called["value"]


def test_query_gpt_dns_error(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://foo.local")


    with pytest.raises(GPTClientError):
        query_gpt("hi")


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


def test_query_gpt_invalid_url(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "bad-url")
    with pytest.raises(GPTClientError, match="Invalid GPT_OSS_API URL"):
        query_gpt("hi")


def test_query_gpt_invalid_url_no_host(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://")
    with pytest.raises(GPTClientError, match="Invalid GPT_OSS_API URL"):
        query_gpt("hi")


def test_query_gpt_no_env(monkeypatch):
    monkeypatch.delenv("GPT_OSS_API", raising=False)
    with pytest.raises(GPTClientNetworkError):
        query_gpt("hi")


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


def test_query_gpt_retry_success(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    calls = {"count": 0}

    def fake_stream(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPError("boom")
        return DummyStream(content=json.dumps({"choices": [{"text": "ok"}]}).encode())

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    assert query_gpt("hi") == "ok"
    assert calls["count"] == 2


def test_query_gpt_retry_failure(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    calls = {"count": 0}

    def fake_stream(self, *args, **kwargs):
        calls["count"] += 1
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(GPTClientNetworkError):
        query_gpt("hi")
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_query_gpt_async_network_error(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    def fake_stream(self, *args, **kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    with pytest.raises(GPTClientNetworkError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_non_json(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    def fake_stream(self, *args, **kwargs):
        return DummyAStream(content=b"not json")

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    with pytest.raises(GPTClientJSONError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_bad_content_type(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyAStream(content=content, headers={"Content-Type": "text/plain"})

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    with pytest.raises(GPTClientResponseError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_missing_fields(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    def fake_stream(self, *args, **kwargs):
        return DummyAStream(content=json.dumps({"foo": "bar"}).encode())

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    with pytest.raises(GPTClientResponseError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_insecure_url(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://example.com")
    with pytest.raises(GPTClientError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("ip", [
    "127.0.0.1",
    "10.0.0.1",
    "172.16.0.1",
    "192.168.1.1",
])
async def test_query_gpt_async_private_ip_allowed(monkeypatch, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://{ip}")

    def fake_stream(self, *args, **kwargs):
        return DummyAStream(content=json.dumps({"choices": [{"text": "ok"}]}).encode())

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    assert await query_gpt_async("hi") == "ok"


@pytest.mark.asyncio
async def test_query_gpt_async_public_ip_blocked(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://8.8.8.8")
    with pytest.raises(GPTClientError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("ip", [
    "::1",
    "fc00::1",
])
async def test_query_gpt_async_private_ipv6_allowed(monkeypatch, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://[{ip}]")

    def fake_stream(self, *args, **kwargs):
        return DummyAStream(content=json.dumps({"choices": [{"text": "ok"}]}).encode())

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    assert await query_gpt_async("hi") == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ip",
    [
        "2001:4860:4860::8888",
        "2606:4700:4700::1111",
    ],
)
async def test_query_gpt_async_public_ipv6_blocked(monkeypatch, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://[{ip}]")
    with pytest.raises(GPTClientError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "http://::1",
    "http://[::1%eth0]",
])
async def test_query_gpt_async_invalid_ipv6(monkeypatch, url):
    monkeypatch.setenv("GPT_OSS_API", url)
    with pytest.raises(GPTClientError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_invalid_url(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "bad-url")
    with pytest.raises(GPTClientError, match="Invalid GPT_OSS_API URL"):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_invalid_url_no_host(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://")
    with pytest.raises(GPTClientError, match="Invalid GPT_OSS_API URL"):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_no_env(monkeypatch):
    monkeypatch.delenv("GPT_OSS_API", raising=False)
    with pytest.raises(GPTClientNetworkError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_retry_success(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    calls = {"count": 0}
    def fake_stream(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPError("boom")
        return DummyAStream(content=json.dumps({"choices": [{"text": "ok"}]}).encode())

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    async def no_sleep(*args, **kwargs):
        pass

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    assert await query_gpt_async("hi") == "ok"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_query_gpt_async_retry_failure(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    calls = {"count": 0}

    def fake_stream(self, *args, **kwargs):
        calls["count"] += 1
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    async def no_sleep(*args, **kwargs):
        pass

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    with pytest.raises(GPTClientNetworkError):
        await query_gpt_async("hi")
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_query_gpt_async_prompt_too_long():
    with pytest.raises(GPTClientError):
        await query_gpt_async("я" * (MAX_PROMPT_BYTES // 2 + 1))


@pytest.mark.asyncio
async def test_query_gpt_async_response_too_long(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    def fake_stream(self, *args, **kwargs):
        return DummyAStream(content=b"x" * (MAX_RESPONSE_BYTES + 1))

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    with pytest.raises(GPTClientError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_json_async(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": '{"signal": "buy"}' }]}).encode()
        return DummyAStream(content=content)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    result = await query_gpt_json_async("hi")
    assert result["signal"] == "buy"
