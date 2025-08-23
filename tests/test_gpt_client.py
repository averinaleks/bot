import sys
import asyncio
import socket

import pytest
import httpx

sys.modules.pop("tenacity", None)
import tenacity

from bot.gpt_client import (
    GPTClientError,
    GPTClientJSONError,
    GPTClientNetworkError,
    GPTClientResponseError,
    MAX_PROMPT_LEN,
    MAX_RESPONSE_LEN,
    query_gpt,
    query_gpt_async,
    query_gpt_json_async,
)


class DummyResponse:
    def __init__(self, json_data=None, json_exc=None, content=b"content"):
        self._json_data = json_data
        self._json_exc = json_exc
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        if self._json_exc:
            raise self._json_exc
        return self._json_data


def test_query_gpt_network_error(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    def fake_post(self, *args, **kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    with pytest.raises(GPTClientNetworkError):
        query_gpt("hi")


def test_query_gpt_non_json(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    def fake_post(self, *args, **kwargs):
        return DummyResponse(json_exc=ValueError("no json"))

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    with pytest.raises(GPTClientJSONError):
        query_gpt("hi")


def test_query_gpt_missing_fields(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    def fake_post(self, *args, **kwargs):
        return DummyResponse(json_data={"foo": "bar"})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    with pytest.raises(GPTClientResponseError):
        query_gpt("hi")


def test_query_gpt_insecure_url(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://example.com")
    with pytest.raises(GPTClientError):
        query_gpt("hi")


def test_query_gpt_uppercase_scheme(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "HTTPS://example.com")

    def fake_post(self, *args, **kwargs):
        return DummyResponse(json_data={"choices": [{"text": "ok"}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    assert query_gpt("hi") == "ok"


def test_query_gpt_prompt_too_long():
    with pytest.raises(GPTClientError):
        query_gpt("x" * (MAX_PROMPT_LEN + 1))


def test_query_gpt_response_too_long(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    def fake_post(self, *args, **kwargs):
        return DummyResponse(
            json_data={"choices": [{"text": "ok"}]},
            content=b"x" * (MAX_RESPONSE_LEN + 1),
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
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

    def fake_post(self, *args, **kwargs):
        return DummyResponse(json_data={"choices": [{"text": "ok"}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    assert query_gpt("hi") == "ok"


def test_query_gpt_public_ip_blocked(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://8.8.8.8")
    with pytest.raises(GPTClientError):
        query_gpt("hi")


def test_query_gpt_private_fqdn_allowed(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://foo.local")

    def fake_gethostbyname(host):
        assert host == "foo.local"
        return "10.0.0.1"

    monkeypatch.setattr(socket, "gethostbyname", fake_gethostbyname)

    def fake_post(self, *args, **kwargs):
        return DummyResponse(json_data={"choices": [{"text": "ok"}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    assert query_gpt("hi") == "ok"


def test_query_gpt_dns_error(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://foo.local")

    def fake_gethostbyname(host):
        raise socket.gaierror("boom")

    monkeypatch.setattr(socket, "gethostbyname", fake_gethostbyname)

    with pytest.raises(GPTClientError):
        query_gpt("hi")


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


def test_query_gpt_retry_success(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    calls = {"count": 0}

    def fake_post(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPError("boom")
        return DummyResponse(json_data={"choices": [{"text": "ok"}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    assert query_gpt("hi") == "ok"
    assert calls["count"] == 2


def test_query_gpt_retry_failure(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    calls = {"count": 0}

    def fake_post(self, *args, **kwargs):
        calls["count"] += 1
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(GPTClientNetworkError):
        query_gpt("hi")
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_query_gpt_async_network_error(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    async def fake_post(self, *args, **kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(GPTClientNetworkError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_non_json(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    class DummyResp:
        content = b"content"
        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("no json")

    async def fake_post(self, *args, **kwargs):
        return DummyResp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(GPTClientJSONError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_missing_fields(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    class DummyResp:
        content = b"content"
        def raise_for_status(self):
            pass

        def json(self):
            return {"foo": "bar"}

    async def fake_post(self, *args, **kwargs):
        return DummyResp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
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

    class DummyResp:
        content = b"content"
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"text": "ok"}]}

    async def fake_post(self, *args, **kwargs):
        return DummyResp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    assert await query_gpt_async("hi") == "ok"


@pytest.mark.asyncio
async def test_query_gpt_async_public_ip_blocked(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://8.8.8.8")
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

    class DummyResp:
        content = b"content"
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"text": "ok"}]}

    async def fake_post(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPError("boom")
        return DummyResp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    async def no_sleep(*args, **kwargs):
        pass

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    assert await query_gpt_async("hi") == "ok"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_query_gpt_async_retry_failure(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    calls = {"count": 0}

    async def fake_post(self, *args, **kwargs):
        calls["count"] += 1
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    async def no_sleep(*args, **kwargs):
        pass

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    with pytest.raises(GPTClientNetworkError):
        await query_gpt_async("hi")
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_query_gpt_async_prompt_too_long():
    with pytest.raises(GPTClientError):
        await query_gpt_async("x" * (MAX_PROMPT_LEN + 1))


@pytest.mark.asyncio
async def test_query_gpt_async_response_too_long(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    class DummyResp:
        content = b"x" * (MAX_RESPONSE_LEN + 1)
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"text": "ok"}]}

    async def fake_post(self, *args, **kwargs):
        return DummyResp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(GPTClientError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_json_async(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")

    class DummyResp:
        content = b"content"
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"text": '{"signal": "buy"}'}]}

    async def fake_post(self, *args, **kwargs):
        return DummyResp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await query_gpt_json_async("hi")
    assert result["signal"] == "buy"
