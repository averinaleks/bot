import pytest
import httpx

from bot.gpt_client import (
    GPTClientError,
    GPTClientJSONError,
    GPTClientNetworkError,
    GPTClientResponseError,
    query_gpt,
    query_gpt_async,
    query_gpt_json_async,
)


class DummyResponse:
    def __init__(self, json_data=None, json_exc=None):
        self._json_data = json_data
        self._json_exc = json_exc

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


def test_query_gpt_no_env(monkeypatch):
    monkeypatch.delenv("GPT_OSS_API", raising=False)
    with pytest.raises(GPTClientNetworkError):
        query_gpt("hi")


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
async def test_query_gpt_async_no_env(monkeypatch):
    monkeypatch.delenv("GPT_OSS_API", raising=False)
    with pytest.raises(GPTClientNetworkError):
        await query_gpt_async("hi")


@pytest.mark.asyncio
async def test_query_gpt_json_async(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://example.com")
    
    class DummyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"text": '{"signal": "buy"}'}]}

    async def fake_post(self, *args, **kwargs):
        return DummyResp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await query_gpt_json_async("hi")
    assert result["signal"] == "buy"
