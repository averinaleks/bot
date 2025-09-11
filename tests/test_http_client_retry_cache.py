import asyncio
import types
import random

import httpx
import pytest

import http_client


@pytest.mark.asyncio
async def test_request_with_retry_metrics(monkeypatch):
    # deterministic jitter
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)
    calls = {"n": 0}

    async def _request(method, url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, request=httpx.Request(method, url))
        return httpx.Response(200, request=httpx.Request(method, url))

    client = types.SimpleNamespace(request=_request)
    http_client.RETRY_METRICS.clear()

    resp = await http_client.request_with_retry(
        "GET", "http://example", client=client, max_attempts=2, backoff_base=0
    )
    assert resp.status_code == 200
    assert http_client.RETRY_METRICS["http://example"] == 1


@pytest.mark.asyncio
async def test_request_with_retry_cache(monkeypatch):
    calls = {"n": 0}

    async def _request(method, url, **kwargs):
        calls["n"] += 1
        return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, url))

    client = types.SimpleNamespace(request=_request)
    http_client.REFERENCE_CACHE.clear()

    resp1 = await http_client.request_with_retry(
        "GET", "http://test/symbols", client=client
    )
    resp2 = await http_client.request_with_retry(
        "GET", "http://test/symbols", client=client
    )
    assert calls["n"] == 1
    assert resp1.text == resp2.text
