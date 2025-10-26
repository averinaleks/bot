import pytest

import httpx

from http_client import (
    DEFAULT_TIMEOUT,
    async_http_client,
    close_async_http_client,
    get_async_http_client,
    get_httpx_client,
)


def test_get_httpx_client_trust_env_false():
    with get_httpx_client() as client:
        assert client.trust_env is False


def test_get_httpx_client_enforces_timeout_floor():
    with get_httpx_client(timeout=0) as client:
        assert isinstance(client.timeout, httpx.Timeout)
        assert client.timeout.connect == pytest.approx(DEFAULT_TIMEOUT)
        assert client.timeout.read == pytest.approx(DEFAULT_TIMEOUT)


def test_get_httpx_client_respects_timeout_object():
    custom = httpx.Timeout(1.5)
    with get_httpx_client(timeout=custom) as client:
        assert isinstance(client.timeout, httpx.Timeout)
        assert client.timeout.connect == pytest.approx(custom.connect)
        assert client.timeout.read == pytest.approx(custom.read)


@pytest.mark.asyncio
async def test_async_http_client_normalises_timeout_argument():
    async with async_http_client(timeout=0) as client:
        assert isinstance(client.timeout, httpx.Timeout)
        assert client.timeout.connect == pytest.approx(DEFAULT_TIMEOUT)


@pytest.mark.asyncio
async def test_async_http_client_preserves_timeout_object():
    custom = httpx.Timeout(2.5)
    async with async_http_client(timeout=custom) as client:
        assert isinstance(client.timeout, httpx.Timeout)
        assert client.timeout.connect == pytest.approx(custom.connect)


@pytest.mark.asyncio
async def test_get_async_http_client_normalises_timeout_argument():
    await close_async_http_client()
    client = await get_async_http_client(timeout=0)
    try:
        assert isinstance(client.timeout, httpx.Timeout)
        assert client.timeout.connect == pytest.approx(DEFAULT_TIMEOUT)
    finally:
        await close_async_http_client()


@pytest.mark.asyncio
async def test_get_async_http_client_respects_timeout_object():
    await close_async_http_client()
    custom = httpx.Timeout(3.1)
    client = await get_async_http_client(timeout=custom)
    try:
        assert isinstance(client.timeout, httpx.Timeout)
        assert client.timeout.connect == pytest.approx(custom.connect)
    finally:
        await close_async_http_client()
