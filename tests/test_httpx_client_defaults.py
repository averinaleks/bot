import pytest

import ssl

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


def test_get_httpx_client_rejects_verify_false():
    with pytest.raises(ValueError):
        with get_httpx_client(verify=False):
            pass


def test_get_httpx_client_rejects_insecure_ssl_context():
    hostname_context = ssl.create_default_context()
    hostname_context.check_hostname = False

    with pytest.raises(ValueError):
        with get_httpx_client(verify=hostname_context):
            pass

    verify_context = ssl.create_default_context()
    verify_context.check_hostname = False
    verify_context.verify_mode = ssl.CERT_NONE

    with pytest.raises(ValueError):
        with get_httpx_client(verify=verify_context):
            pass


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
async def test_async_http_client_rejects_verify_false():
    with pytest.raises(ValueError):
        async with async_http_client(verify=False):
            pass


@pytest.mark.asyncio
async def test_async_http_client_rejects_insecure_ssl_context():
    hostname_context = ssl.create_default_context()
    hostname_context.check_hostname = False

    with pytest.raises(ValueError):
        async with async_http_client(verify=hostname_context):
            pass

    verify_context = ssl.create_default_context()
    verify_context.check_hostname = False
    verify_context.verify_mode = ssl.CERT_NONE

    with pytest.raises(ValueError):
        async with async_http_client(verify=verify_context):
            pass


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
async def test_get_async_http_client_rejects_verify_false():
    await close_async_http_client()
    with pytest.raises(ValueError):
        await get_async_http_client(verify=False)


@pytest.mark.asyncio
async def test_get_async_http_client_rejects_insecure_ssl_context():
    await close_async_http_client()

    hostname_context = ssl.create_default_context()
    hostname_context.check_hostname = False

    with pytest.raises(ValueError):
        await get_async_http_client(verify=hostname_context)

    verify_context = ssl.create_default_context()
    verify_context.check_hostname = False
    verify_context.verify_mode = ssl.CERT_NONE

    with pytest.raises(ValueError):
        await get_async_http_client(verify=verify_context)

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
