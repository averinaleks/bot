import asyncio
import importlib
import sys

import pytest


@pytest.mark.asyncio
async def test_trading_bot_offline_uses_stubs(monkeypatch):
    modules = ["bot.trading_bot", "trading_bot", "bot.config", "config"]
    originals = {name: sys.modules.get(name) for name in modules}
    for name in modules:
        sys.modules.pop(name, None)
    monkeypatch.setenv("OFFLINE_MODE", "1")

    try:
        trading_bot = importlib.import_module("bot.trading_bot")
        assert getattr(trading_bot.httpx, "__offline_stub__", False)
        assert getattr(trading_bot.GPTAdviceModel, "__offline_stub__", False)

        trading_bot.HTTP_CLIENT = None
        trading_bot.HTTP_CLIENT_LOCK = asyncio.Lock()

        client = await trading_bot.get_http_client()
        resp = await client.post("https://example.org/api", json={"ping": "pong"})
        assert resp.status_code == 200
        assert resp.json() == {"ping": "pong"}

        await trading_bot.close_http_client()
    finally:
        for name in modules:
            sys.modules.pop(name, None)
        monkeypatch.setenv("OFFLINE_MODE", "0")
        for name, module in originals.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)


@pytest.mark.asyncio
async def test_http_client_offline_stub(monkeypatch):
    modules = ["http_client", "bot.config", "config"]
    originals = {name: sys.modules.get(name) for name in modules}
    for name in modules:
        sys.modules.pop(name, None)
    monkeypatch.setenv("OFFLINE_MODE", "1")

    try:
        http_client = importlib.import_module("http_client")
        assert getattr(http_client.httpx, "__offline_stub__", False)

        http_client._ASYNC_CLIENT = None
        http_client._ASYNC_CLIENT_LOCK = asyncio.Lock()

        client = await http_client.get_async_http_client()
        resp = await client.get("https://example.org/data")
        assert resp.status_code == 200

        await http_client.close_async_http_client()
    finally:
        for name in modules:
            sys.modules.pop(name, None)
        monkeypatch.setenv("OFFLINE_MODE", "0")
        for name, module in originals.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)
