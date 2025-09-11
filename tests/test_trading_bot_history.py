import asyncio

import pytest

import trading_bot


class DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, timeout=5):
        class Resp:
            status_code = 200

            def json(self):
                return {"history": [[0, 0, 0, 0, 10, 0], [0, 0, 0, 0, 11, 0]]}

        return Resp()


@pytest.mark.asyncio
async def test_fetch_initial_history(monkeypatch):
    monkeypatch.setattr(trading_bot, "httpx", type("X", (), {"AsyncClient": DummyClient}))
    env = {"data_handler_url": "http://test"}
    trading_bot._PRICE_HISTORY.clear()
    await trading_bot.fetch_initial_history("SYM", env)
    assert list(trading_bot._PRICE_HISTORY) == [10.0, 11.0]
