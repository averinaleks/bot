import asyncio
import pytest
from bot import trading_bot


def test_safe_int_invalid(monkeypatch, caplog):
    monkeypatch.setenv("X_INT", "bad")
    with caplog.at_level("WARNING"):
        assert trading_bot.safe_int("X_INT", 7) == 7
    assert "Invalid X_INT" in caplog.text


def test_safe_float_invalid(monkeypatch, caplog):
    monkeypatch.setenv("X_FLOAT", "bad")
    with caplog.at_level("WARNING"):
        assert trading_bot.safe_float("X_FLOAT", 3.5) == 3.5
    assert "Invalid X_FLOAT" in caplog.text


def test_send_trade_timeout_invalid_env(monkeypatch):
    called = {}

    def fake_post(self, url, json=None, timeout=None, headers=None):
        called["timeout"] = timeout
        class Resp:
            status_code = 200
            def json(self):
                return {"status": "ok"}
        return Resp()

    monkeypatch.setattr(trading_bot.httpx.Client, "post", fake_post)
    monkeypatch.setenv("TRADE_MANAGER_TIMEOUT", "oops")
    trading_bot.send_trade("BTCUSDT", "buy", 1.0, {"trade_manager_url": "http://tm"})
    assert called["timeout"] == 5.0


def test_check_services_invalid_env(monkeypatch):
    env = {
        "data_handler_url": "http://dh",
        "model_builder_url": "http://mb",
        "trade_manager_url": "http://tm",
    }
    monkeypatch.setattr(trading_bot, "_load_env", lambda: env)
    monkeypatch.setattr(trading_bot, "DEFAULT_SERVICE_CHECK_RETRIES", 1)
    monkeypatch.setattr(trading_bot, "DEFAULT_SERVICE_CHECK_DELAY", 0.0)
    monkeypatch.setenv("SERVICE_CHECK_RETRIES", "bad")
    monkeypatch.setenv("SERVICE_CHECK_DELAY", "bad")
    calls = {"count": 0}

    class DummyClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def get(self, url, timeout=None):
            calls["count"] += 1
            raise trading_bot.httpx.HTTPError("boom")

    monkeypatch.setattr(trading_bot.httpx, "AsyncClient", lambda *a, **k: DummyClient(), raising=False)
    with pytest.raises(SystemExit):
        asyncio.run(trading_bot.check_services())
    assert calls["count"] == 3
