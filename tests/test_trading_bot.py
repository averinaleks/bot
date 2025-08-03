from bot import trading_bot
import time
import types
import pytest


def test_send_trade_timeout_env(monkeypatch):
    called = {}

    def fake_post(url, json=None, timeout=None):
        called['timeout'] = timeout
        class Resp:
            status_code = 200
        return Resp()

    monkeypatch.setattr(trading_bot.requests, 'post', fake_post)
    monkeypatch.setenv('TRADE_MANAGER_TIMEOUT', '9')
    trading_bot.send_trade('BTCUSDT', 'buy', 100.0, {'trade_manager_url': 'http://tm'})
    assert called['timeout'] == 9.0


def test_load_env_uses_host(monkeypatch):
    monkeypatch.delenv('DATA_HANDLER_URL', raising=False)
    monkeypatch.delenv('MODEL_BUILDER_URL', raising=False)
    monkeypatch.delenv('TRADE_MANAGER_URL', raising=False)
    monkeypatch.setenv('HOST', 'localhost')
    env = trading_bot._load_env()
    assert env['data_handler_url'] == 'http://localhost:8000'
    assert env['model_builder_url'] == 'http://localhost:8001'
    assert env['trade_manager_url'] == 'http://localhost:8002'


def test_load_env_explicit_urls(monkeypatch):
    monkeypatch.setenv('DATA_HANDLER_URL', 'http://127.0.0.1:9000')
    monkeypatch.setenv('MODEL_BUILDER_URL', 'http://127.0.0.1:9001')
    monkeypatch.setenv('TRADE_MANAGER_URL', 'http://127.0.0.1:9002')
    monkeypatch.setenv('HOST', 'should_not_use')
    env = trading_bot._load_env()
    assert env['data_handler_url'] == 'http://127.0.0.1:9000'
    assert env['model_builder_url'] == 'http://127.0.0.1:9001'
    assert env['trade_manager_url'] == 'http://127.0.0.1:9002'




def test_load_env_uses_host_when_missing(monkeypatch):
    """Fallback to HOST when service URLs are absent."""
    for var in ('DATA_HANDLER_URL', 'MODEL_BUILDER_URL', 'TRADE_MANAGER_URL'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('HOST', '127.0.0.1')
    env = trading_bot._load_env()
    assert env['data_handler_url'] == 'http://127.0.0.1:8000'
    assert env['model_builder_url'] == 'http://127.0.0.1:8001'
    assert env['trade_manager_url'] == 'http://127.0.0.1:8002'


def test_send_trade_latency_alert(monkeypatch, fast_sleep):
    called = []

    def fake_post(url, json=None, timeout=None):
        time.sleep(0.01)
        class Resp:
            status_code = 200
        return Resp()

    monkeypatch.setattr(trading_bot.requests, 'post', fake_post)
    monkeypatch.setattr(trading_bot, 'send_telegram_alert', lambda msg: called.append(msg))
    trading_bot.CONFIRMATION_TIMEOUT = 0.0
    trading_bot.send_trade('BTCUSDT', 'buy', 1.0, {'trade_manager_url': 'http://tm'})
    assert called


def test_send_trade_http_error_alert(monkeypatch):
    called = []

    def fake_post(url, json=None, timeout=None):
        class Resp:
            status_code = 500

        return Resp()

    monkeypatch.setattr(trading_bot.requests, 'post', fake_post)
    monkeypatch.setattr(trading_bot, 'send_telegram_alert', lambda msg: called.append(msg))
    trading_bot.send_trade('BTCUSDT', 'sell', 1.0, {'trade_manager_url': 'http://tm'})
    assert called


def test_send_trade_exception_alert(monkeypatch):
    called = []

    def fake_post(url, json=None, timeout=None):
        raise trading_bot.requests.RequestException('boom')

    monkeypatch.setattr(trading_bot.requests, 'post', fake_post)
    monkeypatch.setattr(trading_bot, 'send_telegram_alert', lambda msg: called.append(msg))
    trading_bot.send_trade('BTCUSDT', 'sell', 1.0, {'trade_manager_url': 'http://tm'})
    assert called


def test_send_trade_forwards_params(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        class Resp:
            status_code = 200
        return Resp()

    monkeypatch.setattr(trading_bot.requests, 'post', fake_post)
    trading_bot.send_trade(
        'BTCUSDT',
        'buy',
        1.0,
        {'trade_manager_url': 'http://tm'},
        tp=10,
        sl=5,
        trailing_stop=2,
    )
    assert captured['tp'] == 10
    assert captured['sl'] == 5
    assert captured['trailing_stop'] == 2


@pytest.mark.asyncio
async def test_reactive_trade_latency_alert(monkeypatch, fast_sleep):
    called = []

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, url, timeout=None):
            return types.SimpleNamespace(status_code=200, json=lambda: {"price": 1.0})

        async def post(self, url, json=None, timeout=None):
            if url.endswith("/predict"):
                return types.SimpleNamespace(
                    status_code=200,
                    json=lambda: {"signal": "buy", "tp": 10, "sl": 5, "trailing_stop": 1},
                )
            time.sleep(0.01)
            called.append(json)
            return types.SimpleNamespace(status_code=200, json=lambda: {})

    monkeypatch.setattr(trading_bot.httpx, "AsyncClient", lambda *a, **k: DummyClient(), raising=False)
    monkeypatch.setattr(trading_bot, "_load_env", lambda: {
        "data_handler_url": "http://dh",
        "model_builder_url": "http://mb",
        "trade_manager_url": "http://tm",
    })
    monkeypatch.setattr(trading_bot, "send_telegram_alert", lambda msg: called.append(msg))
    trading_bot.CONFIRMATION_TIMEOUT = 0.0
    await trading_bot.reactive_trade("BTCUSDT")
    assert called and isinstance(called[0], dict)
    assert called[0]["tp"] == 10
    assert called[0]["sl"] == 5
    assert called[0]["trailing_stop"] == 1


def test_run_once_invalid_price(monkeypatch):
    """No trade sent when fetch_price returns a non-positive value."""
    sent = []

    monkeypatch.setattr(trading_bot, "fetch_price", lambda *a, **k: 0.0)
    monkeypatch.setattr(trading_bot, "send_trade", lambda *a, **k: sent.append(True))
    monkeypatch.setattr(trading_bot, "_load_env", lambda: {
        "data_handler_url": "http://dh",
        "model_builder_url": "http://mb",
        "trade_manager_url": "http://tm",
    })

    trading_bot.run_once()
    assert not sent


def test_fetch_price_error(monkeypatch):
    def fake_get(url, timeout=None):
        class Resp:
            status_code = 503
            def json(self):
                return {"error": "down"}
        return Resp()

    monkeypatch.setattr(trading_bot.requests, "get", fake_get)
    price = trading_bot.fetch_price("BTCUSDT", {"data_handler_url": "http://dh"})
    assert price is None


def test_run_once_price_error(monkeypatch):
    called = {"pred": False}

    monkeypatch.setattr(trading_bot, "fetch_price", lambda *a, **k: None)
    monkeypatch.setattr(trading_bot, "get_prediction", lambda *a, **k: called.__setitem__("pred", True))
    monkeypatch.setattr(trading_bot, "_load_env", lambda: {
        "data_handler_url": "http://dh",
        "model_builder_url": "http://mb",
        "trade_manager_url": "http://tm",
    })

    trading_bot.run_once()
    assert not called["pred"]


def test_run_once_forwards_prediction_params(monkeypatch):
    sent = {}
    monkeypatch.setattr(trading_bot, "fetch_price", lambda *a, **k: 100.0)
    monkeypatch.setattr(
        trading_bot,
        "get_prediction",
        lambda *a, **k: {"signal": "buy", "tp": 110, "sl": 90, "trailing_stop": 1},
    )
    monkeypatch.setattr(
        trading_bot,
        "send_trade",
        lambda *a, tp=None, sl=None, trailing_stop=None, **k: sent.update(
            tp=tp, sl=sl, trailing_stop=trailing_stop
        ),
    )
    monkeypatch.setattr(
        trading_bot,
        "_load_env",
        lambda: {
            "data_handler_url": "http://dh",
            "model_builder_url": "http://mb",
            "trade_manager_url": "http://tm",
        },
    )

    trading_bot.run_once()
    assert sent == {"tp": 110, "sl": 90, "trailing_stop": 1}


def test_run_once_env_fallback(monkeypatch):
    sent = {}
    monkeypatch.setattr(trading_bot, "fetch_price", lambda *a, **k: 100.0)
    monkeypatch.setattr(trading_bot, "get_prediction", lambda *a, **k: {"signal": "buy"})
    monkeypatch.setattr(
        trading_bot,
        "send_trade",
        lambda *a, tp=None, sl=None, trailing_stop=None, **k: sent.update(
            tp=tp, sl=sl, trailing_stop=trailing_stop
        ),
    )
    monkeypatch.setattr(
        trading_bot,
        "_load_env",
        lambda: {
            "data_handler_url": "http://dh",
            "model_builder_url": "http://mb",
            "trade_manager_url": "http://tm",
        },
    )
    monkeypatch.setenv("TP", "10")
    monkeypatch.setenv("SL", "5")
    monkeypatch.setenv("TRAILING_STOP", "2")

    trading_bot.run_once()
    assert sent == {"tp": 10.0, "sl": 5.0, "trailing_stop": 2.0}
