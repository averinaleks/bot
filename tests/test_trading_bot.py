from bot import trading_bot
import asyncio
import time
import types
import pytest


def test_send_trade_timeout_env(monkeypatch):
    called = {}

    def fake_post(self, url, json=None, timeout=None, headers=None):
        called['timeout'] = timeout
        class Resp:
            status_code = 200
            def json(self):
                return {"status": "ok"}
        return Resp()

    monkeypatch.setattr(trading_bot.httpx.Client, 'post', fake_post)
    monkeypatch.setenv('TRADE_MANAGER_TIMEOUT', '9')
    result = trading_bot.send_trade('BTCUSDT', 'buy', 100.0, {'trade_manager_url': 'http://tm'})
    assert called['timeout'] == 9.0
    assert result is True


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

    def fake_post(self, url, json=None, timeout=None, headers=None):
        time.sleep(0.01)
        class Resp:
            status_code = 200
            def json(self):
                return {"status": "ok"}
        return Resp()

    monkeypatch.setattr(trading_bot.httpx.Client, 'post', fake_post)
    async def fake_alert(msg):
        called.append(msg)
    monkeypatch.setattr(trading_bot, 'send_telegram_alert', fake_alert)
    trading_bot.CONFIRMATION_TIMEOUT = 0.0
    trading_bot.send_trade('BTCUSDT', 'buy', 1.0, {'trade_manager_url': 'http://tm'})
    assert called


def test_send_trade_http_error_alert(monkeypatch):
    called = []

    def fake_post(self, url, json=None, timeout=None, headers=None):
        class Resp:
            status_code = 500
            def json(self):
                return {}
        return Resp()

    monkeypatch.setattr(trading_bot.httpx.Client, 'post', fake_post)
    async def fake_alert(msg):
        called.append(msg)
    monkeypatch.setattr(trading_bot, 'send_telegram_alert', fake_alert)
    trading_bot.send_trade('BTCUSDT', 'sell', 1.0, {'trade_manager_url': 'http://tm'})
    assert called


def test_send_trade_exception_alert(monkeypatch):
    called = []

    def fake_post(self, url, json=None, timeout=None, headers=None):
        raise trading_bot.httpx.HTTPError('boom')

    monkeypatch.setattr(trading_bot.httpx.Client, 'post', fake_post)
    async def fake_alert(msg):
        called.append(msg)
    monkeypatch.setattr(trading_bot, 'send_telegram_alert', fake_alert)
    trading_bot.send_trade('BTCUSDT', 'sell', 1.0, {'trade_manager_url': 'http://tm'})
    assert called


@pytest.mark.asyncio
async def test_send_trade_in_async_context(monkeypatch):
    await asyncio.sleep(0)
    def fake_post_trade(*args, **kwargs):
        return True, 0.0, None

    monkeypatch.setattr(trading_bot, '_post_trade', fake_post_trade)

    class DummyClient:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(trading_bot.httpx, 'Client', lambda trust_env=False: DummyClient())

    called = {}

    class DummyLoop:
        def run_until_complete(self, coro):
            called['used'] = True
            return coro

    monkeypatch.setattr(trading_bot.asyncio, 'get_running_loop', lambda: DummyLoop())

    result = trading_bot.send_trade('BTCUSDT', 'buy', 1.0, {'trade_manager_url': 'http://tm'})
    assert result is True
    assert called['used']


@pytest.mark.asyncio
async def test_monitor_positions_tp(monkeypatch):
    env = {'trade_manager_url': 'http://tm', 'data_handler_url': 'http://dh'}
    class DummyResp:
        status_code = 200
        def json(self):
            return {'positions': [{'id': '1', 'symbol': 'BTCUSDT', 'side': 'buy', 'tp': 100, 'sl': 90, 'trailing_stop': None, 'entry_price': 95}]}
    called = {}
    class DummyClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
        async def get(self, url, timeout=None):
            return DummyResp()
        async def post(self, url, json=None, timeout=None):
            called['payload'] = json
            return DummyResp()
    dummy = DummyClient()
    monkeypatch.setattr(trading_bot.httpx, 'AsyncClient', lambda *a, **k: dummy)
    async def fake_price(symbol, env):
        return 101
    monkeypatch.setattr(trading_bot, 'fetch_price', fake_price)
    orig_sleep = asyncio.sleep
    async def fast_sleep(_):
        await orig_sleep(0)
    monkeypatch.setattr(trading_bot.asyncio, 'sleep', fast_sleep)
    task = asyncio.create_task(trading_bot.monitor_positions(env, interval=0.01))
    await orig_sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert called['payload']['order_id'] == '1'


@pytest.mark.asyncio
async def test_monitor_positions_sl(monkeypatch):
    env = {'trade_manager_url': 'http://tm', 'data_handler_url': 'http://dh'}
    class DummyResp:
        status_code = 200
        def json(self):
            return {'positions': [{'id': '2', 'symbol': 'BTCUSDT', 'side': 'buy', 'tp': None, 'sl': 90, 'trailing_stop': None, 'entry_price': 100}]}
    called = {}
    class DummyClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
        async def get(self, url, timeout=None):
            return DummyResp()
        async def post(self, url, json=None, timeout=None):
            called['payload'] = json
            return DummyResp()
    dummy = DummyClient()
    monkeypatch.setattr(trading_bot.httpx, 'AsyncClient', lambda *a, **k: dummy)
    async def fake_price(symbol, env):
        return 89
    monkeypatch.setattr(trading_bot, 'fetch_price', fake_price)
    orig_sleep = asyncio.sleep
    async def fast_sleep(_):
        await orig_sleep(0)
    monkeypatch.setattr(trading_bot.asyncio, 'sleep', fast_sleep)
    task = asyncio.create_task(trading_bot.monitor_positions(env, interval=0.01))
    await orig_sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert called['payload']['order_id'] == '2'


@pytest.mark.asyncio
async def test_monitor_positions_trailing_stop(monkeypatch):
    env = {'trade_manager_url': 'http://tm', 'data_handler_url': 'http://dh'}
    class DummyResp:
        status_code = 200
        def json(self):
            return {'positions': [{'id': '3', 'symbol': 'BTCUSDT', 'side': 'buy', 'tp': None, 'sl': None, 'trailing_stop': 1, 'entry_price': 100}]}
    called = {}
    class DummyClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
        async def get(self, url, timeout=None):
            return DummyResp()
        async def post(self, url, json=None, timeout=None):
            called['payload'] = json
            return DummyResp()
    dummy = DummyClient()
    monkeypatch.setattr(trading_bot.httpx, 'AsyncClient', lambda *a, **k: dummy)
    prices = {'i': 0}
    async def fake_price(symbol, env):
        prices['i'] += 1
        return 101 if prices['i'] == 1 else 99
    monkeypatch.setattr(trading_bot, 'fetch_price', fake_price)
    orig_sleep = asyncio.sleep
    async def fast_sleep(_):
        await orig_sleep(0)
    monkeypatch.setattr(trading_bot.asyncio, 'sleep', fast_sleep)
    task = asyncio.create_task(trading_bot.monitor_positions(env, interval=0.01))
    await orig_sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert called['payload']['order_id'] == '3'


def test_send_trade_forwards_params(monkeypatch):
    captured = {}

    def fake_post(self, url, json=None, timeout=None, headers=None):
        captured.update(json)
        class Resp:
            status_code = 200
            def json(self):
                return {"status": "ok"}
        return Resp()

    monkeypatch.setattr(trading_bot.httpx.Client, 'post', fake_post)
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


def test_send_trade_reports_error_field(monkeypatch):
    """An error field triggers alert even with HTTP 200."""
    called = []

    def fake_post(self, url, json=None, timeout=None, headers=None):
        class Resp:
            status_code = 200
            def json(self):
                return {"error": "boom"}
        return Resp()

    monkeypatch.setattr(trading_bot.httpx.Client, 'post', fake_post)
    async def fake_alert(msg):
        called.append(msg)
    monkeypatch.setattr(trading_bot, 'send_telegram_alert', fake_alert)
    ok = trading_bot.send_trade('BTCUSDT', 'buy', 1.0, {'trade_manager_url': 'http://tm'})
    assert not ok
    assert called


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

        async def post(self, url, json=None, timeout=None, headers=None):
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
    async def fake_alert(msg):
        called.append(msg)
    monkeypatch.setattr(trading_bot, "send_telegram_alert", fake_alert)
    trading_bot.CONFIRMATION_TIMEOUT = 0.0
    await trading_bot.reactive_trade("BTCUSDT")
    assert called and isinstance(called[0], dict)
    assert called[0]["tp"] == 10
    assert called[0]["sl"] == 5
    assert called[0]["trailing_stop"] == 1


@pytest.mark.asyncio
async def test_reactive_trade_invalid_json(monkeypatch, caplog):
    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, url, timeout=None):
            return types.SimpleNamespace(status_code=200, json=lambda: {"price": 1.0})

        async def post(self, url, json=None, timeout=None, headers=None):
            if url.endswith("/predict"):
                return types.SimpleNamespace(status_code=200, json=lambda: (_ for _ in ()).throw(ValueError("bad")))
            pytest.fail("Trade manager should not be called")

    monkeypatch.setattr(trading_bot.httpx, "AsyncClient", lambda *a, **k: DummyClient(), raising=False)
    monkeypatch.setattr(trading_bot, "_load_env", lambda: {
        "data_handler_url": "http://dh",
        "model_builder_url": "http://mb",
        "trade_manager_url": "http://tm",
    })
    with caplog.at_level("ERROR"):
        await trading_bot.reactive_trade("BTCUSDT")
    assert "invalid json" in caplog.text.lower()


@pytest.mark.asyncio
async def test_reactive_trade_reports_error_field(monkeypatch):
    """An error field triggers alert even with HTTP 200."""
    called = []

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, url, timeout=None):
            return types.SimpleNamespace(status_code=200, json=lambda: {"price": 1.0})

        async def post(self, url, json=None, timeout=None, headers=None):
            if url.endswith("/predict"):
                return types.SimpleNamespace(status_code=200, json=lambda: {"signal": "buy"})
            return types.SimpleNamespace(status_code=200, json=lambda: {"error": "boom"})

    monkeypatch.setattr(
        trading_bot.httpx, "AsyncClient", lambda *a, **k: DummyClient(), raising=False
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

    async def fake_alert(msg):
        called.append(msg)

    monkeypatch.setattr(trading_bot, "send_telegram_alert", fake_alert)

    await trading_bot.reactive_trade("BTCUSDT")

    assert called


def test_run_once_invalid_price(monkeypatch):
    """No trade sent when fetch_price returns a non-positive value."""
    sent = []

    async def fake_fetch(*a, **k):
        return 0.0
    monkeypatch.setattr(trading_bot, "fetch_price", fake_fetch)
    async def fake_send_trade(*a, **k):
        sent.append(True)

    monkeypatch.setattr(trading_bot, "send_trade_async", fake_send_trade)
    monkeypatch.setattr(trading_bot, "_load_env", lambda: {
        "data_handler_url": "http://dh",
        "model_builder_url": "http://mb",
        "trade_manager_url": "http://tm",
    })

    asyncio.run(trading_bot.run_once_async())
    assert not sent


def test_fetch_price_error(monkeypatch):
    class DummyClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def get(self, url, timeout=None):
            return types.SimpleNamespace(status_code=503, json=lambda: {"error": "down"})

    monkeypatch.setattr(trading_bot.httpx, "AsyncClient", lambda *a, **k: DummyClient(), raising=False)
    price = asyncio.run(trading_bot.fetch_price("BTCUSDT", {"data_handler_url": "http://dh"}))
    assert price is None


def test_fetch_price_invalid_json(monkeypatch, caplog):
    class DummyClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def get(self, url, timeout=None):
            return types.SimpleNamespace(status_code=200, json=lambda: (_ for _ in ()).throw(ValueError("bad")))

    monkeypatch.setattr(trading_bot.httpx, "AsyncClient", lambda *a, **k: DummyClient(), raising=False)
    with caplog.at_level("ERROR"):
        price = asyncio.run(trading_bot.fetch_price("BTCUSDT", {"data_handler_url": "http://dh"}))
    assert price is None
    assert "invalid json" in caplog.text.lower()


def test_get_prediction_invalid_json(monkeypatch, caplog):
    class DummyClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def post(self, url, json=None, timeout=None):
            return types.SimpleNamespace(status_code=200, json=lambda: (_ for _ in ()).throw(ValueError("bad")))

    monkeypatch.setattr(trading_bot.httpx, "AsyncClient", lambda *a, **k: DummyClient(), raising=False)
    with caplog.at_level("ERROR"):
        data = asyncio.run(trading_bot.get_prediction("BTCUSDT", [1, 2, 3, 4, 5], {"model_builder_url": "http://mb"}))
    assert data is None
    assert "invalid json" in caplog.text.lower()


def test_run_once_price_error(monkeypatch):
    called = {"pred": False}

    async def fake_fetch_none(*a, **k):
        return None
    monkeypatch.setattr(trading_bot, "fetch_price", fake_fetch_none)
    async def fake_get_pred(*a, **k):
        called.__setitem__("pred", True)
    monkeypatch.setattr(trading_bot, "get_prediction", fake_get_pred)
    monkeypatch.setattr(trading_bot, "_load_env", lambda: {
        "data_handler_url": "http://dh",
        "model_builder_url": "http://mb",
        "trade_manager_url": "http://tm",
    })

    asyncio.run(trading_bot.run_once_async())
    assert not called["pred"]


def test_run_once_forwards_prediction_params(monkeypatch):
    sent = {}
    async def fake_fetch_price(*a, **k):
        return 100.0
    monkeypatch.setattr(trading_bot, "fetch_price", fake_fetch_price)
    async def fake_get_prediction(*a, **k):
        return {"signal": "buy", "tp": 110, "sl": 90, "trailing_stop": 1}
    monkeypatch.setattr(trading_bot, "get_prediction", fake_get_prediction)
    async def fake_send_trade(
        *a, tp=None, sl=None, trailing_stop=None, **k
    ):
        sent.update(tp=tp, sl=sl, trailing_stop=trailing_stop)

    monkeypatch.setattr(trading_bot, "send_trade_async", fake_send_trade)
    monkeypatch.setattr(
        trading_bot,
        "_load_env",
        lambda: {
            "data_handler_url": "http://dh",
            "model_builder_url": "http://mb",
            "trade_manager_url": "http://tm",
        },
    )

    asyncio.run(trading_bot.run_once_async())
    assert sent == {"tp": 110, "sl": 90, "trailing_stop": 1}


def test_run_once_skips_on_gpt(monkeypatch):
    sent = []

    async def fake_fetch(*a, **k):
        return 100.0

    monkeypatch.setattr(trading_bot, "fetch_price", fake_fetch)

    async def fake_pred(*a, **k):
        return {"signal": "buy"}

    monkeypatch.setattr(trading_bot, "get_prediction", fake_pred)

    async def fake_send(*a, **k):
        sent.append(True)

    monkeypatch.setattr(trading_bot, "send_trade_async", fake_send)

    monkeypatch.setattr(
        trading_bot,
        "_load_env",
        lambda: {
            "data_handler_url": "http://dh",
            "model_builder_url": "http://mb",
            "trade_manager_url": "http://tm",
        },
    )

    trading_bot.GPT_ADVICE["signal"] = "sell"
    asyncio.run(trading_bot.run_once_async())
    trading_bot.GPT_ADVICE["signal"] = None
    assert not sent


def test_run_once_env_fallback(monkeypatch):
    sent = {}
    async def fake_fetch_price2(*a, **k):
        return 100.0
    monkeypatch.setattr(trading_bot, "fetch_price", fake_fetch_price2)
    async def fake_pred_buy(*a, **k):
        return {"signal": "buy"}
    monkeypatch.setattr(trading_bot, "get_prediction", fake_pred_buy)
    async def fake_send_trade2(
        *a, tp=None, sl=None, trailing_stop=None, **k
    ):
        sent.update(tp=tp, sl=sl, trailing_stop=trailing_stop)

    monkeypatch.setattr(trading_bot, "send_trade_async", fake_send_trade2)
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

    asyncio.run(trading_bot.run_once_async())
    assert sent == {"tp": 10.0, "sl": 5.0, "trailing_stop": 2.0}


def test_run_once_config_fallback(monkeypatch):
    """Defaults are computed from config when env and prediction lack values."""
    sent = {}
    async def fake_fetch_price3(*a, **k):
        return 100.0
    monkeypatch.setattr(trading_bot, "fetch_price", fake_fetch_price3)
    async def fake_pred_buy2(*a, **k):
        return {"signal": "buy"}
    monkeypatch.setattr(trading_bot, "get_prediction", fake_pred_buy2)
    async def fake_send_trade3(
        *a, tp=None, sl=None, trailing_stop=None, **k
    ):
        sent.update(tp=tp, sl=sl, trailing_stop=trailing_stop)

    monkeypatch.setattr(trading_bot, "send_trade_async", fake_send_trade3)
    monkeypatch.setattr(
        trading_bot,
        "_load_env",
        lambda: {
            "data_handler_url": "http://dh",
            "model_builder_url": "http://mb",
            "trade_manager_url": "http://tm",
        },
    )
    for var in ("TP", "SL", "TRAILING_STOP"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(trading_bot.CFG, "tp_multiplier", 1.1, raising=False)
    monkeypatch.setattr(trading_bot.CFG, "sl_multiplier", 0.9, raising=False)
    monkeypatch.setattr(trading_bot.CFG, "trailing_stop_multiplier", 0.05, raising=False)

    asyncio.run(trading_bot.run_once_async())
    assert sent == {
        "tp": pytest.approx(110.0),
        "sl": pytest.approx(90.0),
        "trailing_stop": pytest.approx(5.0),
    }


def test_run_once_ignores_invalid_env(monkeypatch):
    """Invalid TP/SL/trailing-stop strings fall back to config defaults."""
    sent = {}
    async def fake_fetch_price4(*a, **k):
        return 100.0
    monkeypatch.setattr(trading_bot, "fetch_price", fake_fetch_price4)
    async def fake_pred_buy3(*a, **k):
        return {"signal": "buy"}
    monkeypatch.setattr(trading_bot, "get_prediction", fake_pred_buy3)
    async def fake_send_trade4(
        *a, tp=None, sl=None, trailing_stop=None, **k
    ):
        sent.update(tp=tp, sl=sl, trailing_stop=trailing_stop)

    monkeypatch.setattr(trading_bot, "send_trade_async", fake_send_trade4)
    monkeypatch.setattr(
        trading_bot,
        "_load_env",
        lambda: {
            "data_handler_url": "http://dh",
            "model_builder_url": "http://mb",
            "trade_manager_url": "http://tm",
        },
    )
    monkeypatch.setenv("TP", "bad")
    monkeypatch.setenv("SL", "oops")
    monkeypatch.setenv("TRAILING_STOP", "uh-oh")
    monkeypatch.setattr(trading_bot.CFG, "tp_multiplier", 1.1, raising=False)
    monkeypatch.setattr(trading_bot.CFG, "sl_multiplier", 0.9, raising=False)
    monkeypatch.setattr(trading_bot.CFG, "trailing_stop_multiplier", 0.05, raising=False)

    asyncio.run(trading_bot.run_once_async())
    assert sent == {
        "tp": pytest.approx(110.0),
        "sl": pytest.approx(90.0),
        "trailing_stop": pytest.approx(5.0),
    }


def test_parse_trade_params_invalid_strings():
    tp, sl, ts = trading_bot._parse_trade_params("bad", "5", "x")
    assert tp is None
    assert sl == 5.0
    assert ts is None


@pytest.mark.parametrize(
    "args, env, expected",
    [
        ((120.0, 80.0, 6.0), {"TP": "130", "SL": "70", "TRAILING_STOP": "7"}, (120.0, 80.0, 6.0)),
        ((None, None, None), {"TP": "130", "SL": "70", "TRAILING_STOP": "7"}, (130.0, 70.0, 7.0)),
        ((None, None, None), {}, (110.0, 90.0, 5.0)),
        ((None, None, None), {"TP": "bad", "SL": "oops", "TRAILING_STOP": "?"}, (110.0, 90.0, 5.0)),
    ],
)
def test_resolve_trade_params(monkeypatch, args, env, expected):
    for var in ("TP", "SL", "TRAILING_STOP"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(trading_bot.CFG, "tp_multiplier", 1.1, raising=False)
    monkeypatch.setattr(trading_bot.CFG, "sl_multiplier", 0.9, raising=False)
    monkeypatch.setattr(trading_bot.CFG, "trailing_stop_multiplier", 0.05, raising=False)

    result = trading_bot._resolve_trade_params(*args, price=100.0)
    assert result == pytest.approx(expected)


def test_resolve_trade_params_gpt(monkeypatch):
    for var in ("TP", "SL", "TRAILING_STOP"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(trading_bot.CFG, "tp_multiplier", 1.0, raising=False)
    monkeypatch.setattr(trading_bot.CFG, "sl_multiplier", 1.0, raising=False)
    trading_bot.GPT_ADVICE["tp_mult"] = 1.5
    trading_bot.GPT_ADVICE["sl_mult"] = 0.5
    result = trading_bot._resolve_trade_params(None, None, None, price=100.0)
    trading_bot.GPT_ADVICE["tp_mult"] = None
    trading_bot.GPT_ADVICE["sl_mult"] = None
    assert result[:2] == pytest.approx((150.0, 50.0))


def test_resolve_trade_params_no_price(monkeypatch):
    for var in ("TP", "SL", "TRAILING_STOP"):
        monkeypatch.delenv(var, raising=False)
    assert trading_bot._resolve_trade_params(None, None, None, None) == (
        None,
        None,
        None,
    )


def test_run_once_logs_prediction(monkeypatch, caplog):
    """A prediction from the model service is logged."""
    async def fake_fetch_price5(*a, **k):
        return 100.0
    monkeypatch.setattr(trading_bot, "fetch_price", fake_fetch_price5)
    async def fake_pred_buy4(*a, **k):
        return {"signal": "buy"}
    monkeypatch.setattr(trading_bot, "get_prediction", fake_pred_buy4)
    async def fake_send_trade5(*a, **k):
        return None

    monkeypatch.setattr(trading_bot, "send_trade_async", fake_send_trade5)
    monkeypatch.setattr(
        trading_bot,
        "_load_env",
        lambda: {
            "data_handler_url": "http://dh",
            "model_builder_url": "http://mb",
            "trade_manager_url": "http://tm",
        },
    )

    with caplog.at_level("INFO"):
        asyncio.run(trading_bot.run_once_async())

    assert "Prediction: buy" in caplog.messages
