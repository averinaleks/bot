import os, sys
import importlib
import types
import logging
import asyncio
import contextlib
import json
import time
import pandas as pd
import pytest
from bot import trading_bot
from bot.config import BotConfig

# Replace utils with a stub that overrides TelegramLogger
real_utils = importlib.import_module('bot.utils')
utils_stub = types.ModuleType('bot.utils')
utils_stub.__dict__.update(real_utils.__dict__)
class DummyTL:
    def __init__(self, *a, **k):
        pass
    async def send_telegram_message(self, *a, **k):
        pass
    @classmethod
    async def shutdown(cls):
        pass
utils_stub.TelegramLogger = DummyTL
utils_stub.logger = logging.getLogger('test')
sys.modules['bot.utils'] = utils_stub
optimizer_stubbed = False
if 'optimizer' not in sys.modules and 'bot.optimizer' not in sys.modules:
    optimizer_stubbed = True
    optimizer_stub = types.ModuleType('optimizer')
    class _PO:
        def __init__(self, *a, **k):
            pass
    optimizer_stub.ParameterOptimizer = _PO
    sys.modules['optimizer'] = optimizer_stub
    sys.modules['bot.optimizer'] = optimizer_stub

from bot.data_handler import DataHandler
from bot import data_handler


@pytest.fixture(autouse=True)
def _set_test_mode(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    yield


@pytest.fixture
def cfg_factory(tmp_path):
    def _factory(**kwargs):
        return BotConfig(cache_dir=str(tmp_path), **kwargs)
    return _factory

if optimizer_stubbed:
    sys.modules.pop('optimizer', None)
    sys.modules.pop('bot.optimizer', None)

class DummyExchange:
    def __init__(self, volumes):
        self.volumes = volumes
    async def fetch_ticker(self, symbol):
        return {'quoteVolume': self.volumes.get(symbol, 0)}


def _expected_rate(tf: str) -> int:
    sec = pd.Timedelta(tf).total_seconds()
    return max(1, int(1800 / sec))

@pytest.mark.asyncio
async def test_select_liquid_pairs_plain_symbol_included(cfg_factory):
    cfg = cfg_factory(max_symbols=5, min_liquidity=0)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    markets = {
        'BTCUSDT': {
            'active': True,
            'contract': True,
            'linear': True,
            'quote': 'USDT',
        },
        'BTC/USDT': {
            'active': True,
            'contract': False,
            'linear': False,
            'quote': 'USDT',
        },
    }
    pairs = await dh.select_liquid_pairs(markets)
    assert 'BTCUSDT' in pairs


@pytest.mark.asyncio
async def test_select_liquid_pairs_prefers_highest_volume(cfg_factory):
    cfg = cfg_factory(max_symbols=5, min_liquidity=0)
    volumes = {'BTCUSDT': 1.0, 'BTC/USDT:USDT': 2.0}
    dh = DataHandler(cfg, None, None, exchange=DummyExchange(volumes))
    markets = {
        'BTCUSDT': {'active': True, 'contract': True, 'quote': 'USDT'},
        'BTC/USDT:USDT': {'active': True, 'contract': True, 'quote': 'USDT'},
    }
    pairs = await dh.select_liquid_pairs(markets)
    assert pairs == ['BTC/USDT:USDT']


@pytest.mark.asyncio
async def test_select_liquid_pairs_filters_by_min_liquidity(cfg_factory):
    cfg = cfg_factory(max_symbols=5, min_liquidity=100)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 50}))
    markets = {
        'BTCUSDT': {'active': True, 'contract': True, 'quote': 'USDT'}
    }
    with pytest.raises(ValueError):
        await dh.select_liquid_pairs(markets)


@pytest.mark.asyncio
async def test_select_liquid_pairs_filters_new_listings(cfg_factory):
    now = pd.Timestamp.utcnow()
    recent = int((now - pd.Timedelta(minutes=5)).timestamp() * 1000)
    old = int((now - pd.Timedelta(hours=2)).timestamp() * 1000)
    volumes = {'NEWUSDT': 1.0, 'OLDUSDT': 1.0}
    cfg = cfg_factory(max_symbols=5, min_liquidity=0, min_data_length=10, timeframe='1m')
    dh = DataHandler(cfg, None, None, exchange=DummyExchange(volumes))
    markets = {
        'NEWUSDT': {
            'active': True,
            'info': {'launchTime': recent},
            'contract': True,
            'quote': 'USDT',
        },
        'OLDUSDT': {
            'active': True,
            'info': {'launchTime': old},
            'contract': True,
            'quote': 'USDT',
        },
    }
    pairs = await dh.select_liquid_pairs(markets)
    assert 'NEWUSDT' not in pairs
    assert 'OLDUSDT' in pairs


def test_dynamic_ws_min_process_rate_short_tf(cfg_factory):
    cfg = cfg_factory(timeframe='1m')
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    assert dh.ws_min_process_rate == _expected_rate('1m')


def test_dynamic_ws_min_process_rate_long_tf(cfg_factory):
    cfg = cfg_factory(timeframe='2h')
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    assert dh.ws_min_process_rate == _expected_rate('2h')


class DummyWS:
    def __init__(self):
        self.sent = []
    async def send(self, message):
        self.sent.append(message)
    async def recv(self):
        return '{"success": true}'


@pytest.mark.asyncio
async def test_ws_rate_limit_zero_no_exception(cfg_factory):
    cfg = cfg_factory(ws_rate_limit=0)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    ws = DummyWS()
    await dh._send_subscriptions(ws, ['BTCUSDT'], 'primary')
    assert ws.sent


def test_price_endpoint_returns_default():
    from bot.data_handler import api_app, DEFAULT_PRICE
    with api_app.test_client() as client:
        resp = client.get('/price/UNKNOWN')
        assert resp.status_code == 200
        assert resp.get_json() == {'price': DEFAULT_PRICE}


@pytest.mark.asyncio
async def test_load_from_disk_buffer_loop(tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path))
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    loop_task = asyncio.create_task(dh.load_from_disk_buffer_loop())

    item = (["BTCUSDT"], "message", "primary")
    await dh.save_to_disk_buffer(1, item)

    for _ in range(10):
        if not dh.ws_queue.empty():
            break
        await asyncio.sleep(0.2)

    assert not dh.ws_queue.empty()
    priority, loaded = await dh.ws_queue.get()
    assert priority == 1
    assert loaded == item

    loop_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await loop_task


@pytest.mark.asyncio
async def test_disk_buffer_path_escape(tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path))
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))

    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    dh.disk_buffer.put_nowait(str(outside))

    with pytest.raises(ValueError):
        await dh.load_from_disk_buffer()


@pytest.mark.asyncio
async def test_cleanup_old_data_recovery(monkeypatch, cfg_factory):
    cfg = cfg_factory(data_cleanup_interval=0)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))

    call = {'n': 0}
    orig_now = pd.Timestamp.now

    def fake_now(*a, **k):
        call['n'] += 1
        if call['n'] == 1:
            raise RuntimeError('boom')
        return orig_now(*a, **k)

    monkeypatch.setattr(pd.Timestamp, 'now', fake_now)

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(data_handler.asyncio, 'sleep', fast_sleep)

    task = asyncio.create_task(dh.cleanup_old_data())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call['n'] >= 2


@pytest.mark.asyncio
async def test_monitor_load_recovery(monkeypatch, cfg_factory):
    cfg = cfg_factory()
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))

    call = {'n': 0}

    async def fake_adjust():
        call['n'] += 1
        if call['n'] == 1:
            raise RuntimeError('boom')

    monkeypatch.setattr(dh, 'adjust_subscriptions', fake_adjust)

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(data_handler.asyncio, 'sleep', fast_sleep)

    task = asyncio.create_task(dh.monitor_load())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call['n'] >= 2


@pytest.mark.asyncio
async def test_funding_rate_loop_recovery(monkeypatch, cfg_factory):
    cfg = cfg_factory(funding_update_interval=0)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    dh.usdt_pairs = ['BTCUSDT']

    call = {'n': 0}

    async def fake_fetch(sym):
        call['n'] += 1
        if call['n'] == 1:
            raise RuntimeError('boom')

    monkeypatch.setattr(dh, 'fetch_funding_rate', fake_fetch)

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(data_handler.asyncio, 'sleep', fast_sleep)

    task = asyncio.create_task(dh.funding_rate_loop())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call['n'] >= 2


@pytest.mark.asyncio
async def test_open_interest_loop_recovery(monkeypatch, cfg_factory):
    cfg = cfg_factory(oi_update_interval=0)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    dh.usdt_pairs = ['BTCUSDT']

    call = {'n': 0}

    async def fake_fetch(sym):
        call['n'] += 1
        if call['n'] == 1:
            raise RuntimeError('boom')

    monkeypatch.setattr(dh, 'fetch_open_interest', fake_fetch)

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(data_handler.asyncio, 'sleep', fast_sleep)

    task = asyncio.create_task(dh.open_interest_loop())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call['n'] >= 2


@pytest.mark.asyncio
async def test_process_ws_queue_recovery(monkeypatch, cfg_factory):
    cfg = cfg_factory()
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))

    processed = []

    async def fake_sync(symbol, df, fr, oi, ob, timeframe='primary'):
        processed.append(symbol)

    monkeypatch.setattr(dh, 'synchronize_and_update', fake_sync)

    call = {'n': 0}
    orig_loads = data_handler.json.loads

    def fake_loads(s):
        call['n'] += 1
        if call['n'] == 1:
            raise ValueError('boom')
        return orig_loads(s)

    monkeypatch.setattr(data_handler.json, 'loads', fake_loads)

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(data_handler.asyncio, 'sleep', fast_sleep)

    msg = json.dumps({
        'topic': 'kline.1.BTCUSDT',
        'data': [{
            'start': int(pd.Timestamp.now(tz='UTC').timestamp() * 1000),
            'open': 1, 'high': 2, 'low': 0.5, 'close': 1.5, 'volume': 1
        }]
    })

    await dh.ws_queue.put((1, (['BTCUSDT'], msg, 'primary')))
    await dh.ws_queue.put((1, (['BTCUSDT'], msg, 'primary')))
    await dh.ws_queue.put((1, (['BTCUSDT'], msg, 'primary')))

    task = asyncio.create_task(dh._process_ws_queue())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert processed == ['BTCUSDT']
    assert call['n'] >= 2


@pytest.mark.asyncio
async def test_stop_handles_close_errors(cfg_factory):
    cfg = cfg_factory()
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))

    class BadWS:
        async def close(self):
            raise RuntimeError('boom')

    class BadPro:
        async def close(self):
            raise RuntimeError('boom')

    dh.ws_pool = {'ws://': [BadWS()]}
    dh.pro_exchange = BadPro()

    await dh.stop()  # should not raise


@pytest.mark.asyncio
async def test_stop_shuts_down_ray(cfg_factory):
    import ray
    cfg = cfg_factory()
    ray.init(
        num_cpus=cfg["ray_num_cpus"], num_gpus=1, ignore_reinit_error=True
    )
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    await dh.stop()
    assert not ray.is_initialized()
@pytest.mark.asyncio
async def test_load_initial_no_attribute_error(monkeypatch, tmp_path):
    class DummyExchange2:
        async def load_markets(self):
            return {
                'BTCUSDT': {
                    'active': True,
                    'contract': True,
                    'quote': 'USDT',
                }
            }
        async def fetch_ticker(self, symbol):
            return {'quoteVolume': 1.0}
    cfg = BotConfig(cache_dir=str(tmp_path), max_symbols=1, min_data_length=1, timeframe='1m', secondary_timeframe='1m', min_liquidity=0)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange2())
    async def fake_orderbook(symbol):
        return {'bids': [[1,1]], 'asks': [[1,1]]}
    async def fake_history(symbol, timeframe, limit, cache_prefix=""):
        df = pd.DataFrame({'open':[1],'high':[1],'low':[1],'close':[1],'volume':[1]}, index=[pd.Timestamp.now(tz='UTC')])
        return symbol, df
    async def fake_rate(symbol):
        return 0.0
    async def fake_oi(symbol):
        return 0.0
    monkeypatch.setattr(dh, 'fetch_orderbook', fake_orderbook)
    monkeypatch.setattr(dh, 'fetch_ohlcv_history', fake_history)
    monkeypatch.setattr(dh, 'fetch_funding_rate', fake_rate)
    monkeypatch.setattr(dh, 'fetch_open_interest', fake_oi)
    monkeypatch.setattr(data_handler, 'check_dataframe_empty', lambda df, context='': False)
    await dh.load_initial()  # should not raise
    assert dh.usdt_pairs == ['BTCUSDT']


@pytest.mark.asyncio
async def test_load_initial_raises_when_no_pairs(monkeypatch, tmp_path):
    class DummyExchange:
        async def load_markets(self):
            return {}

    cfg = BotConfig(cache_dir=str(tmp_path))
    dh = DataHandler(cfg, None, None, exchange=DummyExchange())

    async def fake_select(markets):
        return []

    monkeypatch.setattr(dh, 'select_liquid_pairs', fake_select)

    with pytest.raises(RuntimeError, match="No liquid USDT pairs"):
        await dh.load_initial()


@pytest.mark.asyncio
async def test_fetch_ohlcv_single_empty_not_cached(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_MODE", raising=False)
    class Ex:
        async def fetch_ohlcv(self, symbol, timeframe, limit=200, since=None):
            return []

    cfg = BotConfig(cache_dir=str(tmp_path))
    dh = DataHandler(cfg, None, None, exchange=Ex())

    async def fake_call(exchange, method, *args, **kwargs):
        return await getattr(exchange, method)(*args, **kwargs)

    monkeypatch.setattr(data_handler, 'safe_api_call', fake_call)

    _, df = await dh.fetch_ohlcv_single('BTCUSDT', '1m', limit=5)
    assert df.empty
    assert not (tmp_path / 'BTCUSDT_1m.parquet').exists()


@pytest.mark.asyncio
async def test_fetch_ohlcv_history_empty_not_cached(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_MODE", raising=False)
    class Ex:
        async def fetch_ohlcv(self, symbol, timeframe, limit=200, since=None):
            return []

    cfg = BotConfig(cache_dir=str(tmp_path))
    dh = DataHandler(cfg, None, None, exchange=Ex())

    async def fake_call(exchange, method, *args, **kwargs):
        return await getattr(exchange, method)(*args, **kwargs)

    monkeypatch.setattr(data_handler, 'safe_api_call', fake_call)

    _, df = await dh.fetch_ohlcv_history('BTCUSDT', '1m', total_limit=5)
    assert df.empty
    assert not (tmp_path / 'BTCUSDT_1m.parquet').exists()


@pytest.mark.asyncio
async def test_fetch_open_interest_sets_change(cfg_factory):
    class Ex:
        def __init__(self):
            self.val = 100.0
        async def fetch_open_interest(self, symbol):
            self.val += 10.0
            return {"openInterest": self.val}

    cfg = cfg_factory()
    dh = DataHandler(cfg, None, None, exchange=Ex())

    first = await dh.fetch_open_interest('BTCUSDT')
    assert first == 110.0
    assert dh.open_interest_change['BTCUSDT'] == 0.0
    second = await dh.fetch_open_interest('BTCUSDT')
    expected = (second - first) / first
    assert pytest.approx(dh.open_interest_change['BTCUSDT']) == expected


@pytest.mark.asyncio
async def test_process_ws_queue_no_warning_on_unconfirmed(caplog, cfg_factory):
    cfg = cfg_factory()
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))

    caplog.set_level(logging.WARNING)
    ts = int((pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=2)).timestamp() * 1000)
    msg = json.dumps({
        'topic': 'kline.1.BTCUSDT',
        'data': [{
            'start': ts,
            'open': 1,
            'high': 2,
            'low': 0.5,
            'close': 1.5,
            'volume': 1,
            'confirm': False,
        }]
    })

    await dh.ws_queue.put((1, (['BTCUSDT'], msg, 'primary')))

    task = asyncio.create_task(dh._process_ws_queue())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not any('Получены устаревшие данные' in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_subscribe_to_klines_single_timeframe(monkeypatch, cfg_factory):
    cfg = cfg_factory(timeframe='1m', secondary_timeframe='1m')
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))

    call = {'n': 0}

    async def fake_subscribe_chunk(*a, **k):
        call['n'] += 1

    async def fake_task(*a, **k):
        return None

    monkeypatch.setattr(dh, '_subscribe_chunk', fake_subscribe_chunk)
    monkeypatch.setattr(dh, '_process_ws_queue', fake_task)
    monkeypatch.setattr(dh, 'load_from_disk_buffer_loop', fake_task)
    monkeypatch.setattr(dh, 'monitor_load', fake_task)
    monkeypatch.setattr(dh, 'cleanup_old_data', fake_task)
    monkeypatch.setattr(dh, 'funding_rate_loop', fake_task)
    monkeypatch.setattr(dh, 'open_interest_loop', fake_task)

    await dh.subscribe_to_klines(['BTCUSDT'])
    assert call['n'] == 1


@pytest.mark.asyncio
async def test_feature_callback_invoked(tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path))
    called = []

    async def cb(sym):
        called.append(sym)

    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}), feature_callback=cb)
    symbol = 'BTCUSDT'
    ts = pd.Timestamp.now(tz='UTC')
    df = pd.DataFrame({'open': [1], 'high': [1], 'low': [1], 'close': [1], 'volume': [1]}, index=[ts])
    df['symbol'] = symbol
    df = df.set_index(['symbol', df.index])
    df.index.set_names(['symbol', 'timestamp'], inplace=True)



    await dh.synchronize_and_update(symbol, df, 0.0, 0.0, {'imbalance': 0.0, 'timestamp': time.time()})
    await asyncio.sleep(0)
    assert 'ema30' in dh.ohlcv.columns
    assert called == [symbol]


@pytest.mark.asyncio
async def test_trade_callback_invoked(monkeypatch, tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path))

    records = []

    class DummyClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def get(self, url, timeout=None):
            records.append(("get", url))
            return types.SimpleNamespace(status_code=200, json=lambda: {"price": 1.0})
        async def post(self, url, json=None, timeout=None, headers=None):
            records.append(("post", url, json))
            if url.endswith("/predict"):
                return types.SimpleNamespace(status_code=200, json=lambda: {"signal": "buy"})
            return types.SimpleNamespace(status_code=200, json=lambda: {})

    monkeypatch.setattr(
        trading_bot.httpx,
        "AsyncClient",
        lambda *a, **k: DummyClient(),
        raising=False,
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

    dh = DataHandler(
        cfg,
        None,
        None,
        exchange=DummyExchange({"BTCUSDT": 1.0}),
        trade_callback=trading_bot.reactive_trade,
    )
    symbol = "BTCUSDT"
    ts = pd.Timestamp.now(tz="UTC")
    df = pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}, index=[ts]
    )
    df["symbol"] = symbol
    df = df.set_index(["symbol", df.index])
    df.index.set_names(["symbol", "timestamp"], inplace=True)

    await dh.synchronize_and_update(
        symbol, df, 0.0, 0.0, {"imbalance": 0.0, "timestamp": time.time()}
    )
    await asyncio.sleep(0)

    assert any(r[0] == "post" and r[1].endswith("/predict") for r in records)
    assert any(r[0] == "post" and r[1].endswith("/open_position") for r in records)


@pytest.mark.asyncio
async def test_process_ws_queue_callback_after_sync(monkeypatch, cfg_factory):
    cfg = cfg_factory()
    order = []

    async def fake_sync(symbol, df, fr, oi, ob, timeframe='primary'):
        order.append('sync')

    async def cb(sym):
        order.append('cb')

    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}), feature_callback=cb)
    monkeypatch.setattr(dh, 'synchronize_and_update', fake_sync)

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(data_handler.asyncio, 'sleep', fast_sleep)

    created = []
    orig_create = asyncio.create_task

    def record_create(coro):
        created.append('cb')
        return orig_create(coro)

    monkeypatch.setattr(asyncio, 'create_task', record_create)

    msg = json.dumps({
        'topic': 'kline.1.BTCUSDT',
        'data': [{
            'start': int(pd.Timestamp.now(tz='UTC').timestamp() * 1000),
            'open': 1, 'high': 2, 'low': 0.5, 'close': 1.5, 'volume': 1
        }]
    })

    await dh.ws_queue.put((1, (['BTCUSDT'], msg, 'primary')))

    task = asyncio.create_task(dh._process_ws_queue())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert order == ['sync']
    assert created == ['cb']


def test_calculate_imbalance():
    ob = {
        'bids': [[100, 10], [99, 5]],
        'asks': [[101, 8], [102, 1]],
    }
    expected = (15 - 9) / (15 + 9)
    assert pytest.approx(data_handler.calculate_imbalance(ob)) == expected


def test_detect_clusters():
    ob = {
        'bids': [[100, 10], [99, 5]],
        'asks': [[101, 8], [102, 1]],
    }
    clusters = data_handler.detect_clusters(ob, data_handler.DEFAULT_CLUSTER_THRESHOLD)
    assert clusters


@pytest.mark.asyncio
async def test_sync_updates_metrics(monkeypatch, cfg_factory):
    cfg = cfg_factory()
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    ts = pd.Timestamp.now(tz='UTC')
    df = pd.DataFrame({'open':[1], 'high':[1], 'low':[1], 'close':[1], 'volume':[1]}, index=[ts])
    df['symbol'] = 'BTCUSDT'
    df = df.set_index(['symbol', df.index])
    df.index.set_names(['symbol', 'timestamp'], inplace=True)
    ob = {
        'bids': [[100, 10], [99, 5]],
        'asks': [[101, 8], [102, 1]],
    }
    await dh.synchronize_and_update('BTCUSDT', df, 0.0, 0.0, ob)
    assert 'BTCUSDT' in dh.orderbook_imbalance
    assert 'BTCUSDT' in dh.order_clusters

    with data_handler.api_app.test_client() as client:
        resp = client.get('/imbalance/BTCUSDT')
        assert resp.status_code == 200
        assert 'imbalance' in resp.get_json()
        resp = client.get('/clusters/BTCUSDT')
        assert resp.status_code == 200
        assert 'clusters' in resp.get_json()


@pytest.mark.asyncio
async def test_indicator_cache_update_writable(tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path))
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    ts = pd.Timestamp.now(tz='UTC')
    df = pd.DataFrame({'open':[1], 'high':[1], 'low':[1], 'close':[1], 'volume':[1]}, index=[ts])
    df['symbol'] = 'BTCUSDT'
    df = df.set_index(['symbol', df.index])
    df.index.set_names(['symbol', 'timestamp'], inplace=True)
    await dh.synchronize_and_update('BTCUSDT', df, 0.0, 0.0, {'bids': [], 'asks': []})
    ind = dh.indicators_cache['BTCUSDT_primary']
    new_ts = ts + pd.Timedelta(minutes=1)
    new_df = pd.DataFrame({'close':[2], 'high':[2], 'low':[2], 'volume':[1]}, index=[new_ts])
    try:
        ind.update(new_df)
    except ValueError as exc:
        pytest.fail(f"update raised {exc}")
    assert new_ts in ind.df.index


@pytest.mark.asyncio
async def test_ws_inactivity_triggers_reconnect(monkeypatch, cfg_factory):
    cfg = cfg_factory(ws_inactivity_timeout=1)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))

    class DummyWS:
        def __init__(self):
            self.closed = False
            self.pings = 0
            self.open = True

        async def recv(self):
            await asyncio.sleep(0)
            raise asyncio.TimeoutError

        async def ping(self):
            self.pings += 1

        async def close(self):
            self.closed = True
            self.open = False

    ws = DummyWS()
    monkeypatch.setattr(dh, 'telegram_logger', types.SimpleNamespace(send_telegram_message=lambda *a, **k: asyncio.sleep(0)))
    with pytest.raises(ConnectionError):
        await dh._read_messages(ws, ['BTCUSDT'], 'primary', '1m', 0.1)
    assert ws.closed


@pytest.mark.asyncio
async def test_subscribe_chunk_uses_backup(monkeypatch, cfg_factory):
    cfg = cfg_factory(backup_ws_urls=['ws://backup'])
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))

    calls = []

    class DummyWS:
        def __init__(self):
            self.open = True

        async def close(self):
            self.open = False

    async def fake_connect(url, timeout):
        calls.append(url)
        if url not in dh.ws_pool:
            dh.ws_pool[url] = []
        if len(calls) == 1:
            raise OSError('boom')
        return DummyWS()

    async def fake_send_subs(ws, symbols, timeframe):
        return []

    async def fake_read(ws, *a, **k):
        raise KeyboardInterrupt

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(dh, '_connect_ws', fake_connect)
    monkeypatch.setattr(dh, '_send_subscriptions', fake_send_subs)
    monkeypatch.setattr(dh, '_read_messages', fake_read)
    async def fake_fetch(*a, **k):
        return 'BTCUSDT', pd.DataFrame()

    monkeypatch.setattr(dh, 'fetch_ohlcv_single', fake_fetch)
    monkeypatch.setattr(data_handler, 'check_dataframe_empty', lambda df, context='': True)
    monkeypatch.setattr(data_handler.asyncio, 'sleep', fast_sleep)

    with pytest.raises(KeyboardInterrupt):
        await dh._subscribe_chunk(['BTCUSDT'], 'ws://primary', 0, 'primary')

    assert calls == ['ws://primary', 'ws://backup']

sys.modules['bot.utils'] = real_utils

