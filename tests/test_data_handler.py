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
from config import BotConfig

# Replace utils with a stub that overrides TelegramLogger
real_utils = importlib.import_module('utils')
utils_stub = types.ModuleType('utils')
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
sys.modules['utils'] = utils_stub
optimizer_stubbed = False
if 'optimizer' not in sys.modules:
    optimizer_stubbed = True
    optimizer_stub = types.ModuleType('optimizer')
    class _PO:
        def __init__(self, *a, **k):
            pass
    optimizer_stub.ParameterOptimizer = _PO
    sys.modules['optimizer'] = optimizer_stub
os.environ['TEST_MODE'] = '1'

from data_handler import DataHandler
import data_handler

if optimizer_stubbed:
    sys.modules.pop('optimizer', None)

class DummyExchange:
    def __init__(self, volumes):
        self.volumes = volumes
    async def fetch_ticker(self, symbol):
        return {'quoteVolume': self.volumes.get(symbol, 0)}


def _expected_rate(tf: str) -> int:
    sec = pd.Timedelta(tf).total_seconds()
    return max(1, int(1800 / sec))

@pytest.mark.asyncio
async def test_select_liquid_pairs_plain_symbol_included():
    cfg = BotConfig(cache_dir='/tmp', max_symbols=5)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    markets = {
        'BTCUSDT': {'active': True},
        'BTC/USDT': {'active': True},
    }
    pairs = await dh.select_liquid_pairs(markets)
    assert 'BTCUSDT' in pairs


@pytest.mark.asyncio
async def test_select_liquid_pairs_prefers_highest_volume():
    cfg = BotConfig(cache_dir='/tmp', max_symbols=5)
    volumes = {'BTCUSDT': 1.0, 'BTC/USDT:USDT': 2.0}
    dh = DataHandler(cfg, None, None, exchange=DummyExchange(volumes))
    markets = {
        'BTCUSDT': {'active': True},
        'BTC/USDT:USDT': {'active': True},
    }
    pairs = await dh.select_liquid_pairs(markets)
    assert pairs == ['BTC/USDT:USDT']


def test_dynamic_ws_min_process_rate_short_tf():
    cfg = BotConfig(cache_dir='/tmp', timeframe='1m')
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    assert dh.ws_min_process_rate == _expected_rate('1m')


def test_dynamic_ws_min_process_rate_long_tf():
    cfg = BotConfig(cache_dir='/tmp', timeframe='2h')
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
async def test_ws_rate_limit_zero_no_exception():
    cfg = BotConfig(cache_dir='/tmp', ws_rate_limit=0)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    ws = DummyWS()
    await dh._send_subscriptions(ws, ['BTCUSDT'], 'primary')
    assert ws.sent


def test_price_endpoint_returns_default():
    from data_handler import api_app, DEFAULT_PRICE
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
async def test_cleanup_old_data_recovery(monkeypatch):
    cfg = BotConfig(cache_dir='/tmp', data_cleanup_interval=0)
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
async def test_monitor_load_recovery(monkeypatch):
    cfg = BotConfig(cache_dir='/tmp')
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
async def test_process_ws_queue_recovery(monkeypatch):
    cfg = BotConfig(cache_dir='/tmp')
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
async def test_stop_handles_close_errors():
    cfg = BotConfig(cache_dir='/tmp')
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
async def test_load_initial_no_attribute_error(monkeypatch, tmp_path):
    class DummyExchange2:
        async def load_markets(self):
            return {'BTCUSDT': {'active': True}}
        async def fetch_ticker(self, symbol):
            return {'quoteVolume': 1.0}
    cfg = BotConfig(cache_dir=str(tmp_path), max_symbols=1, min_data_length=1, timeframe='1m', secondary_timeframe='1m')
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
async def test_fetch_ohlcv_single_empty_not_cached(tmp_path, monkeypatch):
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
    assert not (tmp_path / 'BTCUSDT_1m.pkl.gz').exists()


@pytest.mark.asyncio
async def test_fetch_ohlcv_history_empty_not_cached(tmp_path, monkeypatch):
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
    assert not (tmp_path / 'BTCUSDT_1m.pkl.gz').exists()


@pytest.mark.asyncio
async def test_fetch_open_interest_sets_change():
    class Ex:
        def __init__(self):
            self.val = 100.0
        async def fetch_open_interest(self, symbol):
            self.val += 10.0
            return {"openInterest": self.val}

    cfg = BotConfig(cache_dir='/tmp')
    dh = DataHandler(cfg, None, None, exchange=Ex())

    first = await dh.fetch_open_interest('BTCUSDT')
    assert first == 110.0
    assert dh.open_interest_change['BTCUSDT'] == 0.0
    second = await dh.fetch_open_interest('BTCUSDT')
    expected = (second - first) / first
    assert pytest.approx(dh.open_interest_change['BTCUSDT']) == expected


@pytest.mark.asyncio
async def test_process_ws_queue_no_warning_on_unconfirmed(caplog):
    cfg = BotConfig(cache_dir='/tmp')
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
async def test_subscribe_to_klines_single_timeframe(monkeypatch):
    cfg = BotConfig(cache_dir='/tmp', timeframe='1m', secondary_timeframe='1m')
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

    class DummyInd:
        def update(self, _):
            pass

    dh.indicators_cache[f'{symbol}_primary'] = DummyInd()

    await dh.synchronize_and_update(symbol, df, 0.0, 0.0, {'imbalance': 0.0, 'timestamp': time.time()})
    await asyncio.sleep(0)
    assert called == [symbol]


@pytest.mark.asyncio
async def test_process_ws_queue_callback_after_sync(monkeypatch):
    cfg = BotConfig(cache_dir='/tmp')
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

