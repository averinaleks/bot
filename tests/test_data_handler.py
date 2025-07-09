import os, sys
import importlib
import types
import logging
import asyncio
import contextlib
import json
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
