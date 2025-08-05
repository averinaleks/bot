import os, sys
import importlib
import types
import logging
import asyncio
import pandas as pd
import polars as pl
import contextlib
import pytest
from bot.config import BotConfig

# Replace utils with a stub like in test_data_handler
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

# Ensure optimizer stub
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

if optimizer_stubbed:
    sys.modules.pop('optimizer', None)
    sys.modules.pop('bot.optimizer', None)

class DummyExchange:
    def __init__(self, volumes):
        self.volumes = volumes
    async def fetch_ticker(self, symbol):
        return {'quoteVolume': self.volumes.get(symbol, 0)}

@pytest.mark.asyncio
async def test_synchronize_and_update_polars(tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path), use_polars=True)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    symbol = 'BTCUSDT'
    ts = pd.Timestamp.now(tz='UTC')
    df = pd.DataFrame({'open':[1],'high':[1],'low':[1],'close':[1],'volume':[1]}, index=[ts])
    df['symbol'] = symbol
    df = df.set_index(['symbol', df.index])
    df.index.set_names(['symbol', 'timestamp'], inplace=True)
    await dh.synchronize_and_update(symbol, df, 0.0, 0.0, {'bids': [], 'asks': []})
    await asyncio.sleep(0)
    assert isinstance(dh._ohlcv, pl.DataFrame)
    assert symbol in dh.indicators
    assert 'ema30' in dh.indicators[symbol].df.columns

@pytest.mark.asyncio
async def test_cleanup_old_data_polars(monkeypatch, tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path), data_cleanup_interval=0, forget_window=1, use_polars=True)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    now = pd.Timestamp.now(tz='UTC')
    old_ts = now - pd.Timedelta(seconds=5)
    df_pl = pl.DataFrame({
        'symbol': ['BTCUSDT', 'BTCUSDT'],
        'timestamp': [old_ts, now],
        'open': [1,1],
        'high': [1,1],
        'low': [1,1],
        'close': [1,1],
        'volume': [1,1],
    })
    dh._ohlcv = df_pl.clone()
    dh._ohlcv_2h = df_pl.clone()

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

    assert dh._ohlcv.height == 1
    assert dh._ohlcv_2h.height == 1

sys.modules['bot.utils'] = real_utils
