import asyncio
import pandas as pd
import polars as pl
import contextlib
import pytest
from bot.config import BotConfig
from bot.data_handler import DataHandler


@pytest.fixture(autouse=True)
def _set_test_mode(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    yield

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
    monkeypatch.setattr(asyncio, 'sleep', fast_sleep)

    task = asyncio.create_task(dh.cleanup_old_data())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert dh._ohlcv.height == 1
    assert dh._ohlcv_2h.height == 1
