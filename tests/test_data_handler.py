import asyncio
import contextlib
import json
import pandas as pd
import pytest

from bot.config import BotConfig
from bot.data_handler import DataHandler, DEFAULT_PRICE
from bot.data_handler.api import api_app


class DummyExchange:
    def __init__(self, volumes):
        self.volumes = volumes

    async def fetch_ticker(self, symbol):
        return {"quoteVolume": self.volumes.get(symbol, 0)}


@pytest.fixture
def cfg_factory(tmp_path):
    def _factory(**kwargs):
        return BotConfig(cache_dir=str(tmp_path), **kwargs)

    return _factory


def _expected_rate(tf: str) -> int:
    sec = pd.Timedelta(tf).total_seconds()
    return max(1, int(1800 / sec))


@pytest.mark.asyncio
async def test_select_liquid_pairs_plain_symbol_included(cfg_factory):
    cfg = cfg_factory(max_symbols=5, min_liquidity=0)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({"BTCUSDT": 1.0}))
    markets = {
        "BTCUSDT": {"active": True, "contract": True, "linear": True, "quote": "USDT"},
        "BTC/USDT": {"active": True, "contract": False, "linear": False, "quote": "USDT"},
    }
    pairs = await dh.select_liquid_pairs(markets)
    assert "BTCUSDT" in pairs


@pytest.mark.asyncio
async def test_select_liquid_pairs_prefers_highest_volume(cfg_factory):
    cfg = cfg_factory(max_symbols=5, min_liquidity=0)
    volumes = {"BTCUSDT": 1.0, "BTC/USDT:USDT": 2.0}
    dh = DataHandler(cfg, None, None, exchange=DummyExchange(volumes))
    markets = {
        "BTCUSDT": {"active": True, "contract": True, "quote": "USDT"},
        "BTC/USDT:USDT": {"active": True, "contract": True, "quote": "USDT"},
    }
    pairs = await dh.select_liquid_pairs(markets)
    assert pairs == ["BTC/USDT:USDT"]


@pytest.mark.asyncio
async def test_select_liquid_pairs_filters_by_min_liquidity(cfg_factory):
    cfg = cfg_factory(max_symbols=5, min_liquidity=100)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({"BTCUSDT": 50}))
    markets = {"BTCUSDT": {"active": True, "contract": True, "quote": "USDT"}}
    with pytest.raises(ValueError):
        await dh.select_liquid_pairs(markets)


@pytest.mark.asyncio
async def test_select_liquid_pairs_filters_new_listings(cfg_factory):
    now = pd.Timestamp.utcnow()
    recent = int((now - pd.Timedelta(minutes=5)).timestamp() * 1000)
    old = int((now - pd.Timedelta(hours=2)).timestamp() * 1000)
    volumes = {"NEWUSDT": 1.0, "OLDUSDT": 1.0}
    cfg = cfg_factory(max_symbols=5, min_liquidity=0, min_data_length=10, timeframe="1m")
    dh = DataHandler(cfg, None, None, exchange=DummyExchange(volumes))
    markets = {
        "NEWUSDT": {
            "active": True,
            "info": {"launchTime": recent},
            "contract": True,
            "quote": "USDT",
        },
        "OLDUSDT": {
            "active": True,
            "info": {"launchTime": old},
            "contract": True,
            "quote": "USDT",
        },
    }
    pairs = await dh.select_liquid_pairs(markets)
    assert "NEWUSDT" not in pairs
    assert "OLDUSDT" in pairs


def test_dynamic_ws_min_process_rate_short_tf(cfg_factory):
    cfg = cfg_factory(timeframe="1m")
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({"BTCUSDT": 1.0}))
    assert dh.ws_min_process_rate == _expected_rate("1m")


def test_dynamic_ws_min_process_rate_long_tf(cfg_factory):
    cfg = cfg_factory(timeframe="2h")
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({"BTCUSDT": 1.0}))
    assert dh.ws_min_process_rate == _expected_rate("2h")


class DummyWS:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    async def recv(self):
        return json.dumps({"success": True})


@pytest.mark.asyncio
async def test_ws_rate_limit_zero_no_exception(cfg_factory):
    cfg = cfg_factory(ws_rate_limit=0)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({"BTCUSDT": 1.0}))
    ws = DummyWS()
    await dh._send_subscriptions(ws, ["BTCUSDT"], "primary")
    assert ws.sent


def test_price_endpoint_returns_default():
    with api_app.test_client() as client:
        resp = client.get("/price/UNKNOWN")
        assert resp.status_code == 200
        assert resp.get_json() == {"price": DEFAULT_PRICE}


@pytest.mark.asyncio
async def test_load_from_disk_buffer_loop(tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path))
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({"BTCUSDT": 1.0}))
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
async def test_cleanup_old_data_pandas(monkeypatch, tmp_path):
    cfg = BotConfig(
        cache_dir=str(tmp_path),
        data_cleanup_interval=0,
        forget_window=10,
        history_retention=5,
    )
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({"BTCUSDT": 1.0}))
    symbol = "BTCUSDT"
    now = pd.Timestamp.now(tz="UTC")
    timestamps = [now - pd.Timedelta(seconds=i) for i in range(14, -1, -1)]
    idx = pd.MultiIndex.from_product([[symbol], timestamps], names=["symbol", "timestamp"])
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx)
    dh._ohlcv = df.copy()
    dh._ohlcv_2h = df.copy()

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    task = asyncio.create_task(dh.cleanup_old_data())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(dh._ohlcv) == cfg.history_retention
    assert len(dh._ohlcv_2h) == cfg.history_retention
    cutoff = now - pd.Timedelta(seconds=cfg.forget_window)
    assert dh._ohlcv.index.get_level_values("timestamp").min() >= cutoff
    assert dh._ohlcv_2h.index.get_level_values("timestamp").min() >= cutoff
