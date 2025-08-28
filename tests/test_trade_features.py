import importlib
import tempfile
import types
import pandas as pd
import numpy as np
import pytest
from bot.config import BotConfig


class DummyDataHandler:
    def __init__(self):
        self.exchange = types.SimpleNamespace()
        self.usdt_pairs = ["BTCUSDT"]
        idx = pd.MultiIndex.from_tuples(
            [("BTCUSDT", pd.Timestamp("2020-01-01", tz="UTC"))],
            names=["symbol", "timestamp"],
        )
        self.ohlcv = pd.DataFrame({"close": [100.0], "atr": [1.0]}, index=idx)
        self.indicators = {
            "BTCUSDT": types.SimpleNamespace(atr=pd.Series([1.0]))
        }

    async def get_atr(self, symbol: str) -> float:
        return 1.0

    async def is_data_fresh(self, symbol: str, timeframe: str = "primary", max_delay: float = 60) -> bool:  # noqa: D401
        return True


@pytest.fixture
def trade_manager(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    import bot.test_stubs as stubs
    importlib.reload(stubs)
    stubs.apply()
    tm_module = importlib.reload(importlib.import_module("bot.trade_manager"))
    cfg = BotConfig(
        cache_dir=tempfile.mkdtemp(),
        sl_multiplier=1.0,
        tp_multiplier=2.0,
        trailing_stop_multiplier=0.5,
        trailing_stop_percentage=0.5,
        trailing_stop_coeff=0.0,
    )
    tm = tm_module.TradeManager(cfg, DummyDataHandler(), None, None, None)

    async def fake_size(*_a, **_k) -> float:
        return 1.0

    async def fake_order(*_a, **_k):
        return {"retCode": 0, "id": "1"}

    monkeypatch.setattr(tm, "calculate_position_size", fake_size)
    monkeypatch.setattr(tm, "place_order", fake_order)
    tm.save_state = lambda: None
    return tm


@pytest.fixture
def atr_fast_func(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    import bot.test_stubs as stubs
    importlib.reload(stubs)
    stubs.apply()
    dh_module = importlib.reload(importlib.import_module("bot.data_handler"))
    return dh_module.atr_fast


def test_atr_calculation(atr_fast_func):
    high = np.array([10, 11, 12], dtype=float)
    low = np.array([9, 8, 9], dtype=float)
    close = np.array([9.5, 10, 11], dtype=float)
    result = atr_fast_func(high, low, close, 2)
    assert np.allclose(result, np.array([1.0, 2.0, 3.0]))


@pytest.mark.asyncio
async def test_open_and_close_position(trade_manager):
    tm = trade_manager
    await tm.open_position("BTCUSDT", "buy", 100.0, {})
    assert tm._has_position("BTCUSDT")
    await tm.close_position("BTCUSDT", 101.0, "manual")
    assert not tm._has_position("BTCUSDT")


@pytest.mark.asyncio
async def test_stop_loss_closes_position(trade_manager):
    tm = trade_manager
    await tm.open_position("BTCUSDT", "buy", 100.0, {})
    await tm.check_stop_loss_take_profit("BTCUSDT", 99.0)
    assert not tm._has_position("BTCUSDT")


@pytest.mark.asyncio
async def test_trailing_stop(trade_manager):
    tm = trade_manager
    await tm.open_position("BTCUSDT", "buy", 100.0, {})
    await tm.check_trailing_stop("BTCUSDT", 101.0)
    pos = tm.positions.xs("BTCUSDT", level="symbol")
    assert pos["breakeven_triggered"].iloc[0]
    await tm.check_trailing_stop("BTCUSDT", 99.0)
    assert not tm._has_position("BTCUSDT")
