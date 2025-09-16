import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bot.config import BotConfig

@pytest.fixture(autouse=True)
def _test_mode_env(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    yield
    monkeypatch.delenv("TEST_MODE", raising=False)
gym_mod = types.ModuleType("gymnasium")
gym_mod.Env = object
spaces_mod = types.ModuleType("gymnasium.spaces")

class DummyDiscrete:
    def __init__(self, n):
        self.n = n

spaces_mod.Discrete = DummyDiscrete
class DummyBox:
    def __init__(self, low, high, shape=None, dtype=None):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

spaces_mod.Box = DummyBox

sys.modules.setdefault("gymnasium", gym_mod)
sys.modules.setdefault("gymnasium.spaces", spaces_mod)

class DummyIndicators:
    def __init__(self, length):
        base = np.arange(length, dtype=float)
        self.ema30 = pd.Series(base)
        self.ema100 = pd.Series(base)
        self.ema200 = pd.Series(base)
        self.rsi = pd.Series(base)
        self.adx = pd.Series(base)
        self.macd = pd.Series(base)
        self.atr = pd.Series(base)

class DummyDH:
    def __init__(self, df):
        self.ohlcv = df
        self.usdt_pairs = ["BTCUSDT"]
        self.indicators = {}
        self.funding_rates = {}
        self.open_interest = {}
        self.open_interest_change = {}

class DummyTM:
    pass

def make_df(n=5):
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    df = pd.DataFrame({"close": np.arange(n),
                       "open": np.arange(n),
                       "high": np.arange(n),
                       "low": np.arange(n),
                       "volume": np.arange(n)}, index=idx)
    df["symbol"] = "BTCUSDT"
    df = df.set_index(["symbol", df.index])
    return df

@pytest.mark.asyncio
async def test_precompute_features_caches(monkeypatch, tmp_path):
    from bot.model_builder import ModelBuilder
    df = make_df()
    dh = DummyDH(df)
    cfg = BotConfig(cache_dir=str(tmp_path), lstm_timesteps=2, min_data_length=len(df), nn_framework="tensorflow")
    mb = ModelBuilder(cfg, dh, DummyTM())
    ind = DummyIndicators(len(df))
    dh.indicators["BTCUSDT"] = ind

    feats = np.ones((len(df), 15), dtype=np.float32)

    async def fake_prepare(symbol, indicators):
        assert symbol == "BTCUSDT"
        assert indicators is ind
        return feats

    monkeypatch.setattr(mb, "prepare_lstm_features", fake_prepare)

    await mb.precompute_features("BTCUSDT")
    assert "BTCUSDT" in mb.feature_cache
    assert np.array_equal(mb.feature_cache["BTCUSDT"], feats)


def test_model_builder_cache_fallback(monkeypatch, tmp_path):
    from bot import model_builder

    primary = tmp_path / "primary"
    fallback_root = tmp_path / "fallback_root"
    monkeypatch.setattr(model_builder.tempfile, "gettempdir", lambda: str(fallback_root))

    calls: list[str] = []

    class DummyCache:
        def __init__(self, path: str) -> None:
            self.cache_dir = path

    def fake_cache(path: str) -> DummyCache:
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError("denied")
        Path(path).mkdir(parents=True, exist_ok=True)
        return DummyCache(path)

    monkeypatch.setattr(model_builder, "HistoricalDataCache", fake_cache)

    cfg = BotConfig(cache_dir=str(primary))
    dh = DummyDH(make_df())
    mb = model_builder.ModelBuilder(cfg, dh, DummyTM())

    assert isinstance(mb.cache, DummyCache)
    assert calls[0] == str(primary.resolve())
    expected_fallback = (fallback_root / "model_builder_cache").resolve()
    assert calls[1] == str(expected_fallback)
    assert Path(mb.state_file).parent == expected_fallback
    assert cfg["cache_dir"] == str(expected_fallback)


def test_model_builder_cache_handles_total_failure(monkeypatch, tmp_path, caplog):
    from bot import model_builder

    monkeypatch.setattr(model_builder.tempfile, "gettempdir", lambda: str(tmp_path / "fallback"))

    def boom(_path: str):
        raise PermissionError("no access")

    monkeypatch.setattr(model_builder, "HistoricalDataCache", boom)

    cfg = BotConfig(cache_dir=str(tmp_path / "primary"))
    dh = DummyDH(make_df())
    with caplog.at_level(logging.WARNING):
        mb = model_builder.ModelBuilder(cfg, dh, DummyTM())

    assert mb.cache is None
    expected_dir = (tmp_path / "fallback" / "model_builder_cache").resolve()
    assert Path(mb.state_file).parent == expected_dir
    assert expected_dir.exists()
