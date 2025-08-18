import os
import sys
import numpy as np
import pandas as pd
import pytest
ta = pytest.importorskip("ta")
import types
from bot import utils

import importlib.util

# Stub heavy dependencies before importing data_handler
ccxt_mod = types.ModuleType('ccxt')
ccxt_mod.async_support = types.ModuleType('async_support')
ccxt_mod.async_support.bybit = object
ccxt_mod.pro = types.ModuleType('pro')
ccxt_mod.pro.bybit = object
sys.modules.setdefault('ccxt', ccxt_mod)
sys.modules.setdefault('ccxt.async_support', ccxt_mod.async_support)
sys.modules.setdefault('ccxt.pro', ccxt_mod.pro)

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

from bot.data_handler import ema_fast, atr_fast  # noqa: E402

if optimizer_stubbed:
    sys.modules.pop('optimizer', None)
    sys.modules.pop('bot.optimizer', None)


def test_ema_fast_matches_ta():
    values = np.linspace(1, 50, 50)
    result_fast = ema_fast(values, 10)
    expected = ta.trend.ema_indicator(pd.Series(values), window=10, fillna=True).to_numpy()
    assert np.allclose(result_fast, expected, atol=1e-6)


def test_atr_fast_matches_ta():
    rng = np.random.default_rng(0)
    high = pd.Series(rng.random(60) + 10)
    low = high - rng.random(60)
    close = high - rng.random(60)
    result_fast = atr_fast(high.to_numpy(), low.to_numpy(), close.to_numpy(), 14)
    expected = ta.volatility.average_true_range(high, low, close, window=14, fillna=True).to_numpy()
    assert np.allclose(result_fast, expected, atol=1e-6)


def test_filter_outliers_zscore_handles_nans(monkeypatch):
    def simple_z(a):
        a = np.asarray(a, dtype=float)
        return (a - a.mean()) / a.std()

    utils.filter_outliers_zscore.__globals__["zscore"] = simple_z

    # Ensure scipy does not fail when a torch stub without Tensor exists
    if "torch" in sys.modules and not hasattr(sys.modules["torch"], "Tensor"):
        sys.modules["torch"].Tensor = object

    df = pd.DataFrame({"close": [1.0, 2.0, np.nan, 4.0, 5.0]})
    result = utils.filter_outliers_zscore(df, "close", threshold=3.0)
    assert len(result) == len(df)
    assert result["close"].isna().sum() == 1

def test_filter_outliers_zscore_mask_length(monkeypatch):
    def simple_z(a):
        a = np.asarray(a, dtype=float)
        return (a - a.mean()) / a.std()

    utils.filter_outliers_zscore.__globals__["zscore"] = simple_z

    df = pd.DataFrame({"close": [np.nan, 1.0, 2.0, 3.0, np.nan]})
    result = utils.filter_outliers_zscore(df, "close", threshold=2.0)
    assert len(result) == len(df)
    assert result["close"].isna().sum() == 2
