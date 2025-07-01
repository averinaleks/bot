import os
import sys
import numpy as np
import pandas as pd
import ta
import types

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

from data_handler import ema_fast, atr_fast  # noqa: E402


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
