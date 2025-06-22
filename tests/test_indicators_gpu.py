import os, sys; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import ta
import types

numba_mod = types.ModuleType('numba')
numba_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
numba_mod.jit = lambda *a, **k: (lambda f: f)
numba_mod.prange = range
sys.modules.setdefault('numba', numba_mod)
sys.modules.setdefault('numba.cuda', numba_mod.cuda)

ccxt_mod = types.ModuleType('ccxt')
ccxt_mod.async_support = types.ModuleType('async_support')
ccxt_mod.async_support.bybit = object
sys.modules.setdefault('ccxt', ccxt_mod)
sys.modules.setdefault('ccxt.async_support', ccxt_mod.async_support)
sys.modules.setdefault('websockets', types.ModuleType('websockets'))
ray_mod = types.ModuleType('ray')
ray_mod.remote = lambda *a, **k: (lambda f: f)
sys.modules.setdefault('ray', ray_mod)
tenacity_mod = types.ModuleType('tenacity')
tenacity_mod.retry = lambda *a, **k: (lambda f: f)
tenacity_mod.wait_exponential = lambda *a, **k: None
sys.modules['tenacity'] = tenacity_mod
psutil_mod = types.ModuleType('psutil')
psutil_mod.cpu_percent = lambda interval=1: 0
psutil_mod.virtual_memory = lambda: type('mem', (), {'percent': 0})
sys.modules.setdefault('psutil', psutil_mod)
scipy_mod = types.ModuleType('scipy')
stats_mod = types.ModuleType('scipy.stats')
stats_mod.zscore = lambda a, axis=0: (a - np.mean(a, axis=axis)) / np.std(a, axis=axis)
scipy_mod.__version__ = "1.0"
scipy_mod.stats = stats_mod
sys.modules.setdefault('scipy', scipy_mod)
sys.modules.setdefault('scipy.stats', stats_mod)
sys.modules.setdefault('httpx', types.ModuleType('httpx'))
telegram_error_mod = types.ModuleType('telegram.error')
telegram_error_mod.RetryAfter = Exception
sys.modules.setdefault('telegram', types.ModuleType('telegram'))
sys.modules.setdefault('telegram.error', telegram_error_mod)

from data_handler import ema_fast, atr_fast


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
