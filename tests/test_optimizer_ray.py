import os
import sys
import importlib
import types
import numpy as np
import pandas as pd
import pytest
import optuna  # noqa: F401
from optimizer import ParameterOptimizer

if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    import importlib.machinery
    torch.__spec__ = importlib.machinery.ModuleSpec('torch', None)
    sys.modules['torch'] = torch

numba_mod = types.ModuleType('numba')
numba_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
numba_mod.jit = lambda *a, **k: (lambda f: f)
numba_mod.prange = range
sys.modules.setdefault('numba', numba_mod)
sys.modules.setdefault('numba.cuda', numba_mod.cuda)

# ensure real optuna
if 'optuna' in sys.modules and not hasattr(sys.modules['optuna'], 'create_study'):
    del sys.modules['optuna']
    sys.modules.pop('optuna.samplers', None)
    sys.modules.pop('optuna.integration.mlflow', None)
mlflow_mod = types.ModuleType('optuna.integration.mlflow')
mlflow_mod.MLflowCallback = object
sys.modules.setdefault('optuna.integration.mlflow', mlflow_mod)

# stub ray
ray_mod = types.ModuleType('ray')

def _remote(*args, **kwargs):
    def decorator(func):
        def call(*a, **k):
            return func(*a, **k)
        return types.SimpleNamespace(remote=call, _function=func)
    return decorator

ray_mod.remote = _remote
ray_mod.get = lambda x: x
sys.modules['ray'] = ray_mod

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class DummyDataHandler:
    def __init__(self, df):
        self.ohlcv = df
        self.usdt_pairs = ['BTCUSDT']
        self.indicators_cache = {}

def make_df():
    idx = pd.date_range('2020-01-01', periods=30, freq='min')
    df = pd.DataFrame({
        'close': np.linspace(1, 2, len(idx)),
        'open': np.linspace(1, 2, len(idx)),
        'high': np.linspace(1, 2, len(idx)) + 0.1,
        'low': np.linspace(1, 2, len(idx)) - 0.1,
        'volume': np.ones(len(idx)),
    }, index=idx)
    df['symbol'] = 'BTCUSDT'
    return df.set_index(['symbol', df.index])


@pytest.mark.asyncio
async def test_optimize_returns_params():
    df = make_df()
    config = {
        'timeframe': '1m',
        'optuna_trials': 1,
        'optimization_interval': 1,
        'volatility_threshold': 0.02,
        'ema30_period': 30,
        'ema100_period': 100,
        'ema200_period': 200,
        'atr_period_default': 14,
        'tp_multiplier': 2.0,
        'sl_multiplier': 1.0,
        'base_probability_threshold': 0.5,
        'loss_streak_threshold': 2,
        'win_streak_threshold': 2,
        'threshold_adjustment': 0.05,
        'mlflow_enabled': False,
    }
    opt = ParameterOptimizer(config, DummyDataHandler(df))
    params = await opt.optimize('BTCUSDT')
    assert isinstance(params, dict)
    assert 'ema30_period' in params
