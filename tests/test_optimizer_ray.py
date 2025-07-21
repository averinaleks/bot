import os
import sys
import importlib
import types
import numpy as np
import pandas as pd
import pytest
import optuna  # noqa: F401
from config import BotConfig

# Stub heavy dependencies before importing the optimizer
if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    import importlib.machinery
    torch.__spec__ = importlib.machinery.ModuleSpec('torch', None)
    sys.modules['torch'] = torch


sk_mod = types.ModuleType('sklearn')
model_sel = types.ModuleType('sklearn.model_selection')
model_sel.GridSearchCV = object
sk_mod.model_selection = model_sel
base_estimator = types.ModuleType('sklearn.base')
base_estimator.BaseEstimator = object
sk_mod.base = base_estimator
sys.modules.setdefault('sklearn', sk_mod)
sys.modules.setdefault('sklearn.model_selection', model_sel)
sys.modules.setdefault('sklearn.base', base_estimator)
mlflow_mod = types.ModuleType('optuna.integration.mlflow')
mlflow_mod.MLflowCallback = object
sys.modules.setdefault('optuna.integration.mlflow', mlflow_mod)
optuna_mod = types.ModuleType('optuna')
optuna_samplers = types.ModuleType('optuna.samplers')
class _TPESampler:
    def __init__(self, *a, **k):
        pass
optuna_samplers.TPESampler = _TPESampler
optuna_mod.samplers = optuna_samplers
sys.modules.setdefault('optuna', optuna_mod)
sys.modules.setdefault('optuna.samplers', optuna_samplers)

scipy_mod = types.ModuleType('scipy')
stats_mod = types.ModuleType('scipy.stats')
stats_mod.zscore = lambda a, axis=0: (a - a.mean()) / a.std()
scipy_mod.__version__ = "1.0"
scipy_mod.stats = stats_mod
sys.modules.setdefault('scipy', scipy_mod)
sys.modules.setdefault('scipy.stats', stats_mod)

import builtins
from optuna.exceptions import ExperimentalWarning as _OptunaExperimentalWarning
builtins.ExperimentalWarning = _OptunaExperimentalWarning

sys.modules.pop('optimizer', None)
from optimizer import ParameterOptimizer  # noqa: E402
import optimizer


# ensure real optuna




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


def make_high_vol_df():
    idx = pd.date_range('2020-01-01', periods=30, freq='min')
    np.random.seed(0)
    close = 1 + np.random.randn(len(idx)) * 0.1
    df = pd.DataFrame({
        'close': close,
        'open': close,
        'high': close + 0.1,
        'low': close - 0.1,
        'volume': np.ones(len(idx)),
    }, index=idx)
    df['symbol'] = 'BTCUSDT'
    return df.set_index(['symbol', df.index])


@pytest.mark.parametrize('df_builder', [make_df, make_high_vol_df])
@pytest.mark.filterwarnings("ignore:.*multivariate.*:ExperimentalWarning")
@pytest.mark.asyncio
async def test_optimize_returns_params(df_builder):
    df = df_builder()
    config = BotConfig(
        timeframe='1m',
        optuna_trials=1,
        optimization_interval=1,
        volatility_threshold=0.02,
        ema30_period=30,
        ema100_period=100,
        ema200_period=200,
        atr_period_default=14,
        tp_multiplier=2.0,
        sl_multiplier=1.0,
        base_probability_threshold=0.5,
        loss_streak_threshold=2,
        win_streak_threshold=2,
        threshold_adjustment=0.05,
        mlflow_enabled=False,
    )
    opt = ParameterOptimizer(config, DummyDataHandler(df))
    params = await opt.optimize('BTCUSDT')
    assert isinstance(params, dict)
    assert 'ema30_period' in params
    for key in [
        'loss_streak_threshold',
        'win_streak_threshold',
        'threshold_adjustment',
        'risk_sharpe_loss_factor',
        'risk_sharpe_win_factor',
        'risk_vol_min',
        'risk_vol_max',
    ]:
        assert key in params


@pytest.mark.filterwarnings("ignore:.*multivariate.*:ExperimentalWarning")
@pytest.mark.asyncio
async def test_optimize_zero_vol_threshold():
    df = make_df()
    config = BotConfig(
        timeframe='1m',
        optuna_trials=1,
        optimization_interval=1,
        volatility_threshold=0,
        ema30_period=30,
        ema100_period=100,
        ema200_period=200,
        atr_period_default=14,
        tp_multiplier=2.0,
        sl_multiplier=1.0,
        base_probability_threshold=0.5,
        loss_streak_threshold=2,
        win_streak_threshold=2,
        threshold_adjustment=0.05,
        mlflow_enabled=False,
    )
    opt = ParameterOptimizer(config, DummyDataHandler(df))
    params = await opt.optimize('BTCUSDT')
    assert isinstance(params, dict)
    assert 'ema30_period' in params


@pytest.mark.filterwarnings("ignore:.*multivariate.*:ExperimentalWarning")
@pytest.mark.asyncio
async def test_get_opt_interval_called(monkeypatch):
    df = make_high_vol_df()
    config = BotConfig(
        timeframe='1m',
        optuna_trials=1,
        optimization_interval=1,
        volatility_threshold=0.02,
        ema30_period=30,
        ema100_period=100,
        ema200_period=200,
        atr_period_default=14,
        tp_multiplier=2.0,
        sl_multiplier=1.0,
        base_probability_threshold=0.5,
        loss_streak_threshold=2,
        win_streak_threshold=2,
        threshold_adjustment=0.05,
        mlflow_enabled=False,
    )
    opt = ParameterOptimizer(config, DummyDataHandler(df))
    captured = {}

    orig = opt.get_opt_interval

    def spy(symbol, vol):
        captured['args'] = (symbol, vol)
        return orig(symbol, vol)

    monkeypatch.setattr(opt, 'get_opt_interval', spy)
    await opt.optimize('BTCUSDT')
    expected = df['close'].pct_change().std()
    assert captured['args'][0] == 'BTCUSDT'
    assert captured['args'][1] == pytest.approx(expected)


@pytest.mark.filterwarnings("ignore:.*multivariate.*:ExperimentalWarning")
@pytest.mark.asyncio
async def test_custom_n_splits(monkeypatch):
    df = make_df()
    config = BotConfig(
        timeframe='1m',
        optuna_trials=1,
        optimization_interval=1,
        volatility_threshold=0.02,
        n_splits=7,
        ema30_period=30,
        ema100_period=100,
        ema200_period=200,
        atr_period_default=14,
        tp_multiplier=2.0,
        sl_multiplier=1.0,
        base_probability_threshold=0.5,
        loss_streak_threshold=2,
        win_streak_threshold=2,
        threshold_adjustment=0.05,
        mlflow_enabled=False,
    )
    captured = {}

    def dummy_remote(*args, **kwargs):
        captured['val'] = args[-1]
        return 0.0

    monkeypatch.setattr(optimizer._objective_remote, 'remote', dummy_remote)
    opt = ParameterOptimizer(config, DummyDataHandler(df))
    await opt.optimize('BTCUSDT')
    assert captured['val'] == config.n_splits

@pytest.mark.filterwarnings("ignore:.*multivariate.*:ExperimentalWarning")
@pytest.mark.asyncio
async def test_custom_n_splits_three(monkeypatch):
    df = make_df()
    config = BotConfig(
        timeframe='1m',
        optuna_trials=1,
        optimization_interval=1,
        volatility_threshold=0.02,
        n_splits=3,
        ema30_period=30,
        ema100_period=100,
        ema200_period=200,
        atr_period_default=14,
        tp_multiplier=2.0,
        sl_multiplier=1.0,
        base_probability_threshold=0.5,
        loss_streak_threshold=2,
        win_streak_threshold=2,
        threshold_adjustment=0.05,
        mlflow_enabled=False,
    )
    captured = {}

    def dummy_remote(*args, **kwargs):
        captured['val'] = args[-1]
        return 0.0

    monkeypatch.setattr(optimizer._objective_remote, 'remote', dummy_remote)
    opt = ParameterOptimizer(config, DummyDataHandler(df))
    await opt.optimize('BTCUSDT')
    assert captured['val'] == config.n_splits
