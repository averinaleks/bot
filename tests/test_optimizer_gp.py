import sys
import types
import numpy as np
import pandas as pd
import pytest
import logging
from bot.config import BotConfig

# Stub heavy dependencies before importing optimizer
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

# Stub optuna
optuna_mod = types.ModuleType('optuna')
optuna_samplers = types.ModuleType('optuna.samplers')
class _TPESampler:
    def __init__(self, *a, **k):
        pass
optuna_samplers.TPESampler = _TPESampler
optuna_mod.samplers = optuna_samplers
optuna_mod.create_study = lambda *a, **k: types.SimpleNamespace(ask=lambda: types.SimpleNamespace(number=0), best_params={}, best_value=1.0, tell=lambda *a, **k: None)
optuna_exceptions = types.ModuleType('optuna.exceptions')
class ExperimentalWarning(Warning):
    pass
optuna_exceptions.ExperimentalWarning = ExperimentalWarning
optuna_mod.exceptions = optuna_exceptions
sys.modules.setdefault('optuna', optuna_mod)
sys.modules.setdefault('optuna.samplers', optuna_samplers)
sys.modules.setdefault('optuna.exceptions', optuna_exceptions)

# Stub skopt
skopt_mod = types.ModuleType('skopt')
class DummyOpt:
    instantiated = False
    def __init__(self, dims):
        DummyOpt.instantiated = True
        self.space = types.SimpleNamespace(dimensions=dims)
    def ask(self):
        return [10,50,100,5,2,2,0.05,0.5,1.5,0.5,1.5]
    def tell(self, *a, **k):
        pass
skopt_space = types.ModuleType('skopt.space')
class DummyDim:
    def __init__(self, *a, **kw):
        self.name = kw.get('name')
skopt_space.Integer = DummyDim
skopt_space.Real = DummyDim
skopt_mod.Optimizer = DummyOpt
skopt_mod.space = skopt_space
sys.modules['skopt'] = skopt_mod
sys.modules['skopt.space'] = skopt_space

# Import optimizer fresh
sys.modules.pop('optimizer', None)
sys.modules.pop('bot.optimizer', None)
from bot.optimizer import ParameterOptimizer  # noqa: E402
from bot import optimizer

# Stub utils
utils = types.ModuleType('utils')
utils.logger = logging.getLogger('test')
async def _cde(*a, **kw):
    return False
utils.check_dataframe_empty = _cde
utils.check_dataframe_empty_async = _cde
utils.is_cuda_available = lambda: False
sys.modules['utils'] = utils

class DummyDataHandler:
    def __init__(self, df):
        self.usdt_pairs = ['BTCUSDT']
        self.ohlcv = df
        self.telegram_logger = None

def make_df():
    idx = pd.date_range('2020-01-01', periods=30, freq='min')
    df = pd.DataFrame({'close': np.linspace(1,2,len(idx)),
                       'open': np.linspace(1,2,len(idx)),
                       'high': np.linspace(1,2,len(idx))+0.1,
                       'low': np.linspace(1,2,len(idx))-0.1,
                       'volume': np.ones(len(idx))}, index=idx)
    df['symbol'] = 'BTCUSDT'
    return df.set_index(['symbol', df.index])

@pytest.mark.asyncio
async def test_gp_optimizer_selected(monkeypatch):
    df = make_df()
    config = BotConfig(timeframe='1m', optuna_trials=1, optimization_interval=1,
                       volatility_threshold=0.02, optimizer_method='gp',
                       mlflow_enabled=False)
    monkeypatch.setattr(optimizer, '_objective_remote', types.SimpleNamespace(remote=lambda *a, **k: 0.0))
    opt = ParameterOptimizer(config, DummyDataHandler(df))
    await opt.optimize('BTCUSDT')
    assert DummyOpt.instantiated

@pytest.mark.asyncio
async def test_holdout_warning_emitted(monkeypatch, caplog):
    df = make_df()
    config = BotConfig(timeframe='1m', optuna_trials=1, optimization_interval=1,
                       volatility_threshold=0.02, holdout_warning_ratio=0.3,
                       optimizer_method='gp', mlflow_enabled=False)
    def dummy_remote(data, *a, **k):
        return 1.0 if len(data) == len(df) else 0.6
    monkeypatch.setattr(optimizer, '_objective_remote', types.SimpleNamespace(remote=dummy_remote))
    opt = ParameterOptimizer(config, DummyDataHandler(df))
    caplog.set_level(logging.WARNING)
    await opt.optimize('BTCUSDT')
    assert any('hold-out' in rec.getMessage() for rec in caplog.records)

sys.modules.pop('optuna', None)
sys.modules.pop('optuna.samplers', None)
sys.modules.pop('optuna.exceptions', None)
