import sys
import importlib
import types
import numpy as np
import pandas as pd
import pytest
import logging

from bot.config import BotConfig


@pytest.fixture
def _stub_modules(monkeypatch):
    """Inject lightweight stubs for heavy optional dependencies."""
    # torch
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    # scikit-learn
    sk_mod = types.ModuleType("sklearn")
    model_sel = types.ModuleType("sklearn.model_selection")
    model_sel.GridSearchCV = object
    sk_mod.model_selection = model_sel
    base_estimator = types.ModuleType("sklearn.base")
    base_estimator.BaseEstimator = object
    sk_mod.base = base_estimator
    monkeypatch.setitem(sys.modules, "sklearn", sk_mod)
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", model_sel)
    monkeypatch.setitem(sys.modules, "sklearn.base", base_estimator)

    # optuna
    mlflow_mod = types.ModuleType("optuna.integration.mlflow")
    mlflow_mod.MLflowCallback = object
    monkeypatch.setitem(sys.modules, "optuna.integration.mlflow", mlflow_mod)

    optuna_mod = types.ModuleType("optuna")
    optuna_samplers = types.ModuleType("optuna.samplers")

    class _TPESampler:
        def __init__(self, *a, **k):
            pass

    optuna_samplers.TPESampler = _TPESampler
    optuna_mod.samplers = optuna_samplers
    optuna_mod.create_study = lambda *a, **k: types.SimpleNamespace(
        ask=lambda: types.SimpleNamespace(number=0),
        best_params={},
        best_value=1.0,
        tell=lambda *a, **k: None,
    )
    optuna_exceptions = types.ModuleType("optuna.exceptions")

    class ExperimentalWarning(Warning):
        pass

    optuna_exceptions.ExperimentalWarning = ExperimentalWarning
    optuna_mod.exceptions = optuna_exceptions
    monkeypatch.setitem(sys.modules, "optuna", optuna_mod)
    monkeypatch.setitem(sys.modules, "optuna.samplers", optuna_samplers)
    monkeypatch.setitem(sys.modules, "optuna.exceptions", optuna_exceptions)

    # skopt
    skopt_mod = types.ModuleType("skopt")

    class DummyOpt:
        instantiated = False

        def __init__(self, dims):
            DummyOpt.instantiated = True
            self.space = types.SimpleNamespace(dimensions=dims)

        def ask(self):
            return [10, 50, 100, 5, 2, 2, 0.05, 0.5, 1.5, 0.5, 1.5]

        def tell(self, *a, **k):
            pass

    skopt_space = types.ModuleType("skopt.space")

    class DummyDim:
        def __init__(self, *a, **kw):
            self.name = kw.get("name")

    skopt_space.Integer = DummyDim
    skopt_space.Real = DummyDim
    skopt_mod.Optimizer = DummyOpt
    skopt_mod.space = skopt_space
    monkeypatch.setitem(sys.modules, "skopt", skopt_mod)
    monkeypatch.setitem(sys.modules, "skopt.space", skopt_space)

    # utils stub
    utils = types.ModuleType("utils")
    utils.logger = logging.getLogger("test")

    async def _cde(*a, **kw):
        return False

    utils.check_dataframe_empty = _cde
    utils.check_dataframe_empty_async = _cde
    utils.is_cuda_available = lambda: False
    monkeypatch.setitem(sys.modules, "utils", utils)

    # Import optimizer after stubs
    sys.modules.pop("optimizer", None)
    sys.modules.pop("bot.optimizer", None)
    from bot.optimizer import ParameterOptimizer  # noqa: E402
    from bot import optimizer  # noqa: E402

    yield ParameterOptimizer, optimizer, DummyOpt

    monkeypatch.undo()
    sys.modules.pop("optimizer", None)
    sys.modules.pop("bot.optimizer", None)

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
async def test_gp_optimizer_selected(monkeypatch, _stub_modules):
    ParameterOptimizer, optimizer, DummyOpt = _stub_modules
    df = make_df()
    config = BotConfig(timeframe='1m', optuna_trials=1, optimization_interval=1,
                       volatility_threshold=0.02, optimizer_method='gp',
                       mlflow_enabled=False)
    monkeypatch.setattr(optimizer, '_objective_remote', types.SimpleNamespace(remote=lambda *a, **k: 0.0))
    opt = ParameterOptimizer(config, DummyDataHandler(df))
    await opt.optimize('BTCUSDT')
    assert DummyOpt.instantiated

@pytest.mark.asyncio
async def test_holdout_warning_emitted(monkeypatch, caplog, _stub_modules):
    ParameterOptimizer, optimizer, DummyOpt = _stub_modules
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
