import sys
import importlib
import types
import numpy as np
import pandas as pd
import pytest
import builtins

from bot.config import BotConfig


class ExperimentalWarning(Warning):
    pass

_prev_expwarn = getattr(builtins, "ExperimentalWarning", None)
builtins.ExperimentalWarning = ExperimentalWarning


@pytest.fixture(autouse=True)
def _restore_experimental_warning():
    yield
    if _prev_expwarn is None:
        if hasattr(builtins, "ExperimentalWarning"):
            delattr(builtins, "ExperimentalWarning")
    else:
        builtins.ExperimentalWarning = _prev_expwarn


@pytest.fixture
def _stub_modules(monkeypatch):
    """Provide lightweight stand-ins for heavy optional dependencies."""
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

    # optuna stubs
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

    def _create_study(*a, **k):
        class _Trial:
            def __init__(self, number: int):
                self.number = number
                self.params = {}

            def suggest_int(self, name, low, high):
                self.params[name] = low
                return low

            def suggest_float(self, name, low, high):
                self.params[name] = low
                return low

        class _Study:
            def __init__(self):
                self.trials = []
                self.best_params = {}
                self.best_value = 0.0

            def ask(self):
                trial = _Trial(len(self.trials))
                self.trials.append(trial)
                return trial

            def tell(self, trial, value):
                if value is not None and value > self.best_value:
                    self.best_value = value
                    self.best_params = getattr(trial, "params", {})

            def optimize(self, *a, **k):
                pass

        return _Study()

    optuna_mod.create_study = _create_study
    optuna_exceptions = types.ModuleType("optuna.exceptions")
    optuna_exceptions.ExperimentalWarning = ExperimentalWarning
    monkeypatch.setitem(sys.modules, "optuna", optuna_mod)
    monkeypatch.setitem(sys.modules, "optuna.samplers", optuna_samplers)
    monkeypatch.setitem(sys.modules, "optuna.exceptions", optuna_exceptions)

    # scipy
    scipy_mod = types.ModuleType("scipy")
    stats_mod = types.ModuleType("scipy.stats")
    stats_mod.zscore = lambda a, axis=0: (a - a.mean()) / a.std()
    scipy_mod.__version__ = "1.0"
    scipy_mod.stats = stats_mod
    monkeypatch.setitem(sys.modules, "scipy", scipy_mod)
    monkeypatch.setitem(sys.modules, "scipy.stats", stats_mod)

    # minimal data_handler
    dh_mod = types.ModuleType("bot.data_handler")

    class IndicatorsCache:
        def __init__(self, *a, **k):
            pass

    dh_mod.IndicatorsCache = IndicatorsCache
    monkeypatch.setitem(sys.modules, "bot.data_handler", dh_mod)
    monkeypatch.setitem(sys.modules, "data_handler", dh_mod)

    # Import optimizer after stubs are in place
    sys.modules.pop("optimizer", None)
    sys.modules.pop("bot.optimizer", None)
    from bot.optimizer import ParameterOptimizer  # noqa: E402
    from bot import optimizer  # noqa: E402

    yield ParameterOptimizer, optimizer

    monkeypatch.undo()
    sys.modules.pop("optimizer", None)
    sys.modules.pop("bot.optimizer", None)


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
async def test_optimize_returns_params(df_builder, _stub_modules):
    ParameterOptimizer, _ = _stub_modules
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
async def test_optimize_zero_vol_threshold(_stub_modules):
    ParameterOptimizer, _ = _stub_modules
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
async def test_get_opt_interval_called(monkeypatch, _stub_modules):
    ParameterOptimizer, _ = _stub_modules
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
async def test_custom_n_splits(monkeypatch, _stub_modules):
    ParameterOptimizer, optimizer = _stub_modules
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
async def test_custom_n_splits_three(monkeypatch, _stub_modules):
    ParameterOptimizer, optimizer = _stub_modules
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
