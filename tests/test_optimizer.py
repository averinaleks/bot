import os
import sys

import pytest
import types
import logging
from bot.config import BotConfig

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
optuna_samplers.TPESampler = object
optuna_mod.samplers = optuna_samplers
optuna_mod.create_study = lambda *a, **k: types.SimpleNamespace(optimize=lambda *a, **k: None, best_params={})
optuna_exceptions = types.ModuleType('optuna.exceptions')
class ExperimentalWarning(Warning):
    pass
optuna_exceptions.ExperimentalWarning = ExperimentalWarning
optuna_mod.exceptions = optuna_exceptions
sys.modules.setdefault('optuna', optuna_mod)
sys.modules.setdefault('optuna.samplers', optuna_samplers)
sys.modules.setdefault('optuna.exceptions', optuna_exceptions)


sys.modules.pop("optimizer", None)
sys.modules.pop("bot.optimizer", None)
from bot.optimizer import ParameterOptimizer  # noqa: E402


utils = types.ModuleType('utils')
utils.logger = logging.getLogger('test')
async def _cde(*a, **kw):
    return False
utils.check_dataframe_empty = _cde
utils.check_dataframe_empty_async = _cde
utils.is_cuda_available = lambda: False
sys.modules['utils'] = utils

scipy_mod = types.ModuleType('scipy')
stats_mod = types.ModuleType('scipy.stats')
stats_mod.zscore = lambda a, axis=0: (a - a.mean()) / a.std()
scipy_mod.__version__ = "1.0"
scipy_mod.stats = stats_mod
sys.modules.setdefault('scipy', scipy_mod)
sys.modules.setdefault('scipy.stats', stats_mod)

class DummyDataHandler:
    def __init__(self):
        self.usdt_pairs = ["BTCUSDT"]

data_handler = DummyDataHandler()

config = BotConfig(optimization_interval=7200, volatility_threshold=0.02)

opt = ParameterOptimizer(config, data_handler)

@pytest.mark.parametrize("vol,expected", [
    (0.0, opt.base_optimization_interval),
    (0.01, opt.base_optimization_interval / (1 + 0.01 / opt.volatility_threshold)),
    (0.05, 1800)
])
def test_get_opt_interval(vol, expected):
    interval = opt.get_opt_interval("BTCUSDT", vol)
    max_int = max(1800, min(opt.base_optimization_interval * 2, expected))
    assert interval == pytest.approx(max_int)


def test_get_opt_interval_zero_threshold():
    cfg = BotConfig(optimization_interval=7200, volatility_threshold=0)
    opt_zero = ParameterOptimizer(cfg, data_handler)
    interval = opt_zero.get_opt_interval("BTCUSDT", 0.01)
    assert interval >= 1800

sys.modules.pop('optuna', None)
sys.modules.pop('optuna.exceptions', None)
sys.modules.pop('optuna.samplers', None)
sys.modules.pop('utils', None)
