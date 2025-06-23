import os
import sys

import pytest
import types
import logging

# Stub heavy dependencies before importing the optimizer
if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    import importlib.machinery
    torch.__spec__ = importlib.machinery.ModuleSpec('torch', None)
    sys.modules['torch'] = torch

ray_mod = types.ModuleType('ray')
ray_mod.remote = lambda *a, **k: (lambda f: f)
ray_mod.get = lambda x: x
sys.modules.setdefault('ray', ray_mod)

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
sys.modules.setdefault('optuna', optuna_mod)
sys.modules.setdefault('optuna.samplers', optuna_samplers)
sys.modules.setdefault('httpx', types.ModuleType('httpx'))
telegram_error_mod = types.ModuleType('telegram.error')
telegram_error_mod.RetryAfter = Exception
sys.modules.setdefault('telegram', types.ModuleType('telegram'))
sys.modules.setdefault('telegram.error', telegram_error_mod)
psutil_mod = types.ModuleType('psutil')
psutil_mod.cpu_percent = lambda interval=1: 0
psutil_mod.virtual_memory = lambda: type('mem', (), {'percent': 0})
sys.modules.setdefault('psutil', psutil_mod)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from optimizer import ParameterOptimizer  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

utils = types.ModuleType('utils')
utils.logger = logging.getLogger('test')
async def _cde(*a, **kw):
    return False
utils.check_dataframe_empty = _cde
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

config = {"optimization_interval": 7200, "volatility_threshold": 0.02}

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

