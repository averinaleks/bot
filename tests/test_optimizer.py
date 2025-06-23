import os
import sys

import pytest
import types
import logging
from optimizer import ParameterOptimizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    import importlib.machinery
    torch.__spec__ = importlib.machinery.ModuleSpec('torch', None)
    sys.modules['torch'] = torch

utils = types.ModuleType('utils')
utils.logger = logging.getLogger('test')
async def _cde(*a, **kw):
    return False
utils.check_dataframe_empty = _cde
sys.modules['utils'] = utils
mlflow_mod = types.ModuleType('optuna.integration.mlflow')
mlflow_mod.MLflowCallback = object
sys.modules['optuna.integration.mlflow'] = mlflow_mod
optuna_mod = types.ModuleType('optuna')
optuna_samplers = types.ModuleType('optuna.samplers')
optuna_samplers.TPESampler = object
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
sys.modules.setdefault('httpx', types.ModuleType('httpx'))
telegram_error_mod = types.ModuleType('telegram.error')
telegram_error_mod.RetryAfter = Exception
sys.modules.setdefault('telegram', types.ModuleType('telegram'))
sys.modules.setdefault('telegram.error', telegram_error_mod)
psutil_mod = types.ModuleType('psutil')
psutil_mod.cpu_percent = lambda interval=1: 0
psutil_mod.virtual_memory = lambda: type('mem', (), {'percent': 0})
sys.modules.setdefault('psutil', psutil_mod)

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

