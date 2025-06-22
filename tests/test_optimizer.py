import os, sys; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pytest
import types, sys, logging

utils = types.ModuleType('utils')
utils.logger = logging.getLogger('test')
async def _cde(*a, **kw):
    return False
utils.check_dataframe_empty = _cde
sys.modules['utils'] = utils

import types
mlflow_mod = types.ModuleType('optuna.integration.mlflow')
mlflow_mod.MLflowCallback = object
sys.modules['optuna.integration.mlflow'] = mlflow_mod

from optimizer import ParameterOptimizer

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

