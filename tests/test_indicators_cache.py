import sys
import types
import pandas as pd
import numpy as np
from config import BotConfig

optimizer_stubbed = False
if 'optimizer' not in sys.modules:
    optimizer_stubbed = True
    optimizer_stub = types.ModuleType('optimizer')
    class _PO:
        def __init__(self, *a, **k):
            pass
    optimizer_stub.ParameterOptimizer = _PO
    sys.modules['optimizer'] = optimizer_stub

from data_handler import IndicatorsCache

if optimizer_stubbed:
    sys.modules.pop('optimizer', None)


def make_df(length=30):
    data = {
        "close": np.linspace(1, 2, length),
        "high": np.linspace(1, 2, length) + 0.1,
        "low": np.linspace(1, 2, length) - 0.1,
        "volume": np.ones(length),
    }
    return pd.DataFrame(data)


def test_interval_from_config():
    cfg = BotConfig(volume_profile_update_interval=7)
    df = make_df(30)
    ind = IndicatorsCache(df, cfg, 0.1)
    assert ind.volume_profile_update_interval == 7


def test_volume_profile_respects_interval():
    cfg = BotConfig(volume_profile_update_interval=3)
    df = make_df(30)
    ind = IndicatorsCache(df, cfg, 0.1)
    assert ind.volume_profile is not None

    cfg = BotConfig(volume_profile_update_interval=100)
    df = make_df(30)
    ind = IndicatorsCache(df, cfg, 0.1)
    assert ind.volume_profile is None
