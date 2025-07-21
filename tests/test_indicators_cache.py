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
    idx = pd.date_range("2020-01-01", periods=length, freq="1min")
    return pd.DataFrame(data, index=idx)


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


def test_incremental_update():
    cfg = BotConfig(ema30_period=3, ema100_period=5, ema200_period=7, atr_period_default=3)
    df = make_df(30)
    ind = IndicatorsCache(df, cfg, 0.1)
    prev_ema30 = ind.last_ema30
    prev_ema100 = ind.last_ema100
    prev_ema200 = ind.last_ema200
    prev_atr = ind.last_atr
    prev_close = ind.last_close

    new = pd.DataFrame({
        "close": [2.1],
        "high": [2.2],
        "low": [2.0],
        "volume": [1.0],
    }, index=[df.index[-1] + pd.Timedelta(minutes=1)])
    ind.update(new)

    alpha30 = 2 / (cfg.ema30_period + 1)
    alpha100 = 2 / (cfg.ema100_period + 1)
    alpha200 = 2 / (cfg.ema200_period + 1)
    tr = max(2.2 - 2.0, abs(2.2 - prev_close), abs(2.0 - prev_close))
    expected_ema30 = alpha30 * 2.1 + (1 - alpha30) * prev_ema30
    expected_ema100 = alpha100 * 2.1 + (1 - alpha100) * prev_ema100
    expected_ema200 = alpha200 * 2.1 + (1 - alpha200) * prev_ema200
    expected_atr = (prev_atr * (cfg.atr_period_default - 1) + tr) / cfg.atr_period_default

    assert np.isclose(ind.last_ema30, expected_ema30)
    assert np.isclose(ind.last_ema100, expected_ema100)
    assert np.isclose(ind.last_ema200, expected_ema200)
    assert np.isclose(ind.last_atr, expected_atr)
