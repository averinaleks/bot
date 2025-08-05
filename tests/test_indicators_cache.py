import sys
import types
import pandas as pd
import numpy as np
import pytest
ta = pytest.importorskip("ta")
from bot.config import BotConfig

optimizer_stubbed = False
if 'optimizer' not in sys.modules and 'bot.optimizer' not in sys.modules:
    optimizer_stubbed = True
    optimizer_stub = types.ModuleType('optimizer')
    class _PO:
        def __init__(self, *a, **k):
            pass
    optimizer_stub.ParameterOptimizer = _PO
    sys.modules['optimizer'] = optimizer_stub
    sys.modules['bot.optimizer'] = optimizer_stub

strategy_stubbed = False
if 'strategy_optimizer' not in sys.modules and 'bot.strategy_optimizer' not in sys.modules:
    strategy_stubbed = True
    strategy_stub = types.ModuleType('strategy_optimizer')
    class _SO:
        def __init__(self, *a, **k):
            pass
    strategy_stub.StrategyOptimizer = _SO
    sys.modules['strategy_optimizer'] = strategy_stub
    sys.modules['bot.strategy_optimizer'] = strategy_stub

from bot.data_handler import IndicatorsCache

if optimizer_stubbed:
    sys.modules.pop('optimizer', None)
    sys.modules.pop('bot.optimizer', None)
if strategy_stubbed:
    sys.modules.pop('strategy_optimizer', None)
    sys.modules.pop('bot.strategy_optimizer', None)


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


def test_volatility_indicators_present():
    cfg = BotConfig(bollinger_window=3, ulcer_window=3)
    df = make_df(10)
    ind = IndicatorsCache(df, cfg, 0.1)
    assert "bollinger_wband" in ind.df.columns
    assert "ulcer_index" in ind.df.columns


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


def test_rsi_adx_incremental_update():
    cfg = BotConfig(
        ema30_period=3,
        ema100_period=5,
        ema200_period=7,
        atr_period_default=3,
        rsi_window=3,
        adx_window=3,
    )
    df = make_df(30)
    ind = IndicatorsCache(df, cfg, 0.1)
    prev_close = ind.last_close
    prev_high = ind.last_high
    prev_low = ind.last_low
    prev_atr = ind.last_atr
    prev_gain = ind._rsi_avg_gain
    prev_loss = ind._rsi_avg_loss
    prev_dm_plus = ind._dm_plus
    prev_dm_minus = ind._dm_minus
    prev_adx = ind.last_adx

    new = pd.DataFrame(
        {
            "close": [2.1],
            "high": [2.2],
            "low": [2.0],
            "volume": [1.0],
        },
        index=[df.index[-1] + pd.Timedelta(minutes=1)],
    )
    ind.update(new)

    # expected ATR first
    tr = max(2.2 - 2.0, abs(2.2 - prev_close), abs(2.0 - prev_close))
    expected_atr = (prev_atr * (cfg.atr_period_default - 1) + tr) / cfg.atr_period_default

    # expected RSI
    gain = max(2.1 - prev_close, 0)
    loss = max(prev_close - 2.1, 0)
    expected_avg_gain = (prev_gain * (cfg.rsi_window - 1) + gain) / cfg.rsi_window
    expected_avg_loss = (prev_loss * (cfg.rsi_window - 1) + loss) / cfg.rsi_window
    if expected_avg_loss == 0:
        expected_rsi = 100.0
    else:
        rs = expected_avg_gain / expected_avg_loss
        expected_rsi = 100 - 100 / (1 + rs)

    # expected ADX
    up_move = 2.2 - prev_high
    down_move = prev_low - 2.0
    plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
    minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0
    expected_dm_plus = (prev_dm_plus * (cfg.adx_window - 1) + plus_dm) / cfg.adx_window
    expected_dm_minus = (prev_dm_minus * (cfg.adx_window - 1) + minus_dm) / cfg.adx_window
    tr_sum = expected_atr * cfg.adx_window
    plus_di = 0.0 if tr_sum == 0 else 100 * expected_dm_plus / tr_sum
    minus_di = 0.0 if tr_sum == 0 else 100 * expected_dm_minus / tr_sum
    denom = plus_di + minus_di
    dx = 0.0 if denom == 0 else 100 * abs(plus_di - minus_di) / denom
    expected_adx = (prev_adx * (cfg.adx_window - 1) + dx) / cfg.adx_window

    assert np.isclose(ind.last_rsi, expected_rsi)
    assert np.isclose(ind.last_adx, expected_adx)


def test_short_dataframe_no_value_error():
    cfg = BotConfig()
    df = make_df(5)
    ind = IndicatorsCache(df, cfg, 0.1)
    assert ind.adx.isna().all()


def test_dataframe_equal_to_adx_window():
    cfg = BotConfig()
    df = make_df(cfg.adx_window)
    ind = IndicatorsCache(df, cfg, 0.1)
    assert ind.adx.isna().all()


def test_rsi_fallback(monkeypatch):
    cfg = BotConfig()
    df = make_df(10)

    def fail_rsi(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(ta.momentum, "rsi", fail_rsi)
    ind = IndicatorsCache(df, cfg, 0.1)
    assert (ind.rsi == 0).all()
    assert ind._rsi_avg_gain is None
    assert ind._rsi_avg_loss is None
