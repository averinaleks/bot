import importlib
import sys
import numpy as np
import pytest


def _reload_module(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setitem(sys.modules, "numba", None)
    import bot.portfolio_backtest as pb
    importlib.reload(pb)
    return pb


def test_simulate_trades_applies_fee(monkeypatch):
    pb = _reload_module(monkeypatch)
    symbol_ids = np.array([0, 0], dtype=np.int32)
    close = np.array([100, 100], dtype=np.float32)
    high = np.array([100, 103], dtype=np.float32)
    low = np.array([100, 99], dtype=np.float32)
    ema_fast = np.array([2, 2], dtype=np.float32)
    ema_slow = np.array([1, 1], dtype=np.float32)
    atr = np.array([1, 1], dtype=np.float32)
    prob = np.array([0.7, 0.7], dtype=np.float32)

    no_fee = pb._simulate_trades(
        symbol_ids,
        close,
        high,
        low,
        ema_fast,
        ema_slow,
        atr,
        prob,
        0.6,
        1.0,
        2.0,
        0.0,
        1,
        1,
    )
    with_fee = pb._simulate_trades(
        symbol_ids,
        close,
        high,
        low,
        ema_fast,
        ema_slow,
        atr,
        prob,
        0.6,
        1.0,
        2.0,
        0.001,
        1,
        1,
    )
    assert with_fee[0] == pytest.approx(no_fee[0] - 0.002)
