import pandas as pd
import numpy as np
import pytest
from config import BotConfig
from model_builder import TradingEnv


def make_df():
    return pd.DataFrame({
        'close': [1.0, 2.0, 1.0],
        'open': [1.0, 2.0, 1.0],
        'high': [1.0, 2.0, 1.0],
        'low': [1.0, 2.0, 1.0],
        'volume': [0.0, 0.0, 0.0],
    })


def test_drawdown_penalty():
    cfg = BotConfig(drawdown_penalty=0.5)
    env = TradingEnv(make_df(), cfg)
    env.reset()
    _, r1, _, _ = env.step(1)  # buy -> profit 1
    assert r1 == 1.0
    assert env.balance == 1.0
    assert env.max_balance == 1.0
    _, r2, _, _ = env.step(1)  # buy -> loss 1 and drawdown 1
    assert pytest.approx(r2) == -1.5
    assert env.balance == 0.0
    assert env.max_balance == 1.0

