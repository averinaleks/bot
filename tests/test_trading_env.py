import pytest
from bot.config import BotConfig
from bot.model_builder import TradingEnv

def test_drawdown_penalty(sample_ohlcv):
    cfg = BotConfig(drawdown_penalty=0.5)
    env = TradingEnv(sample_ohlcv, cfg)
    env.reset()
    _, r1, _, _ = env.step(1)  # buy -> profit 1
    assert r1 == 1.0
    assert env.balance == 1.0
    assert env.max_balance == 1.0
    _, r2, _, _ = env.step(1)  # buy -> loss 1 and drawdown 1
    assert pytest.approx(r2) == -1.5
    assert env.balance == 0.0
    assert env.max_balance == 1.0


def test_step_rewards(sample_ohlcv):
    env = TradingEnv(sample_ohlcv, BotConfig())
    env.reset()
    _, r1, _, _ = env.step(1)
    assert r1 == 1.0
    _, r2, done, _ = env.step(2)
    assert r2 == 1.0
    assert done
    assert env.balance == 2.0

