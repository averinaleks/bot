import sys
import numpy as np
import pandas as pd
import pytest
from bot.config import BotConfig

sys.modules.pop('strategy_optimizer', None)
sys.modules.pop('bot.strategy_optimizer', None)
from bot.strategy_optimizer import StrategyOptimizer, _portfolio_backtest_remote
from bot.portfolio_backtest import portfolio_backtest


class DummyDataHandler:
    def __init__(self, df_dict):
        self.usdt_pairs = list(df_dict.keys())
        self.ohlcv = pd.concat(df_dict.values())


def make_df(symbol):
    idx = pd.date_range('2020-01-01', periods=30, freq='min')
    df = pd.DataFrame({
        'close': np.linspace(1, 2, len(idx)),
        'open': np.linspace(1, 2, len(idx)),
        'high': np.linspace(1, 2, len(idx)) + 0.1,
        'low': np.linspace(1, 2, len(idx)) - 0.1,
        'volume': np.ones(len(idx)),
    }, index=idx)
    df['symbol'] = symbol
    return df.set_index(['symbol', df.index])


@pytest.mark.asyncio
async def test_strategy_optimizer_returns_params(monkeypatch):
    df_dict = {'BTCUSDT': make_df('BTCUSDT'), 'ETHUSDT': make_df('ETHUSDT')}
    config = BotConfig(timeframe='1m', optuna_trials=1, portfolio_metric='sharpe')
    captured = {}

    def dummy_remote(df_dict, params, timeframe, metric, max_positions):
        captured['metric'] = metric
        return 0.1

    monkeypatch.setattr(_portfolio_backtest_remote, 'remote', dummy_remote)
    opt = StrategyOptimizer(config, DummyDataHandler(df_dict))
    params = await opt.optimize()
    assert isinstance(params, dict)
    assert 'ema30_period' in params
    assert captured['metric'] == 'sharpe'


def test_portfolio_backtest_runs():
    df_dict = {'BTCUSDT': make_df('BTCUSDT'), 'ETHUSDT': make_df('ETHUSDT')}
    params = {
        'ema30_period': 3,
        'ema100_period': 5,
        'tp_multiplier': 1.0,
        'sl_multiplier': 1.0,
        'base_probability_threshold': 0.6,
    }
    ratio = portfolio_backtest(df_dict, params, '1m', max_positions=1)
    assert isinstance(ratio, float)
