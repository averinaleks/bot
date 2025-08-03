import numpy as np
import pandas as pd

from bot.portfolio_backtest import portfolio_backtest


def make_df(symbol: str, direction: str, prob: float) -> pd.DataFrame:
    idx = pd.date_range('2020-01-01', periods=30, freq='min')
    if direction == 'up':
        close = np.linspace(1, 2, len(idx))
    else:
        close = np.linspace(2, 1, len(idx))
    high = close + 0.1
    low = close - 0.1
    df = pd.DataFrame(
        {
            'close': close,
            'open': close,
            'high': high,
            'low': low,
            'volume': np.ones(len(idx)),
            'probability': np.full(len(idx), prob),
        },
        index=idx,
    )
    df['symbol'] = symbol
    return df.set_index(['symbol', df.index])


def test_buy_respects_probability_threshold():
    df_low = {'BTCUSDT': make_df('BTCUSDT', 'up', prob=0.5)}
    df_high = {'BTCUSDT': make_df('BTCUSDT', 'up', prob=0.7)}
    params = {
        'ema30_period': 3,
        'ema100_period': 5,
        'tp_multiplier': 1.0,
        'sl_multiplier': 1.0,
        'base_probability_threshold': 0.6,
    }
    ratio_low = portfolio_backtest(df_low, params, '1m', max_positions=1)
    ratio_high = portfolio_backtest(df_high, params, '1m', max_positions=1)
    assert ratio_low == 0.0
    assert ratio_high > ratio_low


def test_sell_respects_probability_threshold():
    df_low = {'BTCUSDT': make_df('BTCUSDT', 'down', prob=0.9)}
    df_high = {'BTCUSDT': make_df('BTCUSDT', 'down', prob=0.3)}
    params = {
        'ema30_period': 3,
        'ema100_period': 5,
        'tp_multiplier': 1.0,
        'sl_multiplier': 1.0,
        'base_probability_threshold': 0.6,
    }
    ratio_low = portfolio_backtest(df_low, params, '1m', max_positions=1)
    ratio_high = portfolio_backtest(df_high, params, '1m', max_positions=1)
    assert ratio_low == 0.0
    assert ratio_high > ratio_low
