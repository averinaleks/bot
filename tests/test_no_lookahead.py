import numpy as np
import pandas as pd
import types
import pytest

from bot.data_handler import DataHandler
from bot.services import model_builder_service as mbs
from pathlib import Path


class Cfg:
    timeframe = '1m'
    ema30_period = 2
    min_data_length = 0
    min_liquidity = 0
    max_symbols = 1
    use_polars = False


@pytest.mark.asyncio
async def test_data_handler_ema_shift():
    idx = pd.date_range('2020-01-01', periods=3, freq='min')
    df = pd.DataFrame(
        {
            'open': [1, 2, 3],
            'high': [1, 2, 3],
            'low': [1, 2, 3],
            'close': [1, 2, 3],
            'volume': [1, 2, 3],
        },
        index=idx,
    )
    df['symbol'] = 'SYM'
    df = df.set_index('symbol', append=True).swaplevel(0, 1)
    dh = DataHandler(Cfg(), None, None, exchange=types.SimpleNamespace())
    await dh.synchronize_and_update('SYM', df)
    ema = dh.indicators['SYM'].df['ema30']
    expected = df['close'].ewm(span=Cfg.ema30_period, adjust=False).mean().shift(1)
    expected = expected.reset_index(drop=True)
    pd.testing.assert_series_equal(ema, expected, check_names=False)


def test_compute_ema_shift():
    prices = [1.0, 2.0, 3.0, 4.0]
    ema = mbs._compute_ema(prices, span=2)
    series = pd.Series(prices)
    expected = series.ewm(span=2, adjust=False).mean().shift(1).to_numpy()
    assert np.allclose(ema, expected, equal_nan=True)


def test_service_train_with_prices_shift(tmp_path):
    prices = [1.0, 2.0, 3.0, 4.0]
    labels = [0, 1, 0, 1]
    mbs.MODEL_FILE = Path('tmp_model.pkl')
    app = mbs.app
    with app.test_client() as client:
        resp = client.post('/train', json={'prices': prices, 'labels': labels})
        assert resp.status_code == 200
    computed = mbs._compute_ema(prices)
    mask = ~np.isnan(computed)
    assert np.isclose(mbs._scaler.mean_[0], computed[mask].mean())
    if mbs.MODEL_FILE.exists():
        mbs.MODEL_FILE.unlink()


def test_scaler_train_only(tmp_path):
    mbs.MODEL_FILE = Path('tmp_model.pkl')
    app = mbs.app
    with app.test_client() as client:
        resp = client.post('/train', json={'features': [[0.0], [1.0], [2.0]], 'labels': [0, 1, 0]})
        assert resp.status_code == 200
        mean_before = mbs._scaler.mean_.copy()
        resp = client.post('/predict', json={'features': [1.0]})
        assert resp.status_code == 200
        assert np.allclose(mean_before, mbs._scaler.mean_)
    if mbs.MODEL_FILE.exists():
        mbs.MODEL_FILE.unlink()
