import os
import sys
import pytest
import pandas as pd
import asyncio



@pytest.fixture
def csrf_secret(monkeypatch):
    secret = "testsecret"
    monkeypatch.setenv("CSRF_SECRET", secret)
    return secret


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=3, freq="1min", tz="UTC")
    # Use a price sequence that first rises then falls to allow the
    # tests to exercise both profitable and losing trades.
    data = {
        "open": [100.0, 101.0, 100.0],
        "high": [100.0, 101.0, 100.0],
        "low": [100.0, 101.0, 100.0],
        "close": [100.0, 101.0, 100.0],
        "volume": [1.0, 1.0, 1.0],
        "atr": [1.0, 1.0, 1.0],
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def fast_sleep(monkeypatch):
    async def _sleep(_delay):
        return None

