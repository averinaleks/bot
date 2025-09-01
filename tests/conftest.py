import os
import sys
import pytest
import pandas as pd
import asyncio


# Ensure project root is on the Python path so that top-level modules can be
# imported regardless of the current working directory used by the test
# runner.  This mirrors the behaviour of ``pip install -e .`` without requiring
# an explicit installation step.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)



@pytest.fixture
def csrf_secret(monkeypatch):
    secret = "testsecret"
    monkeypatch.setenv("CSRF_SECRET", secret)
    return secret


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=3, freq="1min", tz="UTC")
    data = {
        "open": [100.0, 101.0, 102.0],
        "high": [100.0, 101.0, 102.0],
        "low": [100.0, 101.0, 102.0],
        "close": [100.0, 101.0, 102.0],
        "volume": [1.0, 1.0, 1.0],
        "atr": [1.0, 1.0, 1.0],
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def fast_sleep(monkeypatch):
    async def _sleep(_delay):
        return None

