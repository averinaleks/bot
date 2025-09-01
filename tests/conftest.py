import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the repository root is on ``sys.path`` so that modules such as
# ``config`` or ``trading_bot`` can be imported when tests are executed from the
# ``tests`` directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



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

    monkeypatch.setattr(asyncio, "sleep", _sleep)

