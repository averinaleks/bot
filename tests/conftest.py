import os
import sys
import asyncio
import pytest

from bot import http_client as _http_client

pytest_plugins = ("pytest_asyncio",)

# MLflow spawns a background telemetry thread on import which keeps the
# process alive after tests finish. Disabling telemetry prevents the thread
# from starting and avoids hanging the pytest run.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "1")

# Ensure the project root is on sys.path so tests can import modules like ``config``.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def csrf_secret(monkeypatch):
    secret = "testsecret"
    monkeypatch.setenv("CSRF_SECRET", secret)
    return secret


@pytest.fixture
def sample_ohlcv():
    pd = pytest.importorskip("pandas")
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

    monkeypatch.setattr(asyncio, "sleep", _sleep)


@pytest.fixture(scope="session", autouse=True)
def _close_shared_http_client():
    """Ensure the shared async HTTP client is closed after tests."""
    yield
    asyncio.run(_http_client.close_async_http_client())
