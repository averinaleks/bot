import os
import sys
import types
import sklearn  # ensure real scikit-learn loaded before tests may stub it
import sklearn.model_selection  # preload submodules used in tests
import sklearn.base

import pytest
import pandas as pd

# Ensure the project root is available before tests import project modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _stub_modules():
    """Insert lightweight stubs for optional heavy dependencies."""
    numba_mod = types.ModuleType("numba")
    numba_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    numba_mod.jit = lambda *a, **k: (lambda f: f)
    numba_mod.prange = range
    sys.modules.setdefault("numba", numba_mod)
    sys.modules.setdefault("numba.cuda", numba_mod.cuda)

    pybit_mod = types.ModuleType("pybit")
    ut_mod = types.ModuleType("unified_trading")
    ut_mod.HTTP = object
    pybit_mod.unified_trading = ut_mod
    sys.modules.setdefault("pybit", pybit_mod)
    sys.modules.setdefault("pybit.unified_trading", ut_mod)

    sys.modules.setdefault("httpx", types.ModuleType("httpx"))
    telegram_error_mod = types.ModuleType("telegram.error")
    telegram_error_mod.RetryAfter = Exception
    sys.modules.setdefault("telegram", types.ModuleType("telegram"))
    sys.modules.setdefault("telegram.error", telegram_error_mod)

    psutil_mod = types.ModuleType("psutil")
    psutil_mod.cpu_percent = lambda interval=1: 0
    psutil_mod.virtual_memory = lambda: type("mem", (), {"percent": 0})
    sys.modules.setdefault("psutil", psutil_mod)

    sys.modules.setdefault("websockets", types.ModuleType("websockets"))

    optimizer_mod = types.ModuleType("optimizer")
    class _PO:
        def __init__(self, *a, **k):
            pass
    optimizer_mod.ParameterOptimizer = _PO
    sys.modules.setdefault("optimizer", optimizer_mod)

    class _RayRemoteFunction:
        def __init__(self, func):
            self._function = func

        def remote(self, *args, **kwargs):
            return self._function(*args, **kwargs)

        def options(self, *args, **kwargs):
            return self

    def _ray_remote(func=None, **remote_kwargs):
        if func is None:
            def wrapper(f):
                return _RayRemoteFunction(f)
            return wrapper
        return _RayRemoteFunction(func)

    ray_mod = types.ModuleType("ray")
    _ray_state = {"initialized": False}
    def _init(*a, **k):
        _ray_state["initialized"] = True
    def _is_initialized():
        return _ray_state["initialized"]
    ray_mod.remote = _ray_remote
    ray_mod.get = lambda x: x
    ray_mod.init = _init
    ray_mod.is_initialized = _is_initialized
    sys.modules.setdefault("ray", ray_mod)

    tenacity_mod = types.ModuleType("tenacity")
    tenacity_mod.retry = lambda *a, **k: (lambda f: f)
    tenacity_mod.wait_exponential = lambda *a, **k: None
    tenacity_mod.stop_after_attempt = lambda *a, **k: (lambda f: f)
    sys.modules.setdefault("tenacity", tenacity_mod)

_stub_modules()


@pytest.fixture(autouse=True)
def _add_root_and_stub_modules(monkeypatch):
    """Ensure stubs exist for each test."""
    _stub_modules()
    yield


@pytest.fixture
def sample_ohlcv():
    """Small OHLCV dataframe for simple tests."""
    return pd.DataFrame(
        {
            "close": [1.0, 2.0, 1.0],
            "open": [1.0, 2.0, 1.0],
            "high": [1.0, 2.0, 1.0],
            "low": [1.0, 2.0, 1.0],
            "volume": [0.0, 0.0, 0.0],
        }
    )
