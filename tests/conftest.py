import os
import sys
import types
import importlib.abc
import importlib.util
try:
    import sklearn  # ensure real scikit-learn loaded before tests may stub it
    import sklearn.model_selection  # preload submodules used in tests
    import sklearn.base
    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_SKLEARN = False

import pytest
import pandas as pd

# Register a marker for tests requiring scikit-learn
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_sklearn: mark test that needs the scikit-learn package",
    )


def pytest_runtest_setup(item):
    if not HAVE_SKLEARN and item.get_closest_marker("requires_sklearn"):
        pytest.skip("scikit-learn not installed")

# Ensure the project root and its parent are available before tests import project modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PARENT = os.path.dirname(ROOT)
for path in (PARENT, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


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
    def _shutdown():
        _ray_state["initialized"] = False
    ray_mod.remote = _ray_remote
    ray_mod.get = lambda x: x
    ray_mod.init = _init
    ray_mod.is_initialized = _is_initialized
    ray_mod.shutdown = _shutdown
    sys.modules.setdefault("ray", ray_mod)

    optuna_mod = types.ModuleType("optuna")
    optuna_samplers = types.ModuleType("optuna.samplers")
    class _TPESampler:
        def __init__(self, *a, **k):
            pass
    optuna_samplers.TPESampler = _TPESampler
    optuna_mod.samplers = optuna_samplers

    def _create_study_stub(*a, **k):
        class _Trial:
            def __init__(self, number: int):
                self.number = number
                self.params: dict = {}

            def suggest_int(self, name, low, high):
                self.params[name] = low
                return low

            def suggest_float(self, name, low, high):
                self.params[name] = low
                return low

        class _Study:
            def __init__(self):
                self.trials = []
                self.best_params = {}
                self.best_value = 0.0

            def ask(self):
                trial = _Trial(len(self.trials))
                self.trials.append(trial)
                return trial

            def tell(self, trial, value):
                if value is not None and value > self.best_value:
                    self.best_value = value
                    self.best_params = getattr(trial, "params", {})

            def optimize(self, *a, **k):
                pass

        return _Study()

    optuna_mod.create_study = _create_study_stub
    optuna_integration = types.ModuleType("optuna.integration.mlflow")
    optuna_integration.MLflowCallback = object
    optuna_exceptions = types.ModuleType("optuna.exceptions")
    class _ExpWarn(Warning):
        pass
    optuna_exceptions.ExperimentalWarning = _ExpWarn
    optuna_mod.exceptions = optuna_exceptions
    sys.modules["optuna"] = optuna_mod
    sys.modules["optuna.samplers"] = optuna_samplers
    sys.modules["optuna.integration.mlflow"] = optuna_integration
    sys.modules["optuna.exceptions"] = optuna_exceptions

    tenacity_mod = types.ModuleType("tenacity")
    tenacity_mod.retry = lambda *a, **k: (lambda f: f)
    tenacity_mod.wait_exponential = lambda *a, **k: None
    tenacity_mod.stop_after_attempt = lambda *a, **k: (lambda f: f)
    sys.modules.setdefault("tenacity", tenacity_mod)

    requests_mod = types.ModuleType("requests")
    requests_mod.RequestException = Exception
    requests_mod.get = lambda *a, **k: None
    requests_mod.post = lambda *a, **k: None
    sys.modules.setdefault("requests", requests_mod)

    joblib_mod = types.ModuleType("joblib")
    joblib_mod.dump = lambda *a, **k: None
    joblib_mod.load = lambda *a, **k: {}
    sys.modules.setdefault("joblib", joblib_mod)

    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *a, **k: None
    sys.modules.setdefault("dotenv", dotenv_mod)
    sys.modules.setdefault("dotenv.main", dotenv_mod)

    try:  # use real Flask if available
        import flask  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency missing
        flask_mod = types.ModuleType("flask")

        class _Flask:
            def __init__(self, name):
                self.name = name

            def route(self, *a, **k):
                def decorator(f):
                    return f
                return decorator

            def run(self, *a, **k):
                pass

            def before_request(self, func):
                return func

        flask_mod.Flask = _Flask
        flask_mod.jsonify = lambda *a, **k: dict(*a, **k)
        flask_mod.request = types.SimpleNamespace(get_json=lambda *a, **k: {})
        sys.modules.setdefault("flask", flask_mod)

    ta_mod = types.ModuleType("ta")
    trend_mod = types.ModuleType("ta.trend")
    momentum_mod = types.ModuleType("ta.momentum")
    volatility_mod = types.ModuleType("ta.volatility")

    def _ema_indicator(series, window=14, fillna=True):
        s = pd.Series(series)
        result = s.ewm(span=window, adjust=False).mean()
        return result.fillna(result.iloc[0]) if fillna else result

    def _average_true_range(high, low, close, window=14, fillna=True):
        h = pd.Series(high)
        l = pd.Series(low)
        c = pd.Series(close)
        prev_close = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window, min_periods=1).mean()
        return atr.fillna(0) if fillna else atr

    def _rsi(series, window=14, fillna=True):
        s = pd.Series(series)
        diff = s.diff().fillna(0)
        gain = diff.clip(lower=0)
        loss = -diff.clip(upper=0)
        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0) if fillna else rsi

    def _macd_diff(close, window_slow=26, window_fast=12, window_sign=9, fillna=True):
        c = pd.Series(close)
        fast = c.ewm(span=window_fast, adjust=False).mean()
        slow = c.ewm(span=window_slow, adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=window_sign, adjust=False).mean()
        diff = macd - signal
        return diff.fillna(0) if fillna else diff

    class _BollingerBands:
        def __init__(self, series, window=20, fillna=True):
            self.series = pd.Series(series)

        def bollinger_wband(self):
            return pd.Series([0.0] * len(self.series))

    def _ulcer_index(series, window=14, fillna=True):
        return pd.Series([0.0] * len(pd.Series(series)))

    trend_mod.ema_indicator = _ema_indicator
    trend_mod.macd_diff = _macd_diff
    trend_mod.adx = lambda *a, **k: pd.Series([0.0] * len(pd.Series(a[0])))
    momentum_mod.rsi = _rsi
    volatility_mod.average_true_range = _average_true_range
    volatility_mod.BollingerBands = _BollingerBands
    volatility_mod.ulcer_index = _ulcer_index
    ta_mod.trend = trend_mod
    ta_mod.momentum = momentum_mod
    ta_mod.volatility = volatility_mod
    sys.modules.setdefault("ta", ta_mod)
    sys.modules.setdefault("ta.trend", trend_mod)
    sys.modules.setdefault("ta.momentum", momentum_mod)
    sys.modules.setdefault("ta.volatility", volatility_mod)

    class _PlExpr:
        def __init__(self, func):
            self.func = func

        def __call__(self, df):
            return self.func(df)

    class _PlColumn:
        def __init__(self, name):
            self.name = name

        def __ne__(self, other):
            return _PlExpr(lambda df: df[self.name] != other)

    def col(name):
        return _PlColumn(name)

    class DataFrame:
        def __init__(self, data=None):
            if isinstance(data, pd.DataFrame):
                self._df = data.copy()
            else:
                self._df = pd.DataFrame(data)

        @classmethod
        def from_pandas(cls, df):
            return cls(df)

        def to_pandas(self):
            return self._df.copy()

        def clone(self):
            return DataFrame(self._df.copy())

        @property
        def height(self):
            return len(self._df)

        def filter(self, expr):
            mask = expr(self._df)
            return DataFrame(self._df[mask])

        def sort(self, columns):
            return DataFrame(self._df.sort_values(columns))

    def concat(dfs):
        pdfs = [df._df if isinstance(df, DataFrame) else df for df in dfs]
        return DataFrame(pd.concat(pdfs, ignore_index=True))

    polars_mod = types.ModuleType("polars")
    polars_mod.DataFrame = DataFrame
    polars_mod.from_pandas = DataFrame.from_pandas
    polars_mod.concat = concat
    polars_mod.col = col
    sys.modules.setdefault("polars", polars_mod)

    if not HAVE_SKLEARN:
        sk_mod = types.ModuleType("sklearn")
        preproc_mod = types.ModuleType("sklearn.preprocessing")
        class _Scaler:
            def fit(self, *a, **k):
                return self
            def transform(self, X):
                return X
        preproc_mod.StandardScaler = _Scaler
        metrics_mod = types.ModuleType("sklearn.metrics")
        metrics_mod.brier_score_loss = lambda *a, **k: 0.0
        linear_mod = types.ModuleType("sklearn.linear_model")
        class _LogReg:
            def fit(self, *a, **k):
                return self
            def predict_proba(self, X):
                import numpy as _np
                return _np.zeros((len(X), 2))
        linear_mod.LogisticRegression = _LogReg
        calib_mod = types.ModuleType("sklearn.calibration")
        calib_mod.calibration_curve = lambda *a, **k: ([], [])
        sk_mod.preprocessing = preproc_mod
        sk_mod.metrics = metrics_mod
        sk_mod.linear_model = linear_mod
        sk_mod.calibration = calib_mod
        sys.modules.setdefault("sklearn", sk_mod)
        sys.modules.setdefault("sklearn.preprocessing", preproc_mod)
        sys.modules.setdefault("sklearn.metrics", metrics_mod)
        sys.modules.setdefault("sklearn.linear_model", linear_mod)
        sys.modules.setdefault("sklearn.calibration", calib_mod)

    scipy_mod = types.ModuleType("scipy")
    stats_mod = types.ModuleType("scipy.stats")
    def _zscore(a, axis=0):
        import numpy as _np  # local import to avoid global dependency
        a = _np.asarray(a, dtype=float)
        mean = a.mean(axis=axis, keepdims=True)
        std = a.std(axis=axis, ddof=0, keepdims=True)
        return (a - mean) / std
    stats_mod.zscore = _zscore
    scipy_mod.stats = stats_mod
    sys.modules.setdefault("scipy", scipy_mod)
    sys.modules.setdefault("scipy.stats", stats_mod)

class _OptunaLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return sys.modules.get(spec.name)

    def exec_module(self, module):
        pass


class _OptunaFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "optuna" or fullname.startswith("optuna."):
            if fullname not in sys.modules:
                _stub_modules()
            if fullname in sys.modules:
                return importlib.util.spec_from_loader(fullname, _OptunaLoader())
        return None

sys.meta_path.insert(0, _OptunaFinder())

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
