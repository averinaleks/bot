import importlib.util
import os
import sys
import asyncio
import secrets

# Явно включаем тестовый режим для всех тестов до импорта ``sitecustomize``.
# Это гарантирует, что модуль распознает, что тесты запускаются в среде,
# где доступна облегчённая реализация ``bcrypt`` без необходимости установки
# тяжелой зависимости.
os.environ.setdefault("TEST_MODE", "1")

import pytest

# ``sitecustomize`` готовит окружение (устанавливает опциональные зависимости)
# и включает защитные патчи, необходимые для тестов.  На GitHub Actions модуль
# автоматически подхватывается интерпретатором, но в изолированной среде
# выполнение ``pytest`` через ``python -m`` может пропустить автоматический
# импорт.  Явный импорт гарантирует, что подготовка окружения выполнится до
# загрузки остальных модулей и до обращения тестов к ``pandas`` и другим
# тяжёлым библиотекам.
import sitecustomize  # noqa: F401  # side effect only

from bot import http_client as _http_client


def _has_pytest_asyncio_plugin() -> bool:
    """Return ``True`` when the ``pytest-asyncio`` plugin is importable.

    ``pytest`` automatically exposes a namespace package named
    :mod:`pytest_asyncio` even when the third-party plugin is not installed.
    Importing the top-level module alone would therefore succeed and make the
    fallback below unreachable.  Checking the actual plugin module ensures that
    we only rely on ``pytest-asyncio`` when it is fully available.
    """

    try:
        spec = importlib.util.find_spec("pytest_asyncio.plugin")
    except ModuleNotFoundError:
        # ``find_spec`` raises ``ModuleNotFoundError`` when the top-level
        # ``pytest_asyncio`` package itself is unavailable.  Treat this the
        # same as the plugin not being installed so the synchronous fallback
        # below is used instead of crashing during test collection.
        return False
    return spec is not None


if _has_pytest_asyncio_plugin():  # pragma: no cover - exercised in CI
    pytest_plugins = ("pytest_asyncio",)
else:  # pragma: no cover - exercised when plugin missing on CI
    pytest_plugins: tuple[str, ...] = ()

    def _create_loop() -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    @pytest.fixture
    def event_loop():
        loop = _create_loop()
        try:
            yield loop
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            asyncio.set_event_loop(None)
            loop.close()

    @pytest.hookimpl(tryfirst=True)
    def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
        test_func = pyfuncitem.obj
        if asyncio.iscoroutinefunction(test_func):
            loop = pyfuncitem.funcargs.get("event_loop")
            owns_loop = False
            if loop is None:
                loop = _create_loop()
                owns_loop = True
            try:
                call_kwargs = {
                    name: pyfuncitem.funcargs[name]
                    for name in pyfuncitem._fixtureinfo.argnames
                    if name in pyfuncitem.funcargs
                }
                loop.run_until_complete(test_func(**call_kwargs))
            finally:
                if owns_loop:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    asyncio.set_event_loop(None)
                    loop.close()
            return True
        return None

# MLflow spawns a background telemetry thread on import which keeps the
# process alive after tests finish. Disabling telemetry prevents the thread
# from starting and avoids hanging the pytest run.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "1")

# Ensure the project root is on sys.path so tests can import modules like ``config``.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def csrf_secret(monkeypatch):
    secret = secrets.token_hex(32)
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
