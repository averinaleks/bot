import argparse
import asyncio
import builtins
import importlib.util
import logging
import sys
import traceback
from types import ModuleType, SimpleNamespace

import pytest


if "pandas" not in sys.modules:
    pandas_stub = ModuleType("pandas")
    pandas_stub.DataFrame = lambda *args, **kwargs: SimpleNamespace()
    pandas_stub.to_datetime = lambda *args, **kwargs: SimpleNamespace()
    sys.modules["pandas"] = pandas_stub


from run_bot import _maybe_load_initial, prepare_data_handler, run_trading_cycle


class DummyDomainError(Exception):
    """Domain-specific error used to emulate TradeManager failures."""


# Mimic trade manager module attributes discovered by ``run_trading_cycle``.
TradeManagerTaskError = DummyDomainError


class _DomainTradeManager:
    async def run(self):
        raise DummyDomainError("domain failure")


class _UnexpectedTradeManager:
    async def run(self):
        raise RuntimeError("unexpected failure")


@pytest.mark.asyncio
async def test_run_trading_cycle_reraises_domain_error(caplog):
    manager = _DomainTradeManager()

    with caplog.at_level(logging.ERROR):
        with pytest.raises(DummyDomainError) as excinfo:
            await run_trading_cycle(manager, runtime=None)

    message = str(excinfo.value)
    assert message.startswith("Trading loop aborted after TradeManager error")
    assert message.endswith("domain failure")
    assert excinfo.value.__cause__ is not None

    frames = traceback.extract_tb(excinfo.value.__traceback__)
    assert any(frame.name == "run" for frame in frames)

    record = next(record for record in caplog.records if "Trading loop aborted" in record.message)
    assert record.exc_info is not None


@pytest.mark.asyncio
async def test_run_trading_cycle_logs_and_reraises_unexpected_error(caplog):
    manager = _UnexpectedTradeManager()

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            await run_trading_cycle(manager, runtime=None)

    assert str(excinfo.value) == "unexpected failure"

    frames = traceback.extract_tb(excinfo.value.__traceback__)
    assert any(frame.name == "run" for frame in frames)

    record = next(record for record in caplog.records if record.message == "Unexpected error during trading loop")
    assert record.exc_info is not None


@pytest.mark.asyncio
async def test_maybe_load_initial_logs_expected_errors(caplog):
    class DummyHandler:
        def load_initial(self):
            raise RuntimeError("broken")

    handler = DummyHandler()

    with caplog.at_level(logging.WARNING):
        await _maybe_load_initial(handler)

    record = next(record for record in caplog.records if record.message.startswith("Initial data load failed"))
    assert record.exc_info is not None


@pytest.mark.asyncio
async def test_maybe_load_initial_logs_value_errors(caplog):
    class DummyHandler:
        def load_initial(self):
            raise ValueError("invalid state")

    handler = DummyHandler()

    with caplog.at_level(logging.WARNING):
        await _maybe_load_initial(handler)

    assert any(
        record.message.startswith("Initial data load failed") for record in caplog.records
    )


@pytest.mark.asyncio
async def test_maybe_load_initial_logs_common_value_errors(caplog):
    class DummyHandler:
        def load_initial(self):
            raise RuntimeError("legacy failure")

    handler = DummyHandler()

    with caplog.at_level(logging.WARNING):
        await _maybe_load_initial(handler)

    assert any(
        record.message.startswith("Initial data load failed") for record in caplog.records
    )


def test_prepare_data_handler_without_pandas(monkeypatch, caplog):
    """Ensure pandas fallback stub is used when the library is unavailable."""

    monkeypatch.delitem(sys.modules, "pandas", raising=False)

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    handler = SimpleNamespace()

    class _Cfg:
        def asdict(self):
            return {"atr_period_default": 14}

        def get(self, key, default=None):
            return default

    cfg = _Cfg()

    prepare_data_handler(handler, cfg, symbols=None)

    assert handler.usdt_pairs == ["BTCUSDT"]
    assert "pandas" not in sys.modules

    indicator_df = handler.indicators["BTCUSDT"].df
    indicator_df_2h = handler.indicators_2h["BTCUSDT"].df
    assert indicator_df == ()
    assert indicator_df_2h == ()
    assert indicator_df is handler.ohlcv
    assert indicator_df_2h is handler.ohlcv_2h

    assert handler.funding_rates == {"BTCUSDT": 0.0}
    assert handler.open_interest == {"BTCUSDT": 0.0}


def test_prepare_data_handler_with_pandas_stub(monkeypatch):
    """Ensure a typed empty DataFrame is created when pandas is available."""

    class _FakeDataFrame:
        def __init__(self, data=None, columns=None):
            self.data = data or {}
            self.columns = columns or []

    class _FakeSeries:
        def __init__(self, dtype=None):  # noqa: D401 - simple stub
            self.dtype = dtype

    pandas_stub = SimpleNamespace(DataFrame=_FakeDataFrame, Series=_FakeSeries)
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)

    handler = SimpleNamespace()

    class _Cfg:
        def asdict(self):
            return {}

        def get(self, key, default=None):  # noqa: ARG002
            return default

    cfg = _Cfg()
    prepare_data_handler(handler, cfg, symbols=["ETHUSDT", "BTCUSDT"])

    assert handler.ohlcv.columns == (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    assert set(handler.funding_rates) == {"ETHUSDT", "BTCUSDT"}
    assert set(handler.indicators) == {"ETHUSDT", "BTCUSDT"}


@pytest.mark.asyncio
async def test_run_trading_cycle_cancels_long_running_loop(monkeypatch, caplog):
    class _SlowManager:
        def __init__(self):
            self.cancelled = False
            self.stopped = False

        async def run(self):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        async def stop(self):
            self.stopped = True

    manager = _SlowManager()
    with caplog.at_level(logging.INFO):
        await run_trading_cycle(manager, runtime=0.01)

    assert manager.cancelled is True
    assert manager.stopped is True
    assert any("Runtime limit reached" in record.message for record in caplog.records)


def test_assert_project_layout_rejects_foreign_modules(monkeypatch, tmp_path):
    import run_bot

    foreign_root = tmp_path / "external"
    package_root = foreign_root / "services"
    package_root.mkdir(parents=True)
    init_file = package_root / "__init__.py"
    init_file.write_text("# dummy external services package\n")

    external_spec = importlib.util.spec_from_file_location(
        "services",
        init_file,
        submodule_search_locations=[str(package_root)],
    )

    original_find_spec = importlib.util.find_spec

    def _fake_find_spec(name):
        if name == "services":
            return external_spec
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    with pytest.raises(SystemExit) as excinfo:
        run_bot._assert_project_layout()

    message = str(excinfo.value)
    assert "постороннее" in message
    assert "services" in message
    assert str(package_root) in message


def test_assert_project_layout_allows_partial_clone(monkeypatch, caplog):
    import run_bot

    def _fake_find_spec(name):
        return None

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    with caplog.at_level(logging.WARNING):
        run_bot._assert_project_layout(allow_partial=True)

    assert any("деградированном" in record.message for record in caplog.records)


def test_assert_project_layout_honors_env_root(monkeypatch, tmp_path, caplog):
    import run_bot

    root = tmp_path / "packed"
    for name in ("services", "data_handler", "model_builder", "bot"):
        (root / name).mkdir(parents=True)

    monkeypatch.setenv("BOT_PROJECT_ROOT", str(root))

    with caplog.at_level(logging.WARNING):
        run_bot._assert_project_layout(allow_partial=True)

    assert any(str(root) in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_main_uses_strict_layout_check_by_default(monkeypatch, tmp_path):
    import run_bot

    called = {}

    def _fake_assert(*, allow_partial):
        called["allow_partial"] = allow_partial

    dummy_cfg = SimpleNamespace(cache_dir=str(tmp_path / "cache"), log_dir=str(tmp_path / "logs"))

    monkeypatch.setattr(run_bot, "_assert_project_layout", _fake_assert)
    monkeypatch.setattr(run_bot, "configure_environment", lambda *_, **__: False)
    monkeypatch.setattr(run_bot, "ensure_directories", lambda cfg: None)
    monkeypatch.setattr(run_bot, "_build_components", lambda *_: (object(), object(), object()))

    async def _fake_run_trading_cycle(*_, **__):
        return None

    monkeypatch.setattr(run_bot, "run_trading_cycle", _fake_run_trading_cycle)
    monkeypatch.setattr(run_bot, "_log_mode", lambda *_: None)

    import bot.config as config_module
    import bot.dotenv_utils as dotenv_utils
    import bot.utils as utils_module

    monkeypatch.setattr(dotenv_utils, "load_dotenv", lambda: None)
    monkeypatch.setattr(config_module, "load_config", lambda *_: dummy_cfg)
    monkeypatch.setattr(utils_module, "configure_logging", lambda: None)

    args = argparse.Namespace(
        config=str(tmp_path / "config.json"),
        offline=False,
        auto_offline=True,
        allow_partial_clone=False,
        runtime=0,
        symbols=None,
        command="trade",
    )

    monkeypatch.setattr(run_bot, "parse_args", lambda: args)

    await run_bot.main()

    assert called["allow_partial"] is False
