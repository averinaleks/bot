import builtins
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
async def test_maybe_load_initial_propagates_unexpected_errors():
    class DummyHandler:
        def load_initial(self):
            raise ValueError("invalid state")

    handler = DummyHandler()

    with pytest.raises(ValueError, match="invalid state"):
        await _maybe_load_initial(handler)


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
