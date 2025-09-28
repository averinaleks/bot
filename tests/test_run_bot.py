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


from run_bot import run_trading_cycle


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

    assert str(excinfo.value) == "domain failure"

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
