import logging

import pytest

import run_bot


class _FailingLoader:
    def load_initial(self):  # noqa: D401 - test helper
        """Always fail to emulate broken bootstrap."""

        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_maybe_load_initial_non_strict_logs_warning(caplog):
    caplog.set_level(logging.WARNING, logger="TradingBot")

    handler = _FailingLoader()

    await run_bot._maybe_load_initial(handler, strict=False)

    assert any("Initial data load failed" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_maybe_load_initial_strict_propagates_error():
    handler = _FailingLoader()

    with pytest.raises(RuntimeError, match="boom"):
        await run_bot._maybe_load_initial(handler, strict=True)
