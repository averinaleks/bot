import asyncio
import logging
import pytest

from bot import trading_bot


@pytest.mark.asyncio
async def test_run_async_logs_exception(caplog):
    caplog.set_level(logging.ERROR)

    async def boom():
        raise RuntimeError("boom")

    trading_bot.run_async(boom())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    messages = [record.getMessage() for record in caplog.records]
    assert any("run_async task failed" in m for m in messages)
    assert not trading_bot._TASKS
