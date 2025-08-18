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


@pytest.mark.asyncio
async def test_shutdown_async_tasks_completes_pending_tasks():
    done = False

    async def coro():
        nonlocal done
        await asyncio.sleep(0)
        done = True

    trading_bot.run_async(coro())
    await trading_bot.shutdown_async_tasks()
    assert done
    assert not trading_bot._TASKS


@pytest.mark.asyncio
async def test_close_http_client_shuts_down_tasks():
    done = False

    async def coro():
        nonlocal done
        await asyncio.sleep(0)
        done = True

    trading_bot.run_async(coro())
    await trading_bot.close_http_client()
    assert done
    assert not trading_bot._TASKS
