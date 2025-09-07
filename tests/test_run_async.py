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
    await asyncio.sleep(0)
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
    await asyncio.sleep(0)
    await trading_bot.close_http_client()
    assert done
    assert not trading_bot._TASKS


@pytest.mark.asyncio
async def test_run_async_times_out(caplog):
    caplog.set_level(logging.ERROR)

    async def coro():
        await asyncio.sleep(1)

    trading_bot.run_async(coro(), timeout=0.01)
    await asyncio.sleep(0.05)
    await asyncio.sleep(0)

    messages = [record.getMessage() for record in caplog.records]
    assert any("run_async task failed" in m for m in messages)
    assert not trading_bot._TASKS


@pytest.mark.asyncio
async def test_shutdown_async_tasks_cancels_pending_tasks(caplog):
    caplog.set_level(logging.WARNING)
    started = asyncio.Event()

    async def coro():
        started.set()
        await asyncio.sleep(1)

    trading_bot.run_async(coro())
    await started.wait()
    await trading_bot.shutdown_async_tasks(timeout=0.01)

    messages = [record.getMessage() for record in caplog.records]
    assert any("Cancelling pending tasks" in m for m in messages)
    assert not trading_bot._TASKS
