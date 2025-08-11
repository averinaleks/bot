import logging
import os
import types
import asyncio
import importlib.util
import pytest
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("utils_real", os.path.join(ROOT, "utils.py"))
utils_real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_real)
TelegramLogger = utils_real.TelegramLogger

class DummyBot:
    async def send_message(self, chat_id, text):
        return types.SimpleNamespace(message_id=1)


def test_emit_without_running_loop_no_exception(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")

    async def stub_send(self, message, urgent: bool = False):
        pass

    tl = TelegramLogger(DummyBot(), chat_id=123)
    tl.send_telegram_message = types.MethodType(stub_send, tl)

    logger = logging.getLogger("tl_test")
    logger.addHandler(tl)
    logger.setLevel(logging.ERROR)

    logger.error("test message")

    asyncio.run(TelegramLogger.shutdown())


def test_worker_thread_stops_after_shutdown():

    spec = importlib.util.spec_from_file_location("utils_real", os.path.join(ROOT, "utils.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    TL = mod.TelegramLogger

    class _Bot:
        async def send_message(self, chat_id, text):
            return types.SimpleNamespace(message_id=1)

    start_threads = threading.active_count()
    TL(_Bot(), chat_id=1)
    for _ in range(20):
        if threading.active_count() > start_threads or TL._worker_task is not None:
            break
        time.sleep(0.05)
    assert threading.active_count() > start_threads or TL._worker_task is not None

    asyncio.run(mod.TelegramLogger.shutdown())
    for _ in range(20):
        if threading.active_count() <= start_threads and TL._worker_task is None:
            break
        time.sleep(0.05)
    assert threading.active_count() <= start_threads and TL._worker_task is None


@pytest.mark.asyncio
async def test_long_message_split_into_parts(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")

    class CaptureBot:
        def __init__(self):
            self.sent = []

        async def send_message(self, chat_id, text):
            self.sent.append(text)
            return types.SimpleNamespace(message_id=len(self.sent))

    bot = CaptureBot()
    tl = TelegramLogger(bot, chat_id=1)

    long_message = "x" * 600
    await tl.send_telegram_message(long_message)

    chat_id, text, urgent = TelegramLogger._queue.get_nowait()
    await tl._send(text, chat_id, urgent)

    await TelegramLogger.shutdown()

    assert len(bot.sent) == 2
    assert bot.sent[0] == long_message[:500]
    assert bot.sent[1] == long_message[500:]
    assert "".join(bot.sent) == long_message


@pytest.mark.asyncio
async def test_send_after_shutdown_warning(monkeypatch, caplog):
    monkeypatch.setenv("TEST_MODE", "1")

    bot = DummyBot()
    tl = TelegramLogger(bot, chat_id=1)

    await TelegramLogger.shutdown()

    caplog.set_level(logging.WARNING)
    await tl.send_telegram_message("test")

    assert any(rec.levelno == logging.WARNING for rec in caplog.records)
