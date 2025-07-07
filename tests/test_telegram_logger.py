import logging
import os
import types
import asyncio
import importlib.util
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


def test_emit_without_running_loop_no_exception():
    os.environ["TEST_MODE"] = "1"

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
    os.environ.pop("TEST_MODE", None)

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
        if threading.active_count() > start_threads:
            break
        time.sleep(0.05)
    assert threading.active_count() > start_threads

    asyncio.run(mod.TelegramLogger.shutdown())
    for _ in range(20):
        if threading.active_count() <= start_threads:
            break
        time.sleep(0.05)
    assert threading.active_count() <= start_threads
