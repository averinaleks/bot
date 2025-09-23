import asyncio
import errno
import hashlib
import logging
import os
import stat
import sys
import threading
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from bot import telegram_logger as telegram_logger_module
from bot.telegram_logger import TelegramLogger, resolve_unsent_path

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


@pytest.mark.asyncio
async def test_worker_thread_stops_after_shutdown(monkeypatch, fast_sleep):
    monkeypatch.delenv("TEST_MODE", raising=False)

    class _Bot:
        async def send_message(self, chat_id, text):
            return types.SimpleNamespace(message_id=1)

    await TelegramLogger.shutdown()
    start_threads = threading.active_count()
    tl = TelegramLogger(_Bot(), chat_id=1)

    await asyncio.sleep(0)
    assert (
        threading.active_count() > start_threads
        or tl._worker_task is not None
    )

    await TelegramLogger.shutdown()
    await asyncio.sleep(0)
    assert (
        threading.active_count() <= start_threads
        and tl._worker_task is None
    )


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

    chat_id, text, urgent = tl._queue.get_nowait()
    await tl._send(text, chat_id, urgent)
    tl._queue.task_done()

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


@pytest.mark.asyncio
async def test_worker_propagates_unexpected_exception(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")

    class Bot:
        async def send_message(self, chat_id, text):
            return types.SimpleNamespace(message_id=1)

    tl = TelegramLogger(Bot(), chat_id=1)

    class CustomError(Exception):
        pass

    async def failing_send(self, text, chat_id, urgent):
        raise CustomError

    tl._send = types.MethodType(failing_send, tl)
    tl._queue.put_nowait((1, "boom", False))

    with pytest.raises(CustomError):
        await tl._worker()

    await TelegramLogger.shutdown()


@pytest.mark.asyncio
async def test_two_loggers_independent(monkeypatch):
    monkeypatch.delenv("TEST_MODE", raising=False)

    class _Bot:
        async def send_message(self, chat_id, text):
            return types.SimpleNamespace(message_id=1)

    await TelegramLogger.shutdown()
    tl1 = TelegramLogger(_Bot(), chat_id=1)
    tl2 = TelegramLogger(_Bot(), chat_id=2)

    await tl1.send_telegram_message("one")
    await tl2.send_telegram_message("two")

    msg1 = tl1._queue.get_nowait()
    msg2 = tl2._queue.get_nowait()
    assert msg1[1] == "one"
    assert msg2[1] == "two"

    await TelegramLogger.shutdown()
    assert tl1._worker_task is None
    assert tl2._worker_task is None


@pytest.mark.asyncio
async def test_hash_filter(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")

    class CaptureBot:
        def __init__(self):
            self.sent = []

        async def send_message(self, chat_id, text):
            self.sent.append(text)
            return types.SimpleNamespace(message_id=len(self.sent))

    bot = CaptureBot()
    tl = TelegramLogger(bot, chat_id=1)

    message = "duplicate"
    await tl.send_telegram_message(message, urgent=True)
    chat_id, text, urgent = tl._queue.get_nowait()
    await tl._send(text, chat_id, urgent)
    tl._queue.task_done()

    first_hash = tl.last_hash
    expected_hash = hashlib.blake2s(
        message.encode("utf-8"), digest_size=16
    ).hexdigest()
    assert first_hash == expected_hash

    await tl.send_telegram_message(message, urgent=True)
    chat_id, text, urgent = tl._queue.get_nowait()
    await tl._send(text, chat_id, urgent)
    tl._queue.task_done()

    await TelegramLogger.shutdown()

    assert len(bot.sent) == 1
    assert tl.last_hash == first_hash


def test_resolve_unsent_path_confines_to_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    resolved = resolve_unsent_path(log_dir, "unsent.log")
    assert resolved == log_dir / "unsent.log"


def test_resolve_unsent_path_rejects_escape(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    with pytest.raises(ValueError):
        resolve_unsent_path(log_dir, "../unsent.log")


def test_resolve_unsent_path_rejects_absolute(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    with pytest.raises(ValueError):
        resolve_unsent_path(log_dir, tmp_path / "unsent.log")


def test_save_unsent_writes_to_resolved_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    resolved = resolve_unsent_path(log_dir, "unsent.log")
    tl = TelegramLogger(DummyBot(), chat_id=1, unsent_path=resolved)
    tl._save_unsent(42, "payload")
    assert resolved.read_text(encoding="utf-8").strip() == "42\tpayload"
    asyncio.run(TelegramLogger.shutdown())


def test_save_unsent_rejects_symlink(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("TEST_MODE", "1")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    resolved = resolve_unsent_path(log_dir, "unsent.log")
    target = tmp_path / "outside.txt"
    target.write_text("sentinel", encoding="utf-8")
    if resolved.exists():
        resolved.unlink()
    resolved.symlink_to(target)
    tl = TelegramLogger(DummyBot(), chat_id=1, unsent_path=resolved)
    caplog.set_level(logging.WARNING)
    tl._save_unsent(24, "payload")
    assert target.read_text(encoding="utf-8") == "sentinel"
    assert any(
        "Refusing to write Telegram fallback message" in rec.message
        for rec in caplog.records
    )
    asyncio.run(TelegramLogger.shutdown())


def test_save_unsent_handles_symlink_race(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("TEST_MODE", "1")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    resolved = resolve_unsent_path(log_dir, "unsent.log")
    tl = TelegramLogger(DummyBot(), chat_id=1, unsent_path=resolved)

    def raise_elooop(*_args, **_kwargs):
        raise OSError(errno.ELOOP, "symlink loop")

    monkeypatch.setattr(telegram_logger_module.os, "open", raise_elooop)
    caplog.set_level(logging.WARNING)

    tl._save_unsent(1, "payload")

    assert not resolved.exists()
    assert any(
        "Refusing to write Telegram fallback message" in rec.message
        for rec in caplog.records
    )
    asyncio.run(TelegramLogger.shutdown())


def test_save_unsent_sets_strict_file_permissions(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    resolved = resolve_unsent_path(log_dir, "unsent.log")
    tl = TelegramLogger(DummyBot(), chat_id=1, unsent_path=resolved)

    tl._save_unsent(99, "secure")

    assert stat.S_IMODE(resolved.stat().st_mode) == 0o600
    asyncio.run(TelegramLogger.shutdown())
