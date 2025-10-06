"""Асинхронный логгер для отправки сообщений в Telegram.

Реализует очередь сообщений и повторные попытки с экспоненциальной
задержкой. При неудачной отправке сообщение можно сохранить во
внешний файл для последующей обработки.
"""

from __future__ import annotations

import asyncio
import atexit
import errno
import logging
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any, Optional

if "_TELEGRAM_IMPORT_LOGGED" not in globals():
    _TELEGRAM_IMPORT_LOGGED = False

try:  # pragma: no cover - optional dependency in lightweight environments
    import httpx
except Exception as exc:  # pragma: no cover - fallback when httpx absent
    class _HttpxStub:
        """Minimal stub providing :class:`HTTPError` when httpx is missing."""

        class HTTPError(Exception):
            """Fallback HTTPError used to preserve exception handling."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args)

    httpx = _HttpxStub()  # type: ignore[assignment]
    _HTTPX_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - executed when httpx installed
    _HTTPX_IMPORT_ERROR = None

import hashlib
# Use absolute import to ensure the local configuration module is loaded even
# when a similarly named module exists on ``PYTHONPATH``.
from bot import config as bot_config
from services.logging_utils import sanitize_log_value
from services.offline import OfflineTelegram

try:  # pragma: no cover - optional dependency
    from telegram.error import BadRequest, Forbidden, RetryAfter
except Exception as exc:  # pragma: no cover - missing telegram
    logger_ref = logging.getLogger("TradingBot")
    if not _TELEGRAM_IMPORT_LOGGED:
        logger_ref.error("Telegram package not available: %s", exc)
        _TELEGRAM_IMPORT_LOGGED = True
    else:
        logger_ref.debug("Telegram package not available: %s", exc)

    class _TelegramError(Exception):
        """Base class mirroring telegram.error.TelegramError."""

    class _BadRequest(_TelegramError):
        pass

    class _Forbidden(_TelegramError):
        pass

    class _RetryAfter(_TelegramError):
        def __init__(self, *args: Any, retry_after: float | None = None, **kwargs: Any) -> None:
            super().__init__(*args)
            self.retry_after = retry_after

    BadRequest = _BadRequest  # type: ignore
    Forbidden = _Forbidden  # type: ignore
    RetryAfter = _RetryAfter  # type: ignore


logger = logging.getLogger("TradingBot")


def resolve_unsent_path(
    log_dir: str | os.PathLike[str],
    candidate: str | os.PathLike[str],
) -> Path:
    """Return an absolute path for unsent Telegram messages confined to *log_dir*."""

    base = Path(log_dir).expanduser().resolve(strict=False)
    if not str(candidate):
        raise ValueError("unsent_telegram_path must not be empty")
    path = Path(candidate)
    if path.is_absolute():
        raise ValueError("unsent_telegram_path must be relative to log_dir")
    resolved = (base / path).resolve(strict=False)
    if not resolved.is_relative_to(base):
        raise ValueError("unsent_telegram_path escapes log_dir")
    return resolved


class TelegramLogger(logging.Handler):
    """Handler для пересылки логов и сообщений в Telegram."""

    _instances: set["TelegramLogger"] = set()

    def __new__(cls, *args, **kwargs):
        if _should_use_offline_logger():
            logger.warning(
                "Telegram credentials missing; using OfflineTelegram stub"
            )
            return OfflineTelegram(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        bot,
        chat_id,
        level: int = logging.NOTSET,
        max_queue_size: int | None = None,
        unsent_path: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        super().__init__(level)
        self.bot = bot
        self.chat_id = chat_id
        self.unsent_path = Path(unsent_path) if unsent_path is not None else None
        self.last_message_time = 0.0
        self.message_interval = 1800
        self.message_lock = asyncio.Lock()

        self._queue: asyncio.Queue | None = asyncio.Queue(
            maxsize=max_queue_size or 0
        )
        self._bot: Any | None = bot
        self._stop_event: asyncio.Event | None = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        self._worker_thread: threading.Thread | None = None

        if os.getenv("TEST_MODE") != "1":
            try:
                loop = asyncio.get_running_loop()
                self._worker_task = loop.create_task(self._worker())
            except RuntimeError:
                t = threading.Thread(
                    target=lambda: asyncio.run(self._worker()),
                    daemon=True,
                )
                t.start()
                self._worker_thread = t

        TelegramLogger._instances.add(self)

        self.last_hash = ""

    async def _worker(self) -> None:
        while True:
            if self._queue is None or self._stop_event is None:
                return
            if self._stop_event.is_set():
                return
            queue = self._queue
            try:
                item = await asyncio.wait_for(queue.get(), 1.0)
            except asyncio.TimeoutError:
                continue
            chat_id, text, urgent = item
            if chat_id is None:
                queue.task_done()
                return
            try:
                await self._send(text, chat_id, urgent)
            except (RetryAfter, Forbidden, BadRequest) as exc:
                logger.error(
                    "Ошибка отправки сообщения в Telegram: %s",
                    sanitize_log_value(str(exc)),
                )
                if self.unsent_path:
                    self._save_unsent(chat_id, text)
            except Exception as exc:
                is_httpx_error = isinstance(exc, httpx.HTTPError)
                httpx_is_generic = httpx.HTTPError is Exception or getattr(
                    getattr(httpx.HTTPError, "__module__", ""),
                    "lower",
                    lambda: "",
                )().startswith("builtins")
                if is_httpx_error and not httpx_is_generic:
                    logger.error(
                        "Ошибка отправки сообщения в Telegram: %s",
                        sanitize_log_value(str(exc)),
                    )
                    if self.unsent_path:
                        self._save_unsent(chat_id, text)
                else:
                    logger.exception("Непредвиденная ошибка отправки Telegram")
                    raise
            finally:
                queue.task_done()
            await asyncio.sleep(1)

    async def _send(self, message: str, chat_id: int | str, urgent: bool) -> None:
        async with self.message_lock:
            if (
                not urgent
                and time.time() - self.last_message_time < self.message_interval
            ):
                logger.debug(
                    "Сообщение Telegram пропущено из-за интервала: %s...",
                    message[:100],
                )
                return

            parts = [message[i : i + 500] for i in range(0, len(message), 500)]
            for part in parts:
                part_hash = hashlib.blake2s(
                    part.encode("utf-8"), digest_size=16
                ).hexdigest()
                if part_hash == self.last_hash:
                    logger.debug("Повторное сообщение Telegram пропущено")
                    continue

                delay = 1
                for attempt in range(5):
                    try:
                        bot = self._bot
                        if bot is None:
                            raise RuntimeError("Telegram bot not initialized")
                        result = await bot.send_message(chat_id=chat_id, text=part)
                        if not getattr(result, "message_id", None):
                            logger.error(
                                "Telegram message response without message_id",
                            )
                        else:
                            self.last_hash = part_hash
                        self.last_message_time = time.time()
                        break
                    except RetryAfter as e:
                        wait_time = getattr(e, "retry_after", delay)
                        logger.warning("Flood control: ожидание %sс", wait_time)
                        await asyncio.sleep(wait_time)
                        delay = min(delay * 2, 60)
                    except httpx.HTTPError as e:
                        logger.warning(
                            "HTTP ошибка Telegram: %s. Попытка %s/5",
                            sanitize_log_value(str(e)),
                            attempt + 1,
                        )
                        if attempt < 4:
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, 60)
                        else:
                            raise
                    except (BadRequest, Forbidden) as e:
                        logger.error(
                            "Ошибка Telegram: %s",
                            sanitize_log_value(str(e)),
                        )
                        raise
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.exception(
                            "Ошибка отправки сообщения Telegram: %s",
                            sanitize_log_value(str(e)),
                        )
                        raise
    def _save_unsent(self, chat_id: int | str, text: str) -> None:
        if self.unsent_path is None:
            return

        def _append_secure(path: Path, payload: str) -> None:
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            flags |= getattr(os, "O_CLOEXEC", 0)
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            if nofollow:
                flags |= nofollow

            fd = os.open(path, flags, 0o600)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise OSError(
                        errno.EPERM, "unsent message path must be a regular file"
                    )

                data = payload.encode("utf-8")
                written = os.write(fd, data)
                if written != len(data):
                    raise OSError(errno.EIO, "failed to persist Telegram message")
            finally:
                os.close(fd)

        try:
            path = self.unsent_path
            if path.exists() and path.is_symlink():
                logger.warning(
                    "Refusing to write Telegram fallback message to symlink %s",
                    sanitize_log_value(str(path)),
                )
                return

            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                _append_secure(path, f"{chat_id}\t{text}\n")
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.EPERM}:
                    logger.warning(
                        "Refusing to write Telegram fallback message to %s: %s",
                        sanitize_log_value(str(path)),
                        sanitize_log_value(str(exc)),
                    )
                    return
                raise
        except OSError as exc:  # pragma: no cover - file system errors
            logger.error(
                "Не удалось сохранить сообщение Telegram: %s",
                sanitize_log_value(str(exc)),
            )

    async def send_telegram_message(self, message: str, urgent: bool = False) -> None:
        if self._queue is None:
            logger.warning("TelegramLogger queue is None, message dropped")
            return
        msg = message[:4096]
        try:
            self._queue.put_nowait((self.chat_id, msg, urgent))
        except asyncio.QueueFull:
            logger.warning("Очередь Telegram переполнена, сообщение пропущено")

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging
        try:
            msg = self.format(record)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.send_telegram_message(msg))
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.send_telegram_message(msg))
                    else:
                        raise RuntimeError
                except RuntimeError:
                    threading.Thread(
                        target=lambda: asyncio.run(self.send_telegram_message(msg)),
                        daemon=True,
                    ).start()
        except (ValueError, RuntimeError) as exc:
            logger.error(
                "Ошибка в TelegramLogger: %s",
                sanitize_log_value(str(exc)),
            )

    async def _shutdown(self) -> None:
        if self._stop_event is None:
            return

        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break
            if self._worker_task is not None or self._worker_thread is not None:
                try:
                    self._queue.put_nowait((None, "", False))
                    try:
                        await asyncio.wait_for(self._queue.join(), timeout=5)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Истекло время ожидания очистки очереди TelegramLogger при завершении",
                        )
                    except RuntimeError as exc:
                        logger.debug(
                            "Цикл событий недоступен при очистке очереди TelegramLogger: %s",
                            exc,
                        )
                except asyncio.QueueFull:
                    logger.warning(
                        "Не удалось добавить сигнал завершения в очередь TelegramLogger: очередь переполнена",
                    )

        if self._stop_event is not None:
            self._stop_event.set()

        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                logger.debug("Задача TelegramLogger была отменена при завершении")
            self._worker_task = None

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None

        self._queue = None
        self._stop_event = None
        self._bot = None

        TelegramLogger._instances.discard(self)

    @classmethod
    async def shutdown(cls) -> None:
        if _should_use_offline_logger():
            await OfflineTelegram.shutdown()
            return
        for inst in list(cls._instances):
            await inst._shutdown()


def _should_use_offline_logger() -> bool:
    if bot_config.OFFLINE_MODE:
        return True
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    if os.getenv("TEST_MODE") == "1":
        return False
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    return not (token and chat_id)


def _shutdown_all() -> None:
    coro = TelegramLogger.shutdown()
    try:
        asyncio.run(coro)
    except RuntimeError as exc:
        coro.close()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug(
                "Не удалось получить активный цикл событий для завершения TelegramLogger: %s",
                exc,
            )
        else:
            try:
                loop.create_task(TelegramLogger.shutdown())
            except RuntimeError as task_exc:
                logger.debug(
                    "Не удалось запланировать завершение TelegramLogger в активном цикле: %s",
                    task_exc,
                )


atexit.register(_shutdown_all)

