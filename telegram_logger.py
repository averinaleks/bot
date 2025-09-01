"""Асинхронный логгер для отправки сообщений в Telegram.

Реализует очередь сообщений и повторные попытки с экспоненциальной
задержкой. При неудачной отправке сообщение можно сохранить во
внешний файл для последующей обработки.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Optional

import httpx
try:  # pragma: no cover - optional dependency
    from telegram.error import BadRequest, Forbidden, RetryAfter
except Exception as exc:  # pragma: no cover - missing telegram
    logging.getLogger("TradingBot").error("Telegram package not available: %s", exc)

    class _TelegramError(Exception):
        pass

    BadRequest = Forbidden = RetryAfter = _TelegramError  # type: ignore


logger = logging.getLogger("TradingBot")


class TelegramLogger(logging.Handler):
    """Handler для пересылки логов и сообщений в Telegram."""

    _queue: asyncio.Queue | None = None
    _worker_task: asyncio.Task | None = None
    _worker_thread: threading.Thread | None = None
    _worker_lock = asyncio.Lock()
    _bot: Any | None = None
    _stop_event: asyncio.Event | None = None

    def __init__(
        self,
        bot,
        chat_id,
        level: int = logging.NOTSET,
        max_queue_size: int | None = None,
        unsent_path: Optional[str] = None,
    ) -> None:
        super().__init__(level)
        self.bot = bot
        self.chat_id = chat_id
        self.unsent_path = unsent_path
        self.last_message_time = 0.0
        self.message_interval = 1800
        self.message_lock = asyncio.Lock()

        TelegramLogger._queue = asyncio.Queue(maxsize=max_queue_size or 0)
        TelegramLogger._bot = bot
        TelegramLogger._stop_event = asyncio.Event()
        if os.getenv("TEST_MODE") != "1":
            try:
                loop = asyncio.get_running_loop()
                TelegramLogger._worker_task = loop.create_task(self._worker())
            except RuntimeError:
                t = threading.Thread(
                    target=lambda: asyncio.run(self._worker()),
                    daemon=True,
                )
                t.start()
                TelegramLogger._worker_thread = t

        self.last_sent_text = ""

    async def _worker(self) -> None:
        while True:
            if TelegramLogger._queue is None or TelegramLogger._stop_event is None:
                return
            if TelegramLogger._stop_event.is_set():
                return
            queue = TelegramLogger._queue
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
            except (RetryAfter, Forbidden, BadRequest, httpx.HTTPError) as exc:
                logger.error("Ошибка отправки сообщения в Telegram: %s", exc)
                if self.unsent_path:
                    self._save_unsent(chat_id, text)
            except Exception:
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
                if part == self.last_sent_text:
                    logger.debug("Повторное сообщение Telegram пропущено")
                    continue

                delay = 1
                for attempt in range(5):
                    try:
                        bot = TelegramLogger._bot
                        if bot is None:
                            raise RuntimeError("Telegram bot not initialized")
                        result = await bot.send_message(chat_id=chat_id, text=part)
                        if not getattr(result, "message_id", None):
                            logger.error(
                                "Telegram message response without message_id",
                            )
                        else:
                            self.last_sent_text = part
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
                            e,
                            attempt + 1,
                        )
                        if attempt < 4:
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, 60)
                        else:
                            raise
                    except (BadRequest, Forbidden) as e:
                        logger.error("Ошибка Telegram: %s", e)
                        raise
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.exception(
                            "Ошибка отправки сообщения Telegram: %s",
                            e,
                        )
                        raise

    def _save_unsent(self, chat_id: int | str, text: str) -> None:
        if self.unsent_path is None:
            return
        try:
            with open(self.unsent_path, "a", encoding="utf-8") as f:
                f.write(f"{chat_id}\t{text}\n")
        except OSError as exc:  # pragma: no cover - file system errors
            logger.error("Не удалось сохранить сообщение Telegram: %s", exc)

    async def send_telegram_message(self, message: str, urgent: bool = False) -> None:
        if TelegramLogger._queue is None:
            logger.warning("TelegramLogger queue is None, message dropped")
            return
        msg = message[:4096]
        try:
            TelegramLogger._queue.put_nowait((self.chat_id, msg, urgent))
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
            logger.error("Ошибка в TelegramLogger: %s", exc)

    @classmethod
    async def shutdown(cls) -> None:
        if cls._stop_event is None:
            return
        if cls._queue is not None:
            try:
                cls._queue.put_nowait((None, "", False))
                await cls._queue.join()
            except asyncio.QueueFull:
                pass
        if cls._stop_event is not None:
            cls._stop_event.set()
        if cls._worker_task is not None:
            cls._worker_task.cancel()
            try:
                await cls._worker_task
            except asyncio.CancelledError:
                pass
            cls._worker_task = None
        if cls._worker_thread is not None:
            cls._worker_thread.join()
            cls._worker_thread = None
        cls._queue = None
        cls._stop_event = None
        cls._bot = None

