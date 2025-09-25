"""Shared helpers for the TradeManager service and CLI entry points.

Этот модуль инкапсулирует повторяющуюся логику, используемую Flask-службой
`bot.trade_manager.service` и пакетным интерфейсом `bot.trade_manager`. В него
вынесены функции загрузки конфигурации, проверки токена доступа и создания
экземпляра :class:`TradeManager`. Это позволяет гарантировать, что обе точки
входа используют единый код и разделяют общее состояние.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from bot.dotenv_utils import load_dotenv
from bot.ray_compat import ray

from .core import (
    IS_TEST_MODE as CORE_TEST_MODE,
    TradeManager,
    _register_cleanup_handlers,
    configure_logging,
    logger,
)
from bot.config import load_config

__all__ = [
    "AuthorizationError",
    "IS_TEST_MODE",
    "TRADE_MANAGER_TOKEN",
    "create_trade_manager",
    "get_trade_manager",
    "load_environment",
    "load_trade_manager_config",
    "require_token",
    "set_trade_manager",
]


TRADE_MANAGER_TOKEN: str | None = os.getenv("TRADE_MANAGER_TOKEN")
"""Секретный токен авторизации для REST API."""

IS_TEST_MODE = os.getenv("TEST_MODE") == "1" or CORE_TEST_MODE
"""Глобальный флаг тестового режима."""

_trade_manager: TradeManager | None = None


@dataclass(slots=True)
class AuthorizationError:
    """Информация об ошибке авторизации."""

    reason: str
    status_code: int = 401


def load_environment() -> str | None:
    """Загрузить переменные окружения и обновить токен доступа."""

    global TRADE_MANAGER_TOKEN
    load_dotenv()
    TRADE_MANAGER_TOKEN = os.getenv("TRADE_MANAGER_TOKEN")
    if not TRADE_MANAGER_TOKEN:
        logger.warning(
            "TRADE_MANAGER_TOKEN пуст, все торговые запросы будут отвергнуты"
        )
    return TRADE_MANAGER_TOKEN


def require_token(
    headers: Mapping[str, str],
    *,
    token: str | None = None,
    test_mode: bool | None = None,
) -> AuthorizationError | None:
    """Проверить корректность токена в заголовках запроса."""

    if test_mode is None:
        test_mode = IS_TEST_MODE
    if test_mode:
        return None

    expected = token if token is not None else TRADE_MANAGER_TOKEN
    if not expected:
        return AuthorizationError("missing token")

    provided = headers.get("Authorization", "").strip()
    if provided != f"Bearer {expected}":
        return AuthorizationError("token mismatch")
    return None


def load_trade_manager_config(path: str = "config.json") -> dict[str, Any]:
    """Загрузить конфигурацию торгового бота из указанного файла."""

    logger.info("Загрузка конфигурации из %s", path)
    try:
        cfg = load_config(path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception(
            "Failed to load configuration (%s): %s",
            type(exc).__name__,
            exc,
        )
        raise
    logger.info("Конфигурация успешно загружена")
    return cfg


def get_trade_manager() -> TradeManager | None:
    """Вернуть текущий экземпляр :class:`TradeManager`."""

    return _trade_manager


def set_trade_manager(value: TradeManager | None) -> None:
    """Обновить глобальный экземпляр :class:`TradeManager`."""

    global _trade_manager
    _trade_manager = value


async def create_trade_manager(
    config_path: str = "config.json",
) -> TradeManager | None:
    """Создать и инициализировать :class:`TradeManager`."""

    tm = get_trade_manager()
    if tm is not None:
        return tm

    cfg = load_trade_manager_config(config_path)

    if not ray.is_initialized():
        from security import apply_ray_security_defaults

        logger.info(
            "Инициализация Ray: num_cpus=%s, num_gpus=1",
            cfg["ray_num_cpus"],
        )
        try:
            ray.init(
                **apply_ray_security_defaults(
                    {
                        "num_cpus": cfg["ray_num_cpus"],
                        "num_gpus": 1,
                        "ignore_reinit_error": True,
                    }
                )
            )
            logger.info("Ray успешно инициализирован")
        except RuntimeError as exc:
            logger.exception(
                "Ray initialization failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            raise

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    telegram_bot = None
    if token:
        try:
            from telegram import Bot

            telegram_bot = Bot(token)
            try:
                await telegram_bot.delete_webhook(drop_pending_updates=True)
                logger.info("Удалён существующий Telegram webhook")
            except httpx.HTTPError as exc:  # pragma: no cover - delete_webhook errors
                logger.exception(
                    "Failed to delete Telegram webhook (%s): %s",
                    type(exc).__name__,
                    exc,
                )
        except (RuntimeError, httpx.HTTPError) as exc:  # pragma: no cover - import/runtime errors
            logger.exception(
                "Не удалось создать Telegram Bot (%s): %s",
                type(exc).__name__,
                exc,
            )
            raise

    from bot.data_handler import DataHandler
    from bot.model_builder import ModelBuilder

    logger.info("Создание DataHandler")
    try:
        dh = DataHandler(cfg, telegram_bot, chat_id)
        logger.info("DataHandler успешно создан")
    except RuntimeError as exc:
        logger.exception(
            "Не удалось создать DataHandler (%s): %s",
            type(exc).__name__,
            exc,
        )
        raise

    logger.info("Создание ModelBuilder")
    try:
        mb = ModelBuilder(cfg, dh, None)
        dh.feature_callback = mb.precompute_features
        logger.info("ModelBuilder успешно создан")
        asyncio.create_task(mb.train())
        asyncio.create_task(mb.backtest_loop())
        await dh.load_initial()
        asyncio.create_task(dh.subscribe_to_klines(dh.usdt_pairs))
    except RuntimeError as exc:
        logger.error("Не удалось загрузить исходные данные: %s", exc)
        await dh.stop()
        return None
    except (ValueError, ImportError) as exc:
        logger.exception(
            "Не удалось создать ModelBuilder (%s): %s",
            type(exc).__name__,
            exc,
        )
        raise

    tm = TradeManager(cfg, dh, mb, telegram_bot, chat_id)
    set_trade_manager(tm)
    logger.info("Экземпляр TradeManager создан")

    if telegram_bot:
        from bot.utils import TelegramUpdateListener

        listener = TelegramUpdateListener(telegram_bot)

        async def handle_command(update):
            msg = getattr(update, "message", None)
            if not msg or not msg.text:
                return
            text = msg.text.strip().lower()
            import trading_bot as tb

            if text.startswith("/start"):
                await tb.set_trading_enabled(True)
                try:
                    await telegram_bot.send_message(
                        chat_id=msg.chat_id, text="Trading enabled"
                    )
                except Exception as exc:  # pragma: no cover - network errors
                    logger.error(
                        "Не удалось отправить сообщение в Telegram: %s",
                        exc,
                    )
            elif text.startswith("/stop"):
                await tb.set_trading_enabled(False)
                try:
                    await telegram_bot.send_message(
                        chat_id=msg.chat_id, text="Trading disabled"
                    )
                except Exception as exc:  # pragma: no cover - network errors
                    logger.error(
                        "Не удалось отправить сообщение в Telegram: %s",
                        exc,
                    )
            elif text.startswith("/status"):
                status = "enabled" if await tb.get_trading_enabled() else "disabled"
                positions = []
                tm_local = get_trade_manager()
                if tm_local is not None:
                    try:
                        res = tm_local.get_open_positions()
                        positions = (
                            await res if inspect.isawaitable(res) else res
                        ) or []
                    except Exception as exc:  # pragma: no cover - log and ignore
                        logger.error(
                            "Не удалось получить открытые позиции: %s",
                            exc,
                        )
                message = f"Trading {status}"
                if positions:
                    message += "\n" + "\n".join(str(p) for p in positions)
                try:
                    await telegram_bot.send_message(
                        chat_id=msg.chat_id, text=message
                    )
                except Exception as exc:  # pragma: no cover - network errors
                    logger.error(
                        "Не удалось отправить сообщение в Telegram: %s",
                        exc,
                    )

        threading.Thread(
            target=lambda: asyncio.run(listener.listen(handle_command)),
            daemon=True,
        ).start()
        setattr(tm, "_listener", listener)

    if not IS_TEST_MODE:
        _register_cleanup_handlers(tm)

    return tm


def configure_service_environment() -> None:
    """Подготовить окружение сервиса (логирование и multiprocessing)."""

    configure_logging()
