"""Miscellaneous helper utilities for the trading bot."""

import asyncio
import importlib
import inspect
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
import weakref
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from bot.host_utils import validate_host as _validate_host
from services.logging_utils import sanitize_log_value

try:  # pragma: no cover - optional dependency for HTTP error handling
    import httpx as _httpx_module
except Exception as exc:  # pragma: no cover - gracefully degrade when missing
    class _HttpxStub:
        class HTTPError(Exception):
            """Fallback HTTPError used when httpx is unavailable."""

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args)

    httpx = cast(Any, _HttpxStub())
    _HTTPX_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - executed in environments with httpx installed
    httpx = cast(Any, _httpx_module)
    _HTTPX_IMPORT_ERROR = None

validate_host = _validate_host


logger = logging.getLogger("TradingBot")


_STATE_MODULE_NAME = "bot._utils_state"
class _UtilsStateModule(ModuleType):
    """Private module wrapper storing shared state for utils helpers."""

    numba_aliases: List[weakref.ReferenceType[ModuleType]]
    numba_warning_seen: bool


class _UtilsModule(ModuleType):
    """Custom module that syncs local flags with shared state."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_NUMBA_IMPORT_WARNED":
            _state_module.numba_warning_seen = bool(value)
        super().__setattr__(name, value)


_existing_state = sys.modules.get(_STATE_MODULE_NAME)
if isinstance(_existing_state, ModuleType) and hasattr(_existing_state, "numba_aliases"):
    _state_module = cast(_UtilsStateModule, _existing_state)
else:
    _state_module = _UtilsStateModule(_STATE_MODULE_NAME)
    _state_module.numba_aliases = []
    sys.modules[_STATE_MODULE_NAME] = _state_module

_NUMBA_MODULE_ALIASES: List[weakref.ReferenceType[ModuleType]] = _state_module.numba_aliases

if not hasattr(_state_module, "numba_warning_seen"):
    _state_module.numba_warning_seen = False

_current_module = sys.modules.get(__name__)
if _current_module is not None:
    if not isinstance(_current_module, _UtilsModule):
        _current_module.__class__ = _UtilsModule
    for ref in list(_NUMBA_MODULE_ALIASES):
        if ref() is _current_module:
            break
    else:
        _NUMBA_MODULE_ALIASES.append(weakref.ref(_current_module))

_NUMBA_IMPORT_WARNED = _state_module.numba_warning_seen

if "_TELEGRAMLOGGER_IMPORT_WARNED" not in globals():
    _TELEGRAMLOGGER_IMPORT_WARNED = False

if "_TELEGRAMLOGGER_STUB_INIT_WARNED" not in globals():
    _TELEGRAMLOGGER_STUB_INIT_WARNED = False


class _TelegramLoggerStub:
    """Lightweight fallback used when Telegram logger is unavailable."""

    _is_stub = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - simple stub
        global _TELEGRAMLOGGER_STUB_INIT_WARNED
        if not _TELEGRAMLOGGER_STUB_INIT_WARNED:
            logger.warning("TelegramLogger is unavailable; notifications disabled")
            _TELEGRAMLOGGER_STUB_INIT_WARNED = True

    async def send_telegram_message(self, *args: Any, **kwargs: Any) -> None:
        logger.debug("TelegramLogger stub dropping message")

    @classmethod
    async def shutdown(cls) -> None:
        logger.debug("TelegramLogger stub shutdown invoked")


FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def _wrap_tempfile_mkdtemp() -> None:
    original = getattr(tempfile, "_bot_original_mkdtemp", None)
    if original is not None:
        return

    original = tempfile.mkdtemp
    setattr(tempfile, "_bot_original_mkdtemp", original)

    def _safe_mkdtemp(*args, **kwargs):
        target_dir = kwargs.get("dir")
        if target_dir is None:
            target_dir = tempfile.gettempdir()
        if target_dir:
            try:
                Path(target_dir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.error(
                    "Не удалось подготовить каталог временных файлов %s: %s",
                    sanitize_log_value(str(target_dir)),
                    sanitize_log_value(str(exc)),
                )
                raise RuntimeError(
                    "Не удалось подготовить каталог временных файлов"
                ) from exc
        try:
            return original(*args, **kwargs)
        except FileNotFoundError:
            if target_dir:
                Path(target_dir).mkdir(parents=True, exist_ok=True)
            return original(*args, **kwargs)

    tempfile.mkdtemp = _safe_mkdtemp


_wrap_tempfile_mkdtemp()


def reset_tempdir_cache() -> None:
    """Clear :mod:`tempfile`'s cached temp directory.

    ``tempfile`` stores the last successful directory in ``tempfile.tempdir``.
    Tests monkeypatch :func:`tempfile.gettempdir` to point at short-lived
    locations and expect the cache to be rebuilt after the directory is
    removed.  Without clearing the cache, later calls such as
    :func:`tempfile.mkdtemp` continue to use the now-deleted directory which
    results in ``FileNotFoundError`` during unrelated tests.  Resetting the
    cache forces :mod:`tempfile` to recompute the appropriate base directory on
    the next access.
    """

    try:
        tempfile.tempdir = None
    except (AttributeError, TypeError, PermissionError) as exc:
        # ``tempfile`` guarantees the attribute exists, but guard against
        # environments that restrict attribute assignment on the module.
        logger.warning(
            "Не удалось сбросить кеш временного каталога tempfile: %s",
            exc,
        )


def ensure_writable_directory(
    path: Path,
    *,
    description: str,
    fallback_subdir: str | None = None,
) -> Path:
    """Return a writable directory, falling back to a temp location if needed."""

    primary = path.resolve()
    candidates: list[Path] = []
    if str(primary):
        candidates.append(primary)

    if fallback_subdir:
        fallback = Path(tempfile.gettempdir(), fallback_subdir).resolve()
        reset_tempdir_cache()
        if fallback not in candidates:
            candidates.append(fallback)

    last_error: Exception | None = None

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Не удалось создать каталог %s %s: %s",
                description,
                candidate,
                exc,
            )
            last_error = exc
            continue

        if os.access(candidate, os.W_OK):
            if candidate != primary:
                logger.warning(
                    "Каталог %s %s недоступен для записи, используем %s",
                    description,
                    primary,
                    candidate,
                )
            return candidate

        logger.warning(
            "Каталог %s %s недоступен для записи",
            description,
            candidate,
        )
        last_error = PermissionError(
            f"Каталог {candidate} недоступен для записи"
        )

    checked = ", ".join(str(p) for p in candidates) if candidates else str(primary)
    logger.error(
        "Не удалось подготовить каталог %s. Проверенные пути: %s",
        description,
        checked,
    )
    if last_error is None:
        last_error = PermissionError(
            f"Не удалось подготовить каталог {description}: {checked}"
        )
    raise last_error


def retry(max_attempts: int, delay_fn: Callable[[float], float]) -> Callable[[FuncT], FuncT]:
    """Декоратор повторного выполнения функции с экспоненциальной задержкой.

    ``delay_fn`` получает базовую задержку ``2 ** (attempt - 1)`` и может
    модифицировать её (например, ограничить максимальное значение или добавить
    джиттер). Поддерживаются как обычные, так и асинхронные функции.
    """

    def decorator(func: FuncT) -> FuncT:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any):
                attempt = 1
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        if attempt >= max_attempts:
                            raise
                        delay = delay_fn(2 ** (attempt - 1))
                        await asyncio.sleep(delay)
                        attempt += 1

            return cast(FuncT, async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            attempt = 1
            while True:
                try:
                    return func(*args, **kwargs)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if attempt >= max_attempts:
                        raise
                    delay = delay_fn(2 ** (attempt - 1))
                    time.sleep(delay)
                    attempt += 1

        return cast(FuncT, sync_wrapper)

    return decorator


def suppress_tf_logs() -> None:
    """Установить ``TF_CPP_MIN_LOG_LEVEL=3`` для подавления логов TensorFlow."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def configure_logging() -> None:
    """Настроить переменные окружения и handlers для логирования."""

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    try:
        logger.setLevel(level_name)
    except ValueError:
        logger.warning(
            "LOG_LEVEL '%s' недопустим, используется INFO",
            sanitize_log_value(level_name),
        )
        logger.setLevel(logging.INFO)

    log_dir = os.getenv("LOG_DIR", "/app/logs")
    fallback_dir = os.path.join(os.path.dirname(__file__), "logs")
    fallback_used = False
    try:
        os.makedirs(log_dir, exist_ok=True)
        if not os.access(log_dir, os.W_OK):
            raise PermissionError
    except (OSError, PermissionError):
        log_dir = fallback_dir
        os.makedirs(log_dir, exist_ok=True)
        fallback_used = True

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    log_file_path = os.path.join(log_dir, "trading_bot.log")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Оставляем propagate включённым, чтобы caplog и внешние обработчики могли
    # перехватывать сообщения (важно для юнит-тестов и сторонних интеграций).
    logger.propagate = True

    logger.info(
        "Logging configured. File: %s, level: %s",
        log_file_path,
        logging.getLevelName(logger.level),
    )
    if fallback_used:
        logger.warning(
            "LOG_DIR '%s' недоступен для записи, используется резервная папка %s",
            sanitize_log_value(os.getenv("LOG_DIR", "/app/logs")),
            log_dir,
        )


# Hide Numba performance warnings
try:
    from numba import jit, prange, NumbaPerformanceWarning

    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except ImportError as exc:  # pragma: no cover - allow missing numba package
    if not _state_module.numba_warning_seen:
        logger.warning("Numba import failed: %s", exc)
        logger.warning(
            "Running without Numba JIT acceleration; performance may be degraded."
        )
        _state_module.numba_warning_seen = True
        _NUMBA_IMPORT_WARNED = True

    def jit(*jit_args, **jit_kwargs):
        if jit_args and callable(jit_args[0]) and len(jit_args) == 1 and not jit_kwargs:
            return jit_args[0]

        def decorator(func):
            return func

        return decorator

    def prange(*args: int, **kwargs: int) -> range:
        return range(*args, **kwargs)

try:
    import numpy as np
except ImportError:
    np = cast(Any, None)

try:  # pragma: no cover - prefer package import to avoid shadowing
    from bot.telegram_logger import TelegramLogger as _TelegramLoggerImpl  # noqa: F401
except Exception as exc:  # pragma: no cover - fallback when package import fails
    if not _TELEGRAMLOGGER_IMPORT_WARNED:
        logger.warning(
            "Failed to import TelegramLogger via package: %s",
            sanitize_log_value(str(exc)),
        )
        _TELEGRAMLOGGER_IMPORT_WARNED = True
    try:
        from telegram_logger import TelegramLogger as _TelegramLoggerImpl  # noqa: F401
    except Exception:
        _TelegramLoggerImpl = _TelegramLoggerStub

def _make_safe_telegram_logger(logger_cls: Any) -> Any:
    if getattr(logger_cls, "_is_stub", False):
        return logger_cls

    class _TelegramLoggerFactory:
        __slots__ = ("_base",)

        def __init__(self, base: Any) -> None:
            self._base = base

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            bot = kwargs.get("bot") if kwargs else None
            chat_id = kwargs.get("chat_id") if kwargs else None
            if args:
                bot = args[0]
            if len(args) > 1:
                chat_id = args[1]

            if bot is None or chat_id is None:
                return _TelegramLoggerStub(*args, **kwargs)

            try:
                return self._base(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - guard against runtime failures
                logger.warning(
                    "Failed to instantiate TelegramLogger; falling back to stub: %s",
                    sanitize_log_value(str(exc)),
                )
                return _TelegramLoggerStub(*args, **kwargs)

        async def shutdown(self) -> None:
            shutdown = getattr(self._base, "shutdown", None)
            if shutdown is not None:
                return await shutdown()
            return await _TelegramLoggerStub.shutdown()

        def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple delegation
            return getattr(self._base, item)

    return _TelegramLoggerFactory(logger_cls)


TelegramLogger = _make_safe_telegram_logger(_TelegramLoggerImpl)


RetryAfter: type[Exception]
BadRequest: type[Exception]
Forbidden: type[Exception]

try:
    from telegram.error import (
        BadRequest as _TelegramBadRequest,
        Forbidden as _TelegramForbidden,
        RetryAfter as _TelegramRetryAfter,
    )
except ImportError as exc:  # pragma: no cover - allow missing telegram package
    logging.getLogger("TradingBot").error(
        "Telegram package not available: %s",
        sanitize_log_value(str(exc)),
    )

    class _RetryAfterFallback(Exception):
        """Fallback RetryAfter error when python-telegram-bot is unavailable."""

    class _BadRequestFallback(Exception):
        """Fallback BadRequest error when python-telegram-bot is unavailable."""

    class _ForbiddenFallback(Exception):
        """Fallback Forbidden error when python-telegram-bot is unavailable."""

    RetryAfter = _RetryAfterFallback
    BadRequest = _BadRequestFallback
    Forbidden = _ForbiddenFallback
else:
    RetryAfter = _TelegramRetryAfter
    BadRequest = _TelegramBadRequest
    Forbidden = _TelegramForbidden


HTTP: type
try:
    from pybit.unified_trading import HTTP as _PybitHTTP
except ImportError:  # pragma: no cover - pybit is optional in CI
    class _HTTPFallback:
        """Fallback HTTP stub when pybit is unavailable."""

        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
            raise ImportError("pybit is required for HTTP operations")
    HTTP = _HTTPFallback
else:
    HTTP = _PybitHTTP

# Mapping from ccxt/ccxtpro style timeframes to Bybit interval strings
_BYBIT_INTERVALS = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}

def safe_int(
    value: str | int | None, default: int = 0, *, env_var: str | None = None
) -> int:
    """Safely convert ``value`` to a positive integer.

    Any ``None`` value results in ``default`` without logging. If ``env_var`` is
    provided, invalid or non-positive inputs are logged as warnings.
    """

    if value is None:
        return default
    try:
        result = int(value)
        if result <= 0:
            if env_var:
                logger.warning(
                    "Non-positive %s value '%s', using default %s",
                    sanitize_log_value(env_var),
                    sanitize_log_value(value),
                    default,
                )
            return default
        return result
    except (TypeError, ValueError):
        if env_var:
            logger.warning(
                "Invalid %s value '%s', using default %s",
                sanitize_log_value(env_var),
                sanitize_log_value(value),
                default,
            )
        return default


def _call_bool(func: Callable[[], Any] | None) -> bool:
    """Call ``func`` if callable and coerce the result to ``bool``."""

    if func is None:
        return False
    result = func()
    return bool(result)


def is_cuda_available() -> bool:
    """Safely check whether CUDA is available via PyTorch."""

    # Allow forcing CPU mode via environment variables before importing torch
    if os.environ.get("FORCE_CPU") == "1":
        return False

    nvd = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if nvd is not None and (nvd == "" or nvd.lower() == "none"):
        return False

    # Avoid importing torch when no NVIDIA driver is present. "nvidia-smi" will
    # normally be available only on systems with working CUDA drivers. This
    # short-circuit prevents crashes like "free(): double free detected" that can
    # happen when torch tries to initialise CUDA in a broken environment.
    if shutil.which("nvidia-smi") is None:
        return False

    try:
        torch_mod = importlib.import_module("torch")
        backends = getattr(torch_mod, "backends", None)
        cuda_backend = getattr(backends, "cuda", None)
        if not _call_bool(getattr(cuda_backend, "is_built", None)):
            return False

        cuda_module = getattr(torch_mod, "cuda", None)
        return _call_bool(getattr(cuda_module, "is_available", None))
    except ImportError as exc:  # pragma: no cover - optional dependency
        logging.getLogger("TradingBot").warning(
            "CUDA availability check failed: %s", exc
        )
        return False
    except (OSError, RuntimeError) as exc:  # pragma: no cover - unexpected error
        logging.getLogger("TradingBot").exception(
            "Unexpected CUDA availability error: %s", exc
        )
        raise


def bybit_interval(timeframe: str) -> str:
    """Return the interval string accepted by Bybit APIs."""

    return _BYBIT_INTERVALS.get(timeframe, timeframe)


async def handle_rate_limits(exchange) -> None:
    """Sleep if Bybit rate limit is close to exhaustion."""
    headers = getattr(exchange, "last_response_headers", {}) or {}
    try:
        remaining = int(
            headers.get("X-Bapi-Limit-Status", headers.get("x-bapi-limit-status", 0))
        )
        reset_ts = int(
            headers.get(
                "X-Bapi-Limit-Reset-Timestamp",
                headers.get("x-bapi-limit-reset-timestamp", 0),
            )
        )
    except ValueError:
        return
    if remaining and remaining <= 5:
        wait_time = max(0.0, reset_ts / 1000 - time.time())
        if wait_time > 0:
            logger.info(
                "Rate limit low (%s), sleeping %.2fs",
                remaining,
                wait_time,
            )
            await asyncio.sleep(wait_time)


async def safe_api_call(
    exchange,
    method: str,
    *args,
    max_attempts: int = 5,
    backoff_factor: float = 2,
    test_mode: bool | None = None,
    **kwargs,
):
    """Call a ccxt method with retry, status and retCode verification."""
    if test_mode is None:
        test_mode = os.getenv("TEST_MODE", "").strip().lower() in {"1", "true", "yes"}

    if test_mode:
        # During unit tests we do not want to spend time in the retry loop.
        return await getattr(exchange, method)(*args, **kwargs)

    retriable_exceptions: tuple[type[BaseException], ...] = (
        httpx.HTTPError,
        RuntimeError,
        ValueError,
        TypeError,
    )

    for attempt in range(max_attempts):
        try:
            result = await getattr(exchange, method)(*args, **kwargs)
            await handle_rate_limits(exchange)

            status = getattr(exchange, "last_http_status", 200)
            if status != 200:
                raise RuntimeError(f"HTTP {status}")

            if isinstance(result, dict):
                ret_code = result.get("retCode") or result.get("ret_code")
                if ret_code is not None and ret_code != 0:
                    raise RuntimeError(f"retCode {ret_code}")

            return result
        except retriable_exceptions as exc:
            logger.error("Bybit API error in %s: %s", method, exc)
            if "10002" in str(exc):
                logger.error(
                    "Request not authorized. Check server time sync and recv_window"
                )
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(backoff_factor ** attempt)
        except Exception as exc:  # noqa: BLE001 - deliberate broad guard for observability
            logger.error(
                "Unhandled exception in Bybit API call %s: %s", method, exc, exc_info=True
            )
            raise


class BybitSDKAsync:
    """Asynchronous wrapper around the official Bybit SDK."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self.client = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            return_response_headers=True,
        )
        self.last_http_status = 200
        self.last_response_headers: Dict = {}
        # Clear credentials after initializing the client to avoid lingering secrets
        del api_key, api_secret

    def _call_client(self, method: str, *args, **kwargs):
        res = None
        try:
            res = getattr(self.client, method)(*args, **kwargs)
        except httpx.HTTPError as exc:  # pragma: no cover - network/library errors
            status = getattr(exc, "status_code", None)
            headers = getattr(exc, "resp_headers", {})
            if status is not None:
                self.last_http_status = status
            if headers:
                self.last_response_headers = headers
            raise

        headers = {}
        if isinstance(res, tuple) and len(res) >= 3:
            data, _, headers = res
        else:
            data = res
        self.last_http_status = 200
        self.last_response_headers = headers
        return data

    @staticmethod
    def _format_symbol(symbol: str) -> str:
        """Convert ``BASE/QUOTE:SETTLE`` to ``BASEQUOTE`` for REST calls."""
        symbol = symbol.split(":", 1)[0]
        return symbol.replace("/", "")

    async def load_markets(self) -> Dict[str, Dict]:
        """Return a dictionary of available USDT-margined futures markets."""

        def _sync() -> Dict[str, Dict]:
            res = self._call_client("get_instruments_info", category="linear")
            instruments = res.get("result", {}).get("list", [])
            markets: Dict[str, Dict] = {}
            for info in instruments:
                symbol = info.get("symbol")
                if not symbol:
                    continue
                base = info.get("baseCoin")
                quote = info.get("quoteCoin")
                settle = info.get("settleCoin")
                key = (
                    f"{base}/{quote}:{settle}" if base and quote and settle else symbol
                )
                markets[key] = {
                    "id": symbol,
                    "symbol": key,
                    "base": base,
                    "quote": quote,
                    "settle": settle,
                    "active": str(info.get("status", "")).lower() == "trading",
                }
            return markets

        return await asyncio.to_thread(_sync)

    async def fetch_ticker(self, symbol: str) -> Dict:
        def _sync():
            sym = self._format_symbol(symbol)
            res = self._call_client("get_tickers", category="linear", symbol=sym)
            return res.get("result", {}).get("list", [{}])[0]

        return await asyncio.to_thread(_sync)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 200, since: Optional[int] = None
    ) -> List[List[float]]:
        def _sync():
            params = {
                "category": "linear",
                "symbol": self._format_symbol(symbol),
                "interval": bybit_interval(timeframe),
                "limit": limit,
            }
            if since is not None:
                params["start"] = int(since)
            res = self._call_client("get_kline", **params)
            candles = res.get("result", {}).get("list", [])
            return [
                [
                    int(c[0]),
                    float(c[1]),
                    float(c[2]),
                    float(c[3]),
                    float(c[4]),
                    float(c[5]),
                ]
                for c in candles
            ]

        return await asyncio.to_thread(_sync)

    async def fetch_order_book(self, symbol: str, limit: int = 10) -> Dict:
        def _sync():
            sym = self._format_symbol(symbol)
            res = self._call_client("get_orderbook", category="linear", symbol=sym)
            ob = res.get("result", {})
            return {
                "bids": [[float(p), float(q)] for p, q, *_ in ob.get("b", [])][:limit],
                "asks": [[float(p), float(q)] for p, q, *_ in ob.get("a", [])][:limit],
            }

        return await asyncio.to_thread(_sync)

    async def fetch_funding_rate(self, symbol: str) -> Dict:
        def _sync():
            res = self._call_client(
                "get_funding_rate_history",
                category="linear",
                symbol=self._format_symbol(symbol),
                limit=1,
            )
            items = res.get("result", {}).get("list", [])
            rate = float(items[0]["fundingRate"]) if items else 0.0
            return {"fundingRate": rate}

        return await asyncio.to_thread(_sync)

    async def fetch_open_interest(self, symbol: str) -> Dict:
        def _sync():
            res = self._call_client(
                "get_open_interest",
                category="linear",
                symbol=self._format_symbol(symbol),
                intervalTime="5min",
            )
            items = res.get("result", {}).get("list", [])
            interest = float(items[-1]["openInterest"]) if items else 0.0
            return {"openInterest": interest}

        return await asyncio.to_thread(_sync)

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None,
    ):
        def _sync():
            payload = {
                "category": "linear",
                "symbol": self._format_symbol(symbol),
                "side": side.capitalize(),
                "orderType": order_type.capitalize(),
                "qty": amount,
            }
            if price is not None and order_type == "limit":
                payload["price"] = price
            if params:
                payload.update(params)
            return self._call_client("place_order", **payload)

        return await asyncio.to_thread(_sync)

    async def create_order_with_take_profit_and_stop_loss(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float],
        take_profit: Optional[float],
        stop_loss: Optional[float],
        params: Optional[Dict] = None,
    ):
        def _sync():
            payload = {
                "category": "linear",
                "symbol": self._format_symbol(symbol),
                "side": side.capitalize(),
                "orderType": order_type.capitalize(),
                "qty": amount,
            }
            if price is not None and order_type == "limit":
                payload["price"] = price
            if take_profit is not None:
                payload["takeProfit"] = take_profit
            if stop_loss is not None:
                payload["stopLoss"] = stop_loss
            if params:
                payload.update(params)
            return self._call_client("place_order", **payload)

        return await asyncio.to_thread(_sync)

    async def fetch_balance(self) -> Dict:
        def _sync():
            res = self._call_client("get_wallet_balance", accountType="UNIFIED")
            return res.get("result", {})

        return await asyncio.to_thread(_sync)


class TelegramUpdateListener:
    """Listen for incoming Telegram updates with persistent offset."""

    def __init__(self, bot, offset_file: str = "telegram_offset.txt"):
        self.bot = bot
        self.offset_file = offset_file
        self.offset = self._load_offset()
        self._stop_event = asyncio.Event()

    def _load_offset(self) -> int:
        try:
            with open(self.offset_file, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return 0

    def _save_offset(self) -> None:
        try:
            with open(self.offset_file, "w", encoding="utf-8") as f:
                f.write(str(self.offset))
        except (OSError, ValueError) as exc:
            logger.error("Ошибка сохранения offset Telegram: %s", exc)

    async def listen(self, handler):
        while not self._stop_event.is_set():
            try:
                updates = await self.bot.get_updates(offset=self.offset + 1, timeout=10)
                for upd in updates:
                    self.offset = upd.update_id
                    try:
                        await handler(upd)
                    finally:
                        self._save_offset()
            except asyncio.CancelledError:
                raise
            except (
                RetryAfter,
                BadRequest,
                Forbidden,
                httpx.HTTPError,
                RuntimeError,
            ) as exc:
                logger.error("Ошибка получения обновлений Telegram: %s", exc)
                await asyncio.sleep(5)

    def stop(self) -> None:
        self._stop_event.set()


def check_dataframe_empty(df, context: str = "") -> bool:
    try:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Для проверки DataFrame требуется установленный пакет 'pandas'"
            ) from exc
        if df is None:
            logger.warning("DataFrame является None в контексте: %s", context)
            return True
        if isinstance(df, pd.DataFrame):
            if df.empty:
                logger.warning("DataFrame пуст в контексте: %s", context)
                return True
            if df.isna().all().all():
                logger.warning(
                    "DataFrame содержит только NaN в контексте: %s",
                    context,
                )
                return True
        return False
    except (KeyError, AttributeError, TypeError) as e:
        logger.error(
            "Ошибка проверки DataFrame в контексте %s: %s",
            context,
            e,
        )
        return True


async def check_dataframe_empty_async(df, context: str = "") -> bool:
    """Asynchronously check if a DataFrame is empty.

    This helper allows using the synchronous :func:`check_dataframe_empty`
    when it has been monkeypatched with an async implementation in tests.
    """
    result = check_dataframe_empty(df, context)
    if inspect.isawaitable(result):
        result = await result
    return result


def sanitize_symbol(symbol: str) -> str:
    """Sanitize symbol string for safe filesystem usage."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', symbol)


def sanitize_timeframe(timeframe: str) -> str:
    """Validate that *timeframe* contains only safe characters."""

    value = str(timeframe).strip()
    if not value:
        raise ValueError("Timeframe contains no valid characters")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", value):
        raise ValueError(f"Invalid characters in timeframe: {timeframe!r}")
    return value


_ZSCORE_FN = None


def _import_scipy_stats():
    from importlib import import_module

    return import_module("scipy.stats")


def _resolve_zscore_function(np_module):
    """Return a callable that computes z-scores.

    If SciPy is unavailable (or fails to load due to missing native
    dependencies on the runner) the fallback uses NumPy directly, keeping the
    behaviour identical for the supported use cases in the bot.
    """

    global _ZSCORE_FN

    if _ZSCORE_FN is not None:
        return _ZSCORE_FN

    try:
        scipy_stats = _import_scipy_stats()
    except ImportError:
        def _numpy_zscore(values):
            arr = np_module.asarray(values, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=0))
            if not np_module.isfinite(std) or std == 0.0:
                return np_module.zeros_like(arr, dtype=float)
            return (arr - mean) / std

        _ZSCORE_FN = _numpy_zscore
    else:
        def _scipy_zscore(values):
            result = scipy_stats.zscore(values, ddof=0)
            return np_module.asarray(result, dtype=float)

        _ZSCORE_FN = _scipy_zscore

    return _ZSCORE_FN


def _reset_zscore_cache_for_tests():  # pragma: no cover - testing utility
    global _ZSCORE_FN

    _ZSCORE_FN = None


def filter_outliers_zscore(df, column="close", threshold=3.0):
    try:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Для фильтрации аномалий требуется пакет 'pandas'"
            ) from exc
        try:
            import numpy as np  # noqa: F811 - reused for type narrowing
        except ImportError as exc:
            raise ImportError(
                "Для фильтрации аномалий требуется пакет 'numpy'"
            ) from exc

        series = df[column]
        if len(series.dropna()) < 3:
            logger.warning(
                "Недостаточно данных для z-оценки в %s, возвращается исходный DataFrame",
                column,
            )
            return df

        filled = series.ffill().bfill().fillna(series.mean())
        zscore_fn = _resolve_zscore_function(np)
        z_scores = pd.Series(zscore_fn(filled.to_numpy()), index=df.index)

        mask = (np.abs(z_scores) <= threshold) | series.isna()
        df_filtered = df.copy()
        outliers = ~mask
        if outliers.any():
            logger.info(
                "Заменено %s аномалий в %s с z-оценкой, порог=%.2f",
                int(outliers.sum()),
                column,
                threshold,
            )
            df_filtered.loc[outliers, column] = np.nan
        return df_filtered
    except (KeyError, TypeError) as e:
        logger.error("Ошибка фильтрации аномалий в %s: %s", column, e)
        return df


@jit(nopython=True, parallel=True)
def _calculate_volume_profile(prices, volumes, bins=50):
    if np is None:
        raise RuntimeError("NumPy требуется для расчёта профиля объёма")
    if len(prices) != len(volumes) or len(prices) < 2:
        return np.zeros(bins)
    min_price = np.min(prices)
    max_price = np.max(prices)
    if min_price == max_price:
        return np.zeros(bins)
    bin_edges = np.linspace(min_price, max_price, bins + 1)
    volume_profile = np.zeros(bins)
    for i in prange(len(prices)):
        bin_idx = np.searchsorted(bin_edges, prices[i], side="right") - 1
        if bin_idx == bins:
            bin_idx -= 1
        if 0 <= bin_idx < bins:
            volume_profile[bin_idx] += volumes[i]
    return volume_profile / (np.sum(volume_profile) + 1e-6)


def calculate_volume_profile(prices, volumes, bins=50):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Для расчета профиля объема требуется пакет 'numpy'"
        ) from exc
    globals()["np"] = np
    try:
        return _calculate_volume_profile(prices, volumes, bins)
    except (ValueError, TypeError, ImportError) as exc:
        logger.error("Ошибка вычисления профиля объема: %s", exc)
        return np.zeros(bins)



try:  # pragma: no cover - allow importing without package
    from bot.cache import HistoricalDataCache as _HistoricalDataCache  # noqa: E402
except ImportError as exc:  # pragma: no cover - fallback when package missing
    logger.warning("HistoricalDataCache import failed: %s", exc)
    HistoricalDataCache: Any | None = None
else:
    HistoricalDataCache = cast(Any, _HistoricalDataCache)
