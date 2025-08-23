"""Miscellaneous helper utilities for the trading bot."""

import logging
logger = logging.getLogger("TradingBot")
import os
import json
import re
import asyncio
import time
import inspect
import threading
import warnings
from functools import wraps
from typing import Dict, List, Optional
import gzip
import shutil
from io import StringIO, BytesIO


def configure_logging() -> None:
    """Настроить переменные окружения, связанные с логированием."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# Hide Numba performance warnings
try:
    from numba import jit, prange, NumbaPerformanceWarning  # type: ignore

    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except ImportError as exc:  # pragma: no cover - allow missing numba package
    logging.getLogger("TradingBot").error("Numba import failed: %s", exc)
    _numba_exc = exc

    def _numba_missing(*args, **kwargs):
        raise ImportError("numba is required for JIT-accelerated functions") from _numba_exc

    def jit(*a, **k):
        def wrapper(_f):
            @wraps(_f)
            def inner(*args, **kwargs):
                return _numba_missing(*args, **kwargs)

            return inner

        return wrapper

    def prange(*args):  # type: ignore
        raise ImportError("numba is required for prange") from _numba_exc

try:
    import numpy as np
except ImportError:
    np = None

import httpx

if os.getenv("TEST_MODE") == "1":
    import types
    import sys

    pybit_mod = types.ModuleType("pybit")
    ut_mod = types.ModuleType("unified_trading")
    ut_mod.HTTP = object
    pybit_mod.unified_trading = ut_mod
    sys.modules.setdefault("pybit", pybit_mod)
    sys.modules.setdefault("pybit.unified_trading", ut_mod)
try:
    from telegram.error import RetryAfter, BadRequest, Forbidden
except Exception as exc:  # pragma: no cover - allow missing telegram package
    logging.getLogger("TradingBot").error("Telegram package not available: %s", exc)

    class _TelegramError(Exception):
        pass

    RetryAfter = BadRequest = Forbidden = _TelegramError


from pybit.unified_trading import HTTP

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

    try:  # Lazy import to avoid heavy initialization when unused
        import torch  # type: ignore

        if not torch.backends.cuda.is_built():
            return False
        return torch.cuda.is_available()
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


async def safe_api_call(exchange, method: str, *args, **kwargs):
    """Call a ccxt method with retry, status and retCode verification."""
    if os.getenv("TEST_MODE"):
        # During unit tests we do not want to spend time in the retry loop.
        return await getattr(exchange, method)(*args, **kwargs)

    delay = 1.0
    for attempt in range(5):
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
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.error("Bybit API error in %s: %s", method, exc)
            if "10002" in str(exc):
                logger.error(
                    "Request not authorized. Check server time sync and recv_window"
                )
            if attempt == 4:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10)


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
        api_key = api_secret = None

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


level_name = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, level_name, logging.INFO))

log_dir = os.getenv("LOG_DIR", "/app/logs")
fallback_dir = os.path.join(os.path.dirname(__file__), "logs")
try:
    os.makedirs(log_dir, exist_ok=True)
    if not os.access(log_dir, os.W_OK):
        raise PermissionError
except (OSError, PermissionError):
    log_dir = fallback_dir
    os.makedirs(log_dir, exist_ok=True)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(os.path.join(log_dir, "trading_bot.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(
        "Logging initialized at %s; writing logs to %s",
        logging.getLevelName(logger.level),
        log_dir,
    )


class TelegramLogger(logging.Handler):
    """Logging handler that forwards records to a Telegram chat."""

    _queue: asyncio.Queue | None = None
    _worker_task: asyncio.Task | None = None
    _worker_lock = asyncio.Lock()
    _bot = None
    _stop_event: asyncio.Event | None = None

    def __init__(
        self, bot, chat_id, level=logging.NOTSET, max_queue_size: int | None = None
    ):
        super().__init__(level)
        self.bot = bot
        self.chat_id = chat_id
        self.last_message_time = 0
        self.message_interval = 1800
        self.message_lock = asyncio.Lock()

        if TelegramLogger._queue is None:
            TelegramLogger._queue = asyncio.Queue(maxsize=max_queue_size or 0)
            TelegramLogger._bot = bot
            TelegramLogger._stop_event = asyncio.Event()
            if os.getenv("TEST_MODE") != "1":
                try:
                    loop = asyncio.get_running_loop()
                    TelegramLogger._worker_task = loop.create_task(self._worker())
                except RuntimeError:
                    threading.Thread(
                        target=lambda: asyncio.run(self._worker()),
                        daemon=True,
                    ).start()

        self.last_sent_text = ""

    async def _worker(self):
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
            await self._send(text, chat_id, urgent)
            queue.task_done()
            await asyncio.sleep(1)

    async def _send(self, message: str, chat_id: int | str, urgent: bool):
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
                        result = await TelegramLogger._bot.send_message(
                            chat_id=chat_id, text=part
                        )
                        if not getattr(result, "message_id", None):
                            logger.error("Telegram message response without message_id")
                        else:
                            self.last_sent_text = part
                        self.last_message_time = time.time()
                        break
                    except RetryAfter as e:
                        wait_time = getattr(e, "retry_after", delay)
                        logger.warning(
                            "Flood control: ожидание %sс",
                            wait_time,
                        )
                        await asyncio.sleep(wait_time)
                        delay = min(delay * 2, 60)
                    except httpx.ConnectError as e:
                        logger.warning(
                            "Ошибка соединения Telegram: %s. Попытка %s/5",
                            e,
                            attempt + 1,
                        )
                        if attempt < 4:
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, 60)
                    except BadRequest as e:
                        logger.error("BadRequest Telegram: %s. Проверьте chat_id", e)
                        break
                    except Forbidden as e:
                        logger.error(
                            "Forbidden Telegram: %s. Проверьте токен и chat_id",
                            e,
                        )
                        break
                    except httpx.HTTPError as e:
                        logger.error("HTTP ошибка Telegram: %s", e)
                        break
                    except asyncio.CancelledError:
                        raise
                    except (ValueError, RuntimeError) as e:
                        logger.exception("Ошибка отправки сообщения Telegram: %s", e)
                        return

    async def send_telegram_message(self, message, urgent: bool = False):
        # Telegram allows up to 4096 characters per message. The worker will
        # further split messages into 500 character chunks, so only trim to the
        # API limit here instead of the previous 512 characters.
        if TelegramLogger._queue is None:
            logger.warning("TelegramLogger queue is None, message dropped")
            return
        msg = message[:4096]
        try:
            TelegramLogger._queue.put_nowait((self.chat_id, msg, urgent))
        except asyncio.QueueFull:
            logger.warning("Очередь Telegram переполнена, сообщение пропущено")

    def emit(self, record):
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
        except (ValueError, RuntimeError) as e:
            logger.error("Ошибка в TelegramLogger: %s", e)

    @classmethod
    async def shutdown(cls):
        if cls._stop_event is None:
            return
        if cls._queue is not None:
            try:
                cls._queue.put_nowait((None, "", False))
            except asyncio.QueueFull:
                pass
        cls._stop_event.set()
        if cls._worker_task is not None:
            cls._worker_task.cancel()
            try:
                await cls._worker_task
            except asyncio.CancelledError:
                pass
            cls._worker_task = None
        cls._queue = None
        cls._stop_event = None


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


def filter_outliers_zscore(df, column="close", threshold=3.0):
    try:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Для фильтрации аномалий требуется пакет 'pandas'"
            ) from exc
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "Для фильтрации аномалий требуется пакет 'numpy'"
            ) from exc
        try:
            from scipy.stats import zscore
        except ImportError as exc:
            raise ImportError(
                "Для фильтрации аномалий требуется пакет 'scipy'"
            ) from exc

        series = df[column]
        if len(series.dropna()) < 3:
            logger.warning(
                "Недостаточно данных для z-оценки в %s, возвращается исходный DataFrame",
                column,
            )
            return df

        filled = series.ffill().bfill().fillna(series.mean())
        z_scores = pd.Series(zscore(filled.to_numpy()), index=df.index)

        mask = (np.abs(z_scores) <= threshold) | series.isna()
        df_filtered = df[mask]
        if len(df_filtered) < len(df):
            logger.info(
                "Удалено %s аномалий в %s с z-оценкой, порог=%.2f",
                len(df) - len(df_filtered),
                column,
                threshold,
            )
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


class HistoricalDataCache:
    """Manage on-disk storage for historical OHLCV data."""

    def __init__(self, cache_dir="/app/cache", min_free_disk_gb=0.1):
        self.cache_dir = cache_dir
        self.min_free_disk_gb = min_free_disk_gb
        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.access(self.cache_dir, os.W_OK):
            raise PermissionError(
                f"Нет прав на запись в директорию кэша: {self.cache_dir}"
            )
        # Allow a larger on-disk cache by default to reduce re-fetches
        self.max_cache_size_mb = 2048
        self.cache_ttl = 86400 * 7
        self.current_cache_size_mb = self._calculate_cache_size()
        # Permit using slightly more memory before triggering cleanup
        self.memory_threshold = 0.9
        # Allow disk buffer to grow in proportion to the larger cache size
        self.max_buffer_size_mb = 2048

    def _calculate_cache_size(self):
        total_size = 0
        for dirpath, _, filenames in os.walk(self.cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except FileNotFoundError:
                    # File was removed after os.walk listed it
                    continue
        return total_size / (1024 * 1024)

    def _check_disk_space(self):
        disk_usage = shutil.disk_usage(self.cache_dir)
        if disk_usage.free / (1024**3) < self.min_free_disk_gb:
            logger.warning(
                "Недостаточно свободного места на диске: %.2f ГБ",
                disk_usage.free / (1024**3),
            )
            self._aggressive_clean()
            return False
        return True

    def _check_memory(self, additional_size_mb):
        try:
            import psutil
        except ImportError as exc:
            raise ImportError(
                "Для проверки памяти требуется пакет 'psutil'"
            ) from exc
        memory = psutil.virtual_memory()
        available = getattr(memory, "available", None)
        used_percent = getattr(memory, "percent", 0)
        if used_percent > self.memory_threshold * 100:
            logger.warning("Высокая загрузка памяти: %.1f%%", used_percent)
            self._aggressive_clean()
        if available is None:
            return True
        available_mb = available / (1024 * 1024)
        return (
            self.current_cache_size_mb + additional_size_mb
        ) < available_mb * self.memory_threshold

    def _delete_cache_file(self, path):
        """Remove a cache file and adjust the cached size value."""
        if not os.path.exists(path):
            return
        try:
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
        except OSError:
            file_size_mb = 0
        try:
            os.remove(path)
            self.current_cache_size_mb -= file_size_mb
        except OSError as e:  # pragma: no cover - unexpected deletion failure
            logger.error("Ошибка удаления файла кэша %s: %s", path, e)

    def _aggressive_clean(self):
        try:
            files = [
                (f, os.path.getmtime(os.path.join(self.cache_dir, f)))
                for f in os.listdir(self.cache_dir)
                if os.path.isfile(os.path.join(self.cache_dir, f))
            ]
            if not files:
                return
            files.sort(key=lambda x: x[1])
            target_size = self.max_cache_size_mb * 0.5
            while self.current_cache_size_mb > target_size and files:
                file_name, _ = files.pop(0)
                file_path = os.path.join(self.cache_dir, file_name)
                self._delete_cache_file(file_path)
                logger.info(
                    "Удален файл кэша (агрессивная очистка): %s",
                    file_name,
                )
        except OSError as e:
            logger.error("Ошибка агрессивной очистки кэша: %s", e)

    def _check_buffer_size(self):
        buffer_size_mb = self.current_cache_size_mb
        if buffer_size_mb > self.max_buffer_size_mb:
            logger.warning(
                "Дисковый буфер превысил лимит %s МБ, очистка",
                self.max_buffer_size_mb,
            )
            self._aggressive_clean()

    def save_cached_data(self, symbol, timeframe, data):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Для кэширования данных требуется пакет 'pandas'"
            ) from exc
        safe_symbol = sanitize_symbol(symbol)
        if isinstance(data, pd.DataFrame) and data.empty:
            return
        if not self._check_disk_space():
            logger.error(
                "Невозможно кэшировать %s_%s: нехватка места на диске",
                symbol,
                timeframe,
            )
            return
        filename = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.parquet")
        start_time = time.time()
        try:
            buffer = BytesIO()
            data.to_parquet(buffer, compression="gzip")
            parquet_bytes = buffer.getvalue()
            file_size_mb = len(parquet_bytes) / (1024 * 1024)
            if not self._check_memory(file_size_mb):
                logger.warning(
                    "Недостаточно памяти для кэширования %s_%s, очистка кэша",
                    symbol,
                    timeframe,
                )
                self._aggressive_clean()
                if not self._check_memory(file_size_mb):
                    logger.error(
                        "Невозможно кэшировать %s_%s: нехватка памяти",
                        symbol,
                        timeframe,
                    )
                    return
            self._check_buffer_size()
            old_size_mb = 0
            if os.path.exists(filename):
                try:
                    old_size_mb = os.path.getsize(filename) / (1024 * 1024)
                except OSError:
                    old_size_mb = 0
            with open(filename, "wb") as f:
                f.write(parquet_bytes)
            compressed_size_mb = os.path.getsize(filename) / (1024 * 1024)
            self.current_cache_size_mb += compressed_size_mb - old_size_mb
            elapsed_time = time.time() - start_time
            if elapsed_time > 0.5:
                logger.warning(
                    "Высокая задержка записи Parquet для %s_%s: %.2f сек",
                    symbol,
                    timeframe,
                    elapsed_time,
                )
            logger.info(
                "Данные кэшированы (parquet): %s, размер %.2f МБ",
                filename,
                compressed_size_mb,
            )
            self._aggressive_clean()
        except (OSError, ValueError, TypeError) as e:
            logger.error("Ошибка сохранения кэша для %s_%s: %s", symbol, timeframe, e)

    def load_cached_data(self, symbol, timeframe):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Для загрузки кэша требуется пакет 'pandas'"
            ) from exc
        try:
            safe_symbol = sanitize_symbol(symbol)
            filename = os.path.join(
                self.cache_dir, f"{safe_symbol}_{timeframe}.parquet"
            )
            legacy_json = os.path.join(
                self.cache_dir, f"{safe_symbol}_{timeframe}.json.gz"
            )
            old_gzip = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.pkl.gz")
            old_filename = os.path.join(
                self.cache_dir, f"{safe_symbol}_{timeframe}.pkl"
            )
            if os.path.exists(filename):
                if time.time() - os.path.getmtime(filename) > self.cache_ttl:
                    logger.info("Кэш для %s_%s устарел, удаление", symbol, timeframe)
                    self._delete_cache_file(filename)
                    return None
                start_time = time.time()
                data = pd.read_parquet(filename)
                elapsed_time = time.time() - start_time
                if elapsed_time > 0.5:
                    logger.warning(
                        "Высокая задержка чтения Parquet для %s_%s: %.2f сек",
                        symbol,
                        timeframe,
                        elapsed_time,
                    )
                logger.info("Данные загружены из кэша (parquet): %s", filename)
                return data
            if os.path.exists(legacy_json):
                logger.info(
                    "Обнаружен старый кэш для %s_%s, конвертация в Parquet",
                    symbol,
                    timeframe,
                )
                with gzip.open(legacy_json, "rt") as f:
                    payload = json.load(f)
                data_json = payload.get("data")
                data = pd.read_json(StringIO(data_json), orient="split")
                if not isinstance(data, pd.DataFrame):
                    logger.error(
                        "Неверный тип данных в старом кэше %s_%s: %s",
                        symbol,
                        timeframe,
                        type(data),
                    )
                    self._delete_cache_file(legacy_json)
                    return None
                self.save_cached_data(symbol, timeframe, data)
                self._delete_cache_file(legacy_json)
                return data
            found_pickle = False
            for legacy in (old_gzip, old_filename):
                if os.path.exists(legacy):
                    logger.warning(
                        "Обнаружен небезопасный pickle кэш для %s_%s, файл удалён: %s",
                        symbol,
                        timeframe,
                        legacy,
                    )
                    self._delete_cache_file(legacy)
                    found_pickle = True
            if found_pickle:
                return None
            return None
        except (OSError, ValueError) as e:
            logger.error("Ошибка загрузки кэша для %s_%s: %s", symbol, timeframe, e)
            for f in (filename, legacy_json, old_gzip, old_filename):
                self._delete_cache_file(f)
            return None
