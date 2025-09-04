"""Miscellaneous helper utilities for the trading bot."""

import logging
logger = logging.getLogger("TradingBot")
import os
import re
import ipaddress
import asyncio
import time
import inspect
import threading
import warnings
from functools import wraps
from typing import Dict, List, Optional
import shutil


def suppress_tf_logs() -> None:
    """Установить ``TF_CPP_MIN_LOG_LEVEL=3`` для подавления логов TensorFlow."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def configure_logging() -> None:
    """Настроить переменные окружения и handlers для логирования."""

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    try:
        logger.setLevel(level_name)
    except ValueError:
        logger.warning("LOG_LEVEL '%s' недопустим, используется INFO", level_name)
        logger.setLevel(logging.INFO)

    log_dir = os.getenv("LOG_DIR", "/app/logs")
    fallback_dir = os.path.join(os.path.dirname(__file__), "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        if not os.access(log_dir, os.W_OK):
            raise PermissionError
    except (OSError, PermissionError):
        log_dir = fallback_dir
        os.makedirs(log_dir, exist_ok=True)

    if logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "trading_bot.log"), encoding="utf-8"
    )
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

try:  # pragma: no cover - allow running outside package
    from .telegram_logger import TelegramLogger
except ImportError as exc:  # pragma: no cover - fallback when executed directly
    logger.warning("Failed to import TelegramLogger relatively: %s", exc)
    from telegram_logger import TelegramLogger  # type: ignore

try:
    from telegram.error import RetryAfter, BadRequest, Forbidden
except ImportError as exc:  # pragma: no cover - allow missing telegram package
    logging.getLogger("TradingBot").error("Telegram package not available: %s", exc)

    class _TelegramError(Exception):
        pass

    RetryAfter = BadRequest = Forbidden = _TelegramError


try:
    from pybit.unified_trading import HTTP
except ImportError:  # pragma: no cover - pybit is optional in CI
    class HTTP:  # type: ignore
        """Fallback HTTP stub when pybit is unavailable."""

        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
            raise ImportError("pybit is required for HTTP operations")

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


def validate_host() -> str:
    """Проверить допустимость значения переменной окружения ``HOST``."""
    host = os.getenv("HOST")
    if not host:
        logger.info("HOST не установлен, используется 127.0.0.1")
        return "127.0.0.1"

    if host == "localhost":
        logger.info("HOST 'localhost' интерпретирован как 127.0.0.1")
        return "127.0.0.1"

    try:
        ip = ipaddress.ip_address(host)
        if ip.is_unspecified:
            raise ValueError(f"HOST '{ip}' запрещен")
    except ValueError:
        if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
            raise ValueError(f"Некорректный IP: {host}")
        logger.warning("HOST '%s' не локальный хост", host)
        raise ValueError(f"HOST '{host}' не локальный хост")

    if host != "127.0.0.1":
        logger.warning("HOST '%s' не локальный хост", host)
        raise ValueError(f"HOST '{host}' не локальный хост")
    return host


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
                    env_var,
                    value,
                    default,
                )
            return default
        return result
    except (TypeError, ValueError):
        if env_var:
            logger.warning(
                "Invalid %s value '%s', using default %s",
                env_var,
                value,
                default,
            )
        return default


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



try:  # pragma: no cover - allow importing without package
    from bot.cache import HistoricalDataCache  # noqa: E402
except ImportError as exc:  # pragma: no cover - fallback when package missing
    logger.warning("HistoricalDataCache import failed: %s", exc)
    HistoricalDataCache = None  # type: ignore
