import logging
import os
import pickle
import pandas as pd
import numpy as np
import asyncio
import time
import inspect
from typing import Dict, List, Optional
from scipy.stats import zscore
import gzip
import psutil
import shutil
try:
    from numba import jit, prange  # type: ignore
except Exception:  # pragma: no cover - allow missing numba package
    def jit(*a, **k):
        def wrapper(f):
            return f
        return wrapper

    def prange(*args):  # type: ignore
        return range(*args)

import httpx
try:
    from telegram.error import RetryAfter, BadRequest, Forbidden
except Exception:  # pragma: no cover - allow missing telegram package
    class RetryAfter(Exception):
        pass

    class BadRequest(Exception):
        pass

    class Forbidden(Exception):
        pass
from pybit.unified_trading import HTTP


async def handle_rate_limits(exchange) -> None:
    """Sleep if Bybit rate limit is close to exhaustion."""
    headers = getattr(exchange, "last_response_headers", {}) or {}
    try:
        remaining = int(headers.get("X-Bapi-Limit-Status", headers.get("x-bapi-limit-status", 0)))
        reset_ts = int(headers.get("X-Bapi-Limit-Reset-Timestamp", headers.get("x-bapi-limit-reset-timestamp", 0)))
    except ValueError:
        return
    if remaining and remaining <= 5:
        wait_time = max(0.0, reset_ts / 1000 - time.time())
        if wait_time > 0:
            logger.info(f"Rate limit low ({remaining}), sleeping {wait_time:.2f}s")
            await asyncio.sleep(wait_time)


async def safe_api_call(exchange, method: str, *args, **kwargs):
    """Call a ccxt method with retry, status and retCode verification."""
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
        except Exception as exc:
            logger.error(f"Bybit API error in {method}: {exc}")
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
        self.client = HTTP(api_key=api_key, api_secret=api_secret)
        self.last_http_status = 200
        self.last_response_headers = {}

    async def fetch_ticker(self, symbol: str) -> Dict:
        def _sync():
            res = self.client.get_tickers(category="linear", symbol=symbol.replace(":USDT", "USDT"))
            return res.get("result", {}).get("list", [{}])[0]

        return await asyncio.to_thread(_sync)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 200, since: Optional[int] = None
    ) -> List[List[float]]:
        def _sync():
            params = {
                "category": "linear",
                "symbol": symbol.replace(":USDT", "USDT"),
                "interval": timeframe,
                "limit": limit,
            }
            if since is not None:
                params["start"] = int(since)
            res = self.client.get_kline(**params)
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
            res = self.client.get_orderbook(category="linear", symbol=symbol.replace(":USDT", "USDT"))
            ob = res.get("result", {})
            return {
                "bids": [[float(p), float(q)] for p, q, *_ in ob.get("b", [])][:limit],
                "asks": [[float(p), float(q)] for p, q, *_ in ob.get("a", [])][:limit],
            }

        return await asyncio.to_thread(_sync)

    async def fetch_funding_rate(self, symbol: str) -> Dict:
        def _sync():
            res = self.client.get_funding_rate_history(
                category="linear",
                symbol=symbol.replace(":USDT", "USDT"),
                limit=1,
            )
            items = res.get("result", {}).get("list", [])
            rate = float(items[0]["fundingRate"]) if items else 0.0
            return {"fundingRate": rate}

        return await asyncio.to_thread(_sync)

    async def fetch_open_interest(self, symbol: str) -> Dict:
        def _sync():
            res = self.client.get_open_interest(
                category="linear",
                symbol=symbol.replace(":USDT", "USDT"),
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
                "symbol": symbol.replace(":USDT", "USDT"),
                "side": side.capitalize(),
                "orderType": order_type.capitalize(),
                "qty": amount,
            }
            if price is not None and order_type == "limit":
                payload["price"] = price
            if params:
                payload.update(params)
            return self.client.place_order(**payload)

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
                "symbol": symbol.replace(":USDT", "USDT"),
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
            return self.client.place_order(**payload)

        return await asyncio.to_thread(_sync)

    async def fetch_balance(self) -> Dict:
        def _sync():
            res = self.client.get_wallet_balance(accountType="UNIFIED")
            return res.get("result", {})

        return await asyncio.to_thread(_sync)

logger = logging.getLogger("TradingBot")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

log_dir = os.getenv("LOG_DIR", "/app/logs")
os.makedirs(log_dir, exist_ok=True)
if not os.access(log_dir, os.W_OK):
    raise PermissionError(f"Нет прав на запись в директорию логов: {log_dir}")

file_handler = logging.FileHandler(os.path.join(log_dir, "trading_bot.log"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class TelegramLogger(logging.Handler):
    _queue: asyncio.Queue | None = None
    _worker_task: asyncio.Task | None = None
    _worker_lock = asyncio.Lock()
    _bot = None
    _stop_event: asyncio.Event | None = None

    def __init__(self, bot, chat_id, level=logging.NOTSET, max_queue_size: int | None = None):
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
            TelegramLogger._worker_task = asyncio.create_task(self._worker())

        self.last_sent_text = ""

    async def _worker(self):
        assert TelegramLogger._queue is not None
        assert TelegramLogger._stop_event is not None
        while not TelegramLogger._stop_event.is_set():
            try:
                item = await asyncio.wait_for(TelegramLogger._queue.get(), 1.0)
            except asyncio.TimeoutError:
                continue
            chat_id, text, urgent = item
            await self._send(text, chat_id, urgent)
            TelegramLogger._queue.task_done()
            await asyncio.sleep(1)

    async def _send(self, message: str, chat_id: int | str, urgent: bool):
        async with self.message_lock:
            if (
                not urgent
                and time.time() - self.last_message_time < self.message_interval
            ):
                logger.debug(
                    f"Сообщение Telegram пропущено из-за интервала: {message[:100]}..."
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
                        logger.warning(f"Flood control: ожидание {wait_time}с")
                        await asyncio.sleep(wait_time)
                        delay = min(delay * 2, 60)
                    except httpx.ConnectError as e:
                        logger.warning(
                            f"Ошибка соединения Telegram: {e}. Попытка {attempt + 1}/5"
                        )
                        if attempt < 4:
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, 60)
                    except BadRequest as e:
                        logger.error(
                            f"BadRequest Telegram: {e}. Проверьте chat_id"
                        )
                        break
                    except Forbidden as e:
                        logger.error(
                            f"Forbidden Telegram: {e}. Проверьте токен и chat_id"
                        )
                        break
                    except httpx.HTTPError as e:
                        logger.error(f"HTTP ошибка Telegram: {e}")
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.exception(f"Ошибка отправки сообщения Telegram: {e}")
                        return

    async def send_telegram_message(self, message, urgent: bool = False):
        msg = message[:512]
        try:
            TelegramLogger._queue.put_nowait((self.chat_id, msg, urgent))
        except asyncio.QueueFull:
            logger.warning("Очередь Telegram переполнена, сообщение пропущено")

    def emit(self, record):
        try:
            msg = self.format(record)
            asyncio.create_task(self.send_telegram_message(msg))
        except Exception as e:
            logger.error(f"Ошибка в TelegramLogger: {e}")

    @classmethod
    async def shutdown(cls):
        if cls._stop_event is None:
            return
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
        except Exception:
            return 0

    def _save_offset(self) -> None:
        try:
            with open(self.offset_file, "w", encoding="utf-8") as f:
                f.write(str(self.offset))
        except Exception as exc:
            logger.error(f"Ошибка сохранения offset Telegram: {exc}")

    async def listen(self, handler):
        while not self._stop_event.is_set():
            try:
                updates = await self.bot.get_updates(
                    offset=self.offset + 1, timeout=10
                )
                for upd in updates:
                    self.offset = upd.update_id
                    try:
                        await handler(upd)
                    finally:
                        self._save_offset()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"Ошибка получения обновлений Telegram: {exc}")
                await asyncio.sleep(5)

    def stop(self) -> None:
        self._stop_event.set()


def check_dataframe_empty(df, context: str = "") -> bool:
    try:
        if df is None:
            logger.warning(f"DataFrame является None в контексте: {context}")
            return True
        if isinstance(df, pd.DataFrame):
            if df.empty:
                logger.warning(f"DataFrame пуст в контексте: {context}")
                return True
            if df.isna().all().all():
                logger.warning(f"DataFrame содержит только NaN в контексте: {context}")
                return True
        return False
    except (KeyError, AttributeError, TypeError) as e:
        logger.error(f"Ошибка проверки DataFrame в контексте {context}: {e}")
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
    return symbol.replace("/", "_").replace(":", "_")


def filter_outliers_zscore(df, column="close", threshold=3.0):
    try:
        if len(df[column].dropna()) < 3:
            logger.warning(
                f"Недостаточно данных для z-оценки в {column}, возвращается исходный DataFrame"
            )
            return df
        z_scores = zscore(df[column].dropna())
        volatility = df[column].pct_change().std()
        adjusted_threshold = threshold * (1 + volatility / 0.02)
        df_filtered = df[np.abs(z_scores) <= adjusted_threshold]
        if len(df_filtered) < len(df):
            logger.info(
                f"Удалено {len(df) - len(df_filtered)} аномалий в {column} с z-оценкой, порог={adjusted_threshold:.2f}"
            )
        return df_filtered
    except (KeyError, TypeError) as e:
        logger.error(f"Ошибка фильтрации аномалий в {column}: {e}")
        return df


@jit(nopython=True, parallel=True)
def _calculate_volume_profile(prices, volumes, bins=50):
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
        if 0 <= bin_idx < bins:
            volume_profile[bin_idx] += volumes[i]
    return volume_profile / (np.sum(volume_profile) + 1e-6)


def calculate_volume_profile(prices, volumes, bins=50):
    try:
        return _calculate_volume_profile(prices, volumes, bins)
    except (ValueError, TypeError) as exc:
        logger.error(f"Ошибка вычисления профиля объема: {exc}")
        return np.zeros(bins)


class HistoricalDataCache:
    def __init__(self, cache_dir="/app/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.access(self.cache_dir, os.W_OK):
            raise PermissionError(
                f"Нет прав на запись в директорию кэша: {self.cache_dir}"
            )
        self.max_cache_size_mb = 512
        self.cache_ttl = 86400 * 7
        self.current_cache_size_mb = self._calculate_cache_size()
        self.memory_threshold = 0.8
        self.max_buffer_size_mb = 512

    def _calculate_cache_size(self):
        total_size = 0
        for dirpath, _, filenames in os.walk(self.cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)

    def _check_disk_space(self):
        disk_usage = shutil.disk_usage(self.cache_dir)
        if disk_usage.free / (1024**3) < 0.5:
            logger.warning(
                f"Недостаточно свободного места на диске: {disk_usage.free / (1024 ** 3):.2f} ГБ"
            )
            self._aggressive_clean()
            return False
        return True

    def _check_memory(self, additional_size_mb):
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        used_percent = memory.percent
        if used_percent > self.memory_threshold * 100:
            logger.warning(f"Высокая загрузка памяти: {used_percent:.1f}%")
            self._aggressive_clean()
        return (
            self.current_cache_size_mb + additional_size_mb
        ) < available_mb * self.memory_threshold

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
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                os.remove(file_path)
                self.current_cache_size_mb -= file_size_mb
                logger.info(
                    f"Удален файл кэша (агрессивная очистка): {file_name}, освобождено {file_size_mb:.2f} МБ"
                )
        except OSError as e:
            logger.error(f"Ошибка агрессивной очистки кэша: {e}")

    def _check_buffer_size(self):
        buffer_size_mb = self._calculate_cache_size()
        if buffer_size_mb > self.max_buffer_size_mb:
            logger.warning(
                f"Дисковый буфер превысил лимит {self.max_buffer_size_mb} МБ, очистка"
            )
            self._aggressive_clean()

    def save_cached_data(self, symbol, timeframe, data):
        try:
            safe_symbol = sanitize_symbol(symbol)
            if not self._check_disk_space():
                logger.error(
                    f"Невозможно кэшировать {symbol}_{timeframe}: нехватка места на диске"
                )
                return
            filename = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.pkl.gz")
            temp_filename = os.path.join(
                self.cache_dir, f"temp_{safe_symbol}_{timeframe}.pkl"
            )
            start_time = time.time()
            with open(temp_filename, "wb") as f:
                pickle.dump(data, f)
            file_size_mb = os.path.getsize(temp_filename) / (1024 * 1024)
            os.remove(temp_filename)
            if not self._check_memory(file_size_mb):
                logger.warning(
                    f"Недостаточно памяти для кэширования {symbol}_{timeframe}, очистка кэша"
                )
                self._aggressive_clean()
                if not self._check_memory(file_size_mb):
                    logger.error(
                        f"Невозможно кэшировать {symbol}_{timeframe}: нехватка памяти"
                    )
                    return
            self._check_buffer_size()
            with gzip.open(filename, "wb") as f:
                pickle.dump(data, f)
            compressed_size_mb = os.path.getsize(filename) / (1024 * 1024)
            self.current_cache_size_mb += compressed_size_mb
            elapsed_time = time.time() - start_time
            if elapsed_time > 0.5:
                logger.warning(
                    f"Высокая задержка сжатия gzip для {symbol}_{timeframe}: {elapsed_time:.2f} сек"
                )
            logger.info(
                f"Данные кэшированы (gzip): {filename}, размер {compressed_size_mb:.2f} МБ"
            )
            self._aggressive_clean()
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша для {symbol}_{timeframe}: {e}")

    def load_cached_data(self, symbol, timeframe):
        try:
            safe_symbol = sanitize_symbol(symbol)
            filename = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.pkl.gz")
            if os.path.exists(filename):
                if time.time() - os.path.getmtime(filename) > self.cache_ttl:
                    logger.info(f"Кэш для {symbol}_{timeframe} устарел, удаление")
                    os.remove(filename)
                    return None
                start_time = time.time()
                with gzip.open(filename, "rb") as f:
                    data = pickle.load(f)
                elapsed_time = time.time() - start_time
                if elapsed_time > 0.5:
                    logger.warning(
                        f"Высокая задержка чтения gzip для {symbol}_{timeframe}: {elapsed_time:.2f} сек"
                    )
                logger.info(f"Данные загружены из кэша (gzip): {filename}")
                return data
            old_filename = os.path.join(
                self.cache_dir, f"{safe_symbol}_{timeframe}.pkl"
            )
            if os.path.exists(old_filename):
                logger.info(
                    f"Обнаружен старый кэш для {symbol}_{timeframe}, конвертация в gzip"
                )
                with open(old_filename, "rb") as f:
                    data = pickle.load(f)
                self.save_cached_data(symbol, timeframe, data)
                os.remove(old_filename)
                return data
            return None
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша для {symbol}_{timeframe}: {e}")
            return None
