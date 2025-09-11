import gzip
import json
import os
import shutil
import time
from io import BytesIO, StringIO
import logging
from bot.data_handler.utils import ensure_utc

logger = logging.getLogger("TradingBot")


def _sanitize_symbol(symbol: str) -> str:
    """Sanitize symbol string using utility helper.

    The import is performed lazily to avoid circular dependencies with
    ``bot.utils``.  Some callers import this module as ``bot.cache`` while
    others may import it as the top level ``cache`` module.  A purely
    relative import fails in the latter case (``ImportError: attempted
    relative import with no known parent package``).  To support both usage
    styles we try the relative import first and fall back to an absolute
    import if necessary.
    """
    try:  # pragma: no cover - import style depends on caller
        from .utils import sanitize_symbol  # type: ignore  # noqa: WPS433
    except ImportError:  # pragma: no cover
        from utils import sanitize_symbol  # type: ignore  # noqa: WPS433

    return sanitize_symbol(symbol)


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
            # Guard against negative cache sizes in case accounting drifts
            self.current_cache_size_mb = max(
                self.current_cache_size_mb - file_size_mb,
                0,
            )
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
        safe_symbol = _sanitize_symbol(symbol)
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
            try:
                data.to_parquet(buffer, compression="gzip")
            except Exception as exc:  # pragma: no cover - pyarrow missing
                # Fallback to JSON so the cache works without optional deps
                logger.warning(
                    "Parquet support unavailable, falling back to JSON: %s",
                    exc,
                )
                buffer = StringIO()
                data.to_json(buffer, orient="split", date_format="iso")
            parquet_bytes = buffer.getvalue()
            if isinstance(parquet_bytes, str):
                parquet_bytes = parquet_bytes.encode("utf-8")
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
            safe_symbol = _sanitize_symbol(symbol)
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
                try:
                    data = pd.read_parquet(filename)
                    fmt = "parquet"
                except Exception as exc:
                    logger.warning(
                        "Parquet read failed, attempting JSON: %s",
                        exc,
                    )
                    with open(filename, "r", encoding="utf-8") as f:
                        data = pd.read_json(f, orient="split")
                    fmt = "json"
                elapsed_time = time.time() - start_time
                if elapsed_time > 0.5:
                    logger.warning(
                        "Высокая задержка чтения %s для %s_%s: %.2f сек",
                        fmt,
                        symbol,
                        timeframe,
                        elapsed_time,
                    )
                logger.info("Данные загружены из кэша (%s): %s", fmt, filename)
                if isinstance(data, pd.DataFrame):
                    if "timestamp" in data.columns:
                        data["timestamp"] = ensure_utc(data["timestamp"])
                    elif isinstance(data.index, pd.DatetimeIndex):
                        data.index = ensure_utc(data.index)
                return data
            if os.path.exists(legacy_json):
                logger.info(
                    "Обнаружен старый кэш для %s_%s, конвертация в Parquet",
                    symbol,
                    timeframe,
                )
                with gzip.open(legacy_json, "rt", encoding="utf-8") as f:
                    payload = json.load(f)
                data_json = payload.get("data")
                data = pd.read_json(StringIO(data_json), orient="split")
                if isinstance(data, pd.DataFrame):
                    if "timestamp" in data.columns:
                        data["timestamp"] = ensure_utc(data["timestamp"])
                    elif isinstance(data.index, pd.DatetimeIndex):
                        data.index = ensure_utc(data.index)
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
