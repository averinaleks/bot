import errno
import gzip
import importlib
import json
import os
import re
import shutil
import time
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path
from types import ModuleType
import logging

logger = logging.getLogger("TradingBot")


@lru_cache(maxsize=1)
def _utils_module() -> ModuleType:
    """Return the project utilities module regardless of import style."""

    try:
        return importlib.import_module("bot.utils")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        if exc.name not in {"bot", "bot.utils"}:
            raise
    return importlib.import_module("utils")


def _sanitize_symbol(symbol: str) -> str:
    """Sanitize symbol string using utility helper."""

    utils = _utils_module()
    sanitize_symbol = getattr(utils, "sanitize_symbol")
    cleaned = sanitize_symbol(symbol)
    # Reject symbols that would produce ambiguous or hidden filenames.  After
    # sanitisation we expect at least one visible character that is neither a
    # leading dot nor part of a ".." sequence which might hint at path
    # traversal attempts.
    if not cleaned:
        raise ValueError("Symbol contains no valid characters")
    if cleaned.startswith(".") or ".." in cleaned:
        raise ValueError("Symbol sanitization resulted in unsafe value")
    return cleaned


_SAFE_CACHE_TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_cache_component(value: str, *, description: str) -> str:
    """Return *value* when it is safe for inclusion in cache filenames."""

    if not value:
        raise ValueError(f"{description} component is empty")
    if value.startswith(".") or ".." in value:
        raise ValueError(f"{description} component contains unsafe dot segments")
    if "/" in value or "\\" in value:
        raise ValueError(f"{description} component contains path separators")
    if not _SAFE_CACHE_TOKEN.fullmatch(value):
        raise ValueError(f"{description} component contains invalid characters")
    return value


def _sanitize_timeframe(timeframe: str) -> str:
    """Sanitize timeframe string using utility helper."""

    utils = _utils_module()
    sanitize_timeframe = getattr(utils, "sanitize_timeframe")
    return sanitize_timeframe(timeframe)


@lru_cache(maxsize=1)
def _data_handler_utils() -> ModuleType:
    """Return the data handler utilities module regardless of import style."""

    try:
        return importlib.import_module("bot.data_handler.utils")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        if exc.name not in {"bot", "bot.data_handler", "bot.data_handler.utils"}:
            raise
    return importlib.import_module("data_handler.utils")


def _ensure_utc(ts):
    """Import ``ensure_utc`` lazily to support multiple import styles."""
    module = _data_handler_utils()
    ensure_utc = getattr(module, "ensure_utc")
    return ensure_utc(ts)


class HistoricalDataCache:
    """Manage on-disk storage for historical OHLCV data."""

    def __init__(self, cache_dir="/app/cache", min_free_disk_gb=0.1):
        raw_path = Path(cache_dir)
        self.cache_dir = str(raw_path)
        self.min_free_disk_gb = min_free_disk_gb

        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            if raw_path.is_symlink():
                raise ValueError(
                    f"Cache directory {raw_path} must not be a symbolic link"
                )
            resolved_path = raw_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Failed to resolve cache directory {raw_path}: {exc}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Failed to inspect cache directory {raw_path}: {exc}"
            ) from exc

        self._base_path = resolved_path
        self.cache_dir = str(resolved_path)

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
                if os.path.islink(fp):
                    continue
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
        target = Path(path)
        try:
            target.relative_to(self._base_path)
        except ValueError:
            logger.warning("Пропуск удаления файла вне каталога кэша: %s", target)
            return
        if not target.exists():
            return
        try:
            file_size_mb = target.stat().st_size / (1024 * 1024)
        except OSError:
            file_size_mb = 0
        try:
            target.unlink()
            # Guard against negative cache sizes in case accounting drifts
            self.current_cache_size_mb = max(
                self.current_cache_size_mb - file_size_mb,
                0,
            )
        except OSError as e:  # pragma: no cover - unexpected deletion failure
            logger.error("Ошибка удаления файла кэша %s: %s", target, e)

    def _cache_file(self, safe_symbol: str, safe_timeframe: str, suffix: str) -> Path:
        symbol_component = _validate_cache_component(safe_symbol, description="symbol")
        timeframe_component = _validate_cache_component(
            safe_timeframe, description="timeframe"
        )
        filename = f"{symbol_component}_{timeframe_component}{suffix}"
        candidate = self._base_path / filename
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self._base_path)
        except ValueError as exc:
            raise ValueError("cache path escapes base directory") from exc
        return resolved

    def _ensure_not_symlink(self, path: Path) -> bool:
        try:
            if path.is_symlink():
                logger.warning(
                    "Обнаружена символическая ссылка в каталоге кэша: %s", path
                )
                self._delete_cache_file(path)
                return False
        except OSError as exc:
            logger.warning("Не удалось проверить символическую ссылку %s: %s", path, exc)
            return False
        return True

    def _open_secure_for_write(self, path: Path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if nofollow:
            flags |= nofollow
        try:
            fd = os.open(path, flags, 0o600)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EPERM):
                logger.warning(
                    "Отказ записи в кэш: путь %s является символической ссылкой", path
                )
            raise
        return os.fdopen(fd, "wb")

    def _open_secure_for_read(self, path: Path):
        flags = os.O_RDONLY
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if nofollow:
            flags |= nofollow
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EPERM):
                logger.warning(
                    "Отказ чтения кэша: путь %s является символической ссылкой",
                    path,
                )
            raise
        try:
            return os.fdopen(fd, "rb")
        except Exception:
            os.close(fd)
            raise

    def _aggressive_clean(self):
        try:
            files = []
            for entry in self._base_path.iterdir():
                try:
                    if not entry.is_file() or entry.is_symlink():
                        continue
                    files.append((entry, entry.stat().st_mtime))
                except OSError:
                    continue
            if not files:
                return
            files.sort(key=lambda x: x[1])
            target_size = self.max_cache_size_mb * 0.5
            while self.current_cache_size_mb > target_size and files:
                file_path, _ = files.pop(0)
                self._delete_cache_file(file_path)
                logger.info(
                    "Удален файл кэша (агрессивная очистка): %s",
                    file_path.name,
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
        try:
            safe_symbol = _sanitize_symbol(symbol)
        except ValueError:
            logger.error(
                "Невозможно кэшировать %s_%s: недопустимое значение символа",
                symbol,
                timeframe,
            )
            return
        try:
            safe_timeframe = _sanitize_timeframe(timeframe)
        except ValueError:
            logger.error(
                "Невозможно кэшировать %s_%s: недопустимое значение таймфрейма",
                symbol,
                timeframe,
            )
            return
        if isinstance(data, pd.DataFrame) and data.empty:
            return
        if not self._check_disk_space():
            logger.error(
                "Невозможно кэшировать %s_%s: нехватка места на диске",
                symbol,
                timeframe,
            )
            return
        filename = self._cache_file(safe_symbol, safe_timeframe, ".parquet")
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
            if filename.exists() and self._ensure_not_symlink(filename):
                try:
                    old_size_mb = filename.stat().st_size / (1024 * 1024)
                except OSError:
                    old_size_mb = 0
            with self._open_secure_for_write(filename) as fh:
                fh.write(parquet_bytes)
            try:
                compressed_size_mb = filename.stat().st_size / (1024 * 1024)
            except OSError:
                compressed_size_mb = len(parquet_bytes) / (1024 * 1024)
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
            try:
                safe_symbol = _sanitize_symbol(symbol)
            except ValueError:
                logger.warning(
                    "Запрошен недопустимый символ %s для таймфрейма %s",
                    symbol,
                    timeframe,
                )
                return None
            try:
                safe_timeframe = _sanitize_timeframe(timeframe)
            except ValueError:
                logger.warning(
                    "Запрошен недопустимый таймфрейм %s для символа %s",
                    timeframe,
                    symbol,
                )
                return None
            filename = self._cache_file(safe_symbol, safe_timeframe, ".parquet")
            legacy_json = self._cache_file(safe_symbol, safe_timeframe, ".json.gz")
            old_gzip = self._cache_file(safe_symbol, safe_timeframe, ".pkl.gz")
            old_filename = self._cache_file(safe_symbol, safe_timeframe, ".pkl")
            if filename.exists():
                if not self._ensure_not_symlink(filename):
                    return None
                if time.time() - filename.stat().st_mtime > self.cache_ttl:
                    logger.info("Кэш для %s_%s устарел, удаление", symbol, timeframe)
                    self._delete_cache_file(filename)
                    return None
                start_time = time.time()
                try:
                    with self._open_secure_for_read(filename) as fh:
                        data = pd.read_parquet(fh)
                    fmt = "parquet"
                except OSError as exc:
                    if exc.errno in (errno.ELOOP, errno.EPERM):
                        logger.warning(
                            "Обнаружен небезопасный кэш %s_%s: %s",
                            symbol,
                            timeframe,
                            exc,
                        )
                        self._delete_cache_file(filename)
                        return None
                    raise
                except Exception as exc:
                    logger.warning(
                        "Parquet read failed, attempting JSON: %s",
                        exc,
                    )
                    try:
                        with self._open_secure_for_read(filename) as fh:
                            raw_json = fh.read()
                    except OSError as sec_exc:
                        if sec_exc.errno in (errno.ELOOP, errno.EPERM):
                            logger.warning(
                                "Обнаружен небезопасный JSON-кэш %s_%s: %s",
                                symbol,
                                timeframe,
                                sec_exc,
                            )
                            self._delete_cache_file(filename)
                            return None
                        raise
                    data = pd.read_json(
                        StringIO(raw_json.decode("utf-8")), orient="split"
                    )
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
                        data["timestamp"] = _ensure_utc(data["timestamp"])
                    elif isinstance(data.index, pd.MultiIndex) and "timestamp" in data.index.names:
                        # ``MultiIndex.set_levels`` requires providing the
                        # unique level values which is awkward for repeated
                        # timestamps.  Constructing a new MultiIndex from the
                        # level arrays preserves order while allowing us to
                        # normalise the ``timestamp`` level to UTC.
                        names = list(data.index.names)
                        arrays = []
                        for pos, name in enumerate(names):
                            level_values = data.index.get_level_values(pos)
                            if name == "timestamp":
                                level_values = _ensure_utc(level_values)
                            arrays.append(level_values)
                        data.index = pd.MultiIndex.from_arrays(arrays, names=names)
                    elif isinstance(data.index, pd.DatetimeIndex):
                        data.index = _ensure_utc(data.index)
                return data
            if legacy_json.exists() and self._ensure_not_symlink(legacy_json):
                logger.info(
                    "Обнаружен старый кэш для %s_%s, конвертация в Parquet",
                    symbol,
                    timeframe,
                )
                try:
                    with self._open_secure_for_read(legacy_json) as fh:
                        compressed = fh.read()
                except OSError as exc:
                    if exc.errno in (errno.ELOOP, errno.EPERM):
                        logger.warning(
                            "Обнаружен небезопасный legacy-кэш %s_%s: %s",
                            symbol,
                            timeframe,
                            exc,
                        )
                        self._delete_cache_file(legacy_json)
                        return None
                    raise
                try:
                    payload = json.loads(
                        gzip.decompress(compressed).decode("utf-8")
                    )
                except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.error(
                        "Не удалось прочитать старый JSON-кэш %s_%s: %s",
                        symbol,
                        timeframe,
                        exc,
                    )
                    self._delete_cache_file(legacy_json)
                    return None
                data_json = payload.get("data")
                data = pd.read_json(StringIO(data_json), orient="split")
                if isinstance(data, pd.DataFrame):
                    if "timestamp" in data.columns:
                        data["timestamp"] = _ensure_utc(data["timestamp"])
                    elif isinstance(data.index, pd.DatetimeIndex):
                        data.index = _ensure_utc(data.index)
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
                if legacy.exists() and self._ensure_not_symlink(legacy):
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
