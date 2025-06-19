import logging
import os
import pickle
import pandas as pd
import numpy as np
from telegram.ext import Application
import torch
import asyncio
import time
from scipy.stats import zscore
import gzip
import psutil
import shutil
from numba import jit, prange

logger = logging.getLogger('TradingBot')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log_dir = os.getenv('LOG_DIR', '/app/logs')
os.makedirs(log_dir, exist_ok=True)
if not os.access(log_dir, os.W_OK):
    raise PermissionError(f"Нет прав на запись в директорию логов: {log_dir}")

file_handler = logging.FileHandler(os.path.join(log_dir, 'trading_bot.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class TelegramLogger(logging.Handler):
    def __init__(self, bot, chat_id, level=logging.NOTSET):
        super().__init__(level)
        self.bot = bot
        self.chat_id = chat_id
        self.last_message_time = 0
        self.message_interval = 1800
        self.message_lock = asyncio.Lock()

    async def send_telegram_message(self, message):
        async with self.message_lock:
            try:
                if time.time() - self.last_message_time >= self.message_interval:
                    await self.bot.send_message(chat_id=self.chat_id, text=message[:4096])
                    self.last_message_time = time.time()
                else:
                    logger.debug(f"Сообщение Telegram пропущено из-за интервала: {message[:100]}...")
            except Exception as e:
                logger.error(f"Ошибка отправки сообщения Telegram: {e}")

    def emit(self, record):
        try:
            msg = self.format(record)
            asyncio.create_task(self.send_telegram_message(msg))
        except Exception as e:
            logger.error(f"Ошибка в TelegramLogger: {e}")

def check_dataframe_empty(df, context=""):
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
    except Exception as e:
        logger.error(f"Ошибка проверки DataFrame в контексте {context}: {e}")
        return False

def filter_outliers_zscore(df, column='close', threshold=3.0):
    try:
        if len(df[column].dropna()) < 3:
            logger.warning(f"Недостаточно данных для z-оценки в {column}, возвращается исходный DataFrame")
            return df
        z_scores = zscore(df[column].dropna())
        volatility = df[column].pct_change().std()
        adjusted_threshold = threshold * (1 + volatility / 0.02)
        df_filtered = df[np.abs(z_scores) <= adjusted_threshold]
        if len(df_filtered) < len(df):
            logger.info(f"Удалено {len(df) - len(df_filtered)} аномалий в {column} с z-оценкой, порог={adjusted_threshold:.2f}")
        return df_filtered
    except Exception as e:
        logger.error(f"Ошибка фильтрации аномалий в {column}: {e}")
        return df

@jit(nopython=True, parallel=True)
def calculate_volume_profile(prices, volumes, bins=50):
    try:
        if len(prices) != len(volumes) or len(prices) < 2:
            return np.zeros(bins)
        min_price = np.min(prices)
        max_price = np.max(prices)
        if min_price == max_price:
            return np.zeros(bins)
        bin_edges = np.linspace(min_price, max_price, bins + 1)
        volume_profile = np.zeros(bins)
        for i in prange(len(prices)):
            bin_idx = np.searchsorted(bin_edges, prices[i], side='right') - 1
            if 0 <= bin_idx < bins:
                volume_profile[bin_idx] += volumes[i]
        return volume_profile / (np.sum(volume_profile) + 1e-6)
    except Exception as e:
        logger.error(f"Ошибка вычисления профиля объема: {e}")
        return np.zeros(bins)

class HistoricalDataCache:
    def __init__(self, cache_dir='/app/cache'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.access(self.cache_dir, os.W_OK):
            raise PermissionError(f"Нет прав на запись в директорию кэша: {self.cache_dir}")
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
        if disk_usage.free / (1024 ** 3) < 0.5:
            logger.warning(f"Недостаточно свободного места на диске: {disk_usage.free / (1024 ** 3):.2f} ГБ")
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
        return (self.current_cache_size_mb + additional_size_mb) < available_mb * self.memory_threshold

    def _aggressive_clean(self):
        try:
            current_time = time.time()
            files = [(f, os.path.getmtime(os.path.join(self.cache_dir, f)))
                     for f in os.listdir(self.cache_dir) if os.path.isfile(os.path.join(self.cache_dir, f))]
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
                logger.info(f"Удален файл кэша (агрессивная очистка): {file_name}, освобождено {file_size_mb:.2f} МБ")
        except Exception as e:
            logger.error(f"Ошибка агрессивной очистки кэша: {e}")

    def _check_buffer_size(self):
        buffer_size_mb = self._calculate_cache_size()
        if buffer_size_mb > self.max_buffer_size_mb:
            logger.warning(f"Дисковый буфер превысил лимит {self.max_buffer_size_mb} МБ, очистка")
            self._aggressive_clean()

    def save_cached_data(self, symbol, timeframe, data):
        try:
            if not self._check_disk_space():
                logger.error(f"Невозможно кэшировать {symbol}_{timeframe}: нехватка места на диске")
                return
            filename = os.path.join(self.cache_dir, f"{symbol}_{timeframe}.pkl.gz")
            temp_filename = os.path.join(self.cache_dir, f"temp_{symbol}_{timeframe}.pkl")
            start_time = time.time()
            with open(temp_filename, 'wb') as f:
                pickle.dump(data, f)
            file_size_mb = os.path.getsize(temp_filename) / (1024 * 1024)
            os.remove(temp_filename)
            if not self._check_memory(file_size_mb):
                logger.warning(f"Недостаточно памяти для кэширования {symbol}_{timeframe}, очистка кэша")
                self._aggressive_clean()
                if not self._check_memory(file_size_mb):
                    logger.error(f"Невозможно кэшировать {symbol}_{timeframe}: нехватка памяти")
                    return
            self._check_buffer_size()
            with gzip.open(filename, 'wb') as f:
                pickle.dump(data, f)
            compressed_size_mb = os.path.getsize(filename) / (1024 * 1024)
            self.current_cache_size_mb += compressed_size_mb
            elapsed_time = time.time() - start_time
            if elapsed_time > 0.5:
                logger.warning(f"Высокая задержка сжатия gzip для {symbol}_{timeframe}: {elapsed_time:.2f} сек")
            logger.info(f"Данные кэшированы (gzip): {filename}, размер {compressed_size_mb:.2f} МБ")
            self._aggressive_clean()
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша для {symbol}_{timeframe}: {e}")

    def load_cached_data(self, symbol, timeframe):
        try:
            filename = os.path.join(self.cache_dir, f"{symbol}_{timeframe}.pkl.gz")
            if os.path.exists(filename):
                if time.time() - os.path.getmtime(filename) > self.cache_ttl:
                    logger.info(f"Кэш для {symbol}_{timeframe} устарел, удаление")
                    os.remove(filename)
                    return None
                start_time = time.time()
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                elapsed_time = time.time() - start_time
                if elapsed_time > 0.5:
                    logger.warning(f"Высокая задержка чтения gzip для {symbol}_{timeframe}: {elapsed_time:.2f} сек")
                logger.info(f"Данные загружены из кэша (gzip): {filename}")
                return data
            old_filename = os.path.join(self.cache_dir, f"{symbol}_{timeframe}.pkl")
            if os.path.exists(old_filename):
                logger.info(f"Обнаружен старый кэш для {symbol}_{timeframe}, конвертация в gzip")
                with open(old_filename, 'rb') as f:
                    data = pickle.load(f)
                self.save_cached_data(symbol, timeframe, data)
                os.remove(old_filename)
                return data
            return None
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша для {symbol}_{timeframe}: {e}")
            return None
