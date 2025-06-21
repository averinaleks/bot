import asyncio
import json
import time
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
import websockets
from utils import (
    logger,
    check_dataframe_empty,
    HistoricalDataCache,
    filter_outliers_zscore,
    TelegramLogger,
    calculate_volume_profile as utils_volume_profile,
)
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import List, Dict, Optional
import ta
import os
from queue import Queue
import pickle
import psutil
import ray

class IndicatorsCache:
    def __init__(self, df: pd.DataFrame, config: dict, volatility: float, timeframe: str = 'primary'):
        self.df = df
        self.config = config
        self.volatility = volatility
        self.last_volume_profile_update = 0
        self.volume_profile_update_interval = 5
        try:
            if timeframe == 'primary':
                self.ema30 = ta.trend.ema_indicator(df['close'], window=config['ema30_period'], fillna=True)
                self.ema100 = ta.trend.ema_indicator(df['close'], window=config['ema100_period'], fillna=True)
                self.ema200 = ta.trend.ema_indicator(df['close'], window=config['ema200_period'], fillna=True)
                # Используем average_true_range из ta.volatility
                self.atr = ta.volatility.average_true_range(
                    df['high'], df['low'], df['close'],
                    window=config['atr_period_default'], fillna=True
                )
                self.rsi = ta.momentum.rsi(df['close'], window=14, fillna=True)
                self.adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14, fillna=True)
                self.macd = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            elif timeframe == 'secondary':
                self.ema30 = ta.trend.ema_indicator(df['close'], window=config['ema30_period'], fillna=True)
                self.ema100 = ta.trend.ema_indicator(df['close'], window=config['ema100_period'], fillna=True)
            self.volume_profile = None
            if len(df) - self.last_volume_profile_update >= self.volume_profile_update_interval:
                self.volume_profile = self.calculate_volume_profile(df)
                self.last_volume_profile_update = len(df)
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов ({timeframe}): {e}")
            self.ema30 = self.ema100 = self.ema200 = self.atr = self.rsi = self.adx = self.macd = self.volume_profile = None

    def calculate_volume_profile(self, df: pd.DataFrame) -> pd.Series:
        try:
            prices = df['close'].to_numpy(dtype=np.float32)
            volumes = df['volume'].to_numpy(dtype=np.float32)
            vp = utils_volume_profile(prices, volumes, bins=50)
            price_bins = np.linspace(prices.min(), prices.max(), num=len(vp))
            return pd.Series(vp, index=price_bins)
        except Exception as e:
            logger.error(f"Ошибка расчета Volume Profile: {e}")
            return None


@ray.remote(num_cpus=1)
def calc_indicators(df: pd.DataFrame, config: dict, volatility: float, timeframe: str):
    return IndicatorsCache(df, config, volatility, timeframe)

class DataHandler:
    def __init__(self, config: dict, exchange: ccxt_async.bybit, telegram_bot, chat_id):
        self.config = config
        self.exchange = exchange
        self.telegram_logger = TelegramLogger(telegram_bot, chat_id)
        self.cache = HistoricalDataCache(config['cache_dir'])
        self.ohlcv = pd.DataFrame()
        self.ohlcv_2h = pd.DataFrame()
        self.funding_rates = {}
        self.open_interest = {}
        self.orderbook = pd.DataFrame()
        self.indicators = {}
        self.indicators_cache = {}
        self.indicators_2h = {}
        self.indicators_cache_2h = {}
        self.usdt_pairs = []
        self.ohlcv_lock = asyncio.Lock()
        self.ohlcv_2h_lock = asyncio.Lock()
        self.funding_lock = asyncio.Lock()
        self.oi_lock = asyncio.Lock()
        self.orderbook_lock = asyncio.Lock()
        self.cleanup_lock = asyncio.Lock()
        self.ws_rate_timestamps = []
        self.process_rate_timestamps = []
        self.ws_min_process_rate = config.get('ws_min_process_rate', 30)
        self.process_rate_window = 1
        self.cleanup_task = None
        self.ws_queue = asyncio.PriorityQueue(maxsize=config.get('ws_queue_size', 10000))
        self.disk_buffer = Queue(maxsize=config.get('disk_buffer_size', 10000))
        self.buffer_dir = os.path.join(config['cache_dir'], 'ws_buffer')
        os.makedirs(self.buffer_dir, exist_ok=True)
        self.processed_timestamps = {}
        self.processed_timestamps_2h = {}
        self.symbol_priority = {}
        self.backup_ws_urls = config.get('backup_ws_urls', [])
        self.ws_latency = {}
        self.latency_log_interval = 3600
        self.restart_attempts = 0
        self.max_restart_attempts = 20
        self.max_subscriptions = config.get('max_subscriptions_per_connection', 50)
        self.active_subscriptions = 0
        self.load_threshold = 0.8
        self.ws_pool = {}

    async def load_initial(self):
        try:
            markets = await self.exchange.load_markets()
            self.usdt_pairs = await self.select_liquid_pairs(markets)
            logger.info(f"Найдено {len(self.usdt_pairs)} USDT-пар с высокой ликвидностью")
            tasks = []
            for symbol in self.usdt_pairs:
                orderbook = await self.fetch_orderbook(symbol)
                bid_volume = sum([bid[1] for bid in orderbook.get('bids', [])[:5]]) if orderbook.get('bids') else 0
                ask_volume = sum([ask[1] for ask in orderbook.get('asks', [])[:5]]) if orderbook.get('asks') else 0
                liquidity = min(bid_volume, ask_volume)
                self.symbol_priority[symbol] = -liquidity
                tasks.append(self.fetch_ohlcv_single(symbol, self.config['timeframe'], cache_prefix=''))
                tasks.append(self.fetch_ohlcv_single(symbol, self.config['secondary_timeframe'], cache_prefix='2h_'))
                tasks.append(self.fetch_funding_rate(symbol))
                tasks.append(self.fetch_open_interest(symbol))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, tuple) and len(result) == 2:  # Результаты fetch_ohlcv_single
                    symbol, df = result
                    timeframe = self.config['timeframe'] if i % 2 == 0 else self.config['secondary_timeframe']
                    if not check_dataframe_empty(df, f"load_initial {symbol} {timeframe}"):
                        df['symbol'] = symbol
                        df = df.set_index(['symbol', df.index])
                        await self.synchronize_and_update(
                            symbol, df,
                            self.funding_rates.get(symbol, 0.0),
                            self.open_interest.get(symbol, 0.0),
                            {'imbalance': 0.0, 'timestamp': time.time()},
                            timeframe='primary' if timeframe == self.config['timeframe'] else 'secondary'
                        )
        except Exception as e:
            logger.error(f"Ошибка загрузки начальных данных: {e}")
            await self.telegram_logger.send_telegram_message(f"Ошибка загрузки данных: {e}")

    async def select_liquid_pairs(self, markets: Dict) -> List[str]:
        pair_volumes = []
        for symbol, market in markets.items():
            if market.get('active') and symbol.endswith('USDT'):
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    volume = float(ticker.get('quoteVolume') or 0)
                except Exception as e:
                    logger.error(f"Ошибка получения тикера для {symbol}: {e}")
                    volume = 0.0
                pair_volumes.append((symbol, volume))
        pair_volumes.sort(key=lambda x: x[1], reverse=True)
        top_limit = self.config.get('max_subscriptions_per_connection', 50)
        return [s for s, _ in pair_volumes[:top_limit]]

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_ohlcv_single(self, symbol: str, timeframe: str, limit: int = 200, cache_prefix: str = '') -> tuple:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < limit * 0.8:
                logger.warning(f"Неполные данные OHLCV для {symbol} ({timeframe}), получено {len(ohlcv)} из {limit}")
                return symbol, pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(np.float32)
            df = filter_outliers_zscore(df, 'close')
            if df['close'].isna().sum() / len(df) > 0.05:
                logger.warning(f"Слишком много пропусков в данных для {symbol} ({timeframe}) (>5%), использование forward-fill")
                df = df.fillna(method='ffill')
            time_diffs = df.index.to_series().diff().dt.total_seconds()
            max_gap = pd.Timedelta(timeframe).total_seconds() * 2
            if time_diffs.max() > max_gap:
                logger.warning(
                    f"Обнаружен значительный разрыв в данных для {symbol} ({timeframe}): {time_diffs.max()/60:.2f} минут"
                )
                await self.telegram_logger.send_telegram_message(
                    f"⚠️ Разрыв в данных для {symbol} ({timeframe}): {time_diffs.max()/60:.2f} минут"
                )
                return symbol, pd.DataFrame()
            df = df.interpolate(method='time', limit_direction='both')
            self.cache.save_cached_data(f"{cache_prefix}{symbol}", timeframe, df)
            return symbol, pd.DataFrame(df)
        except Exception as e:
            logger.error(f"Ошибка получения OHLCV для {symbol} ({timeframe}): {e}")
            return symbol, pd.DataFrame()

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_funding_rate(self, symbol: str) -> float:
        try:
            futures_symbol = self.fix_symbol(symbol)
            funding = await self.exchange.fetch_funding_rate(futures_symbol)
            rate = float(funding.get('fundingRate', 0.0))
            async with self.funding_lock:
                self.funding_rates[symbol] = rate
            return rate
        except Exception as e:
            logger.error(f"Ошибка получения ставки финансирования для {symbol}: {e}")
            return 0.0

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_open_interest(self, symbol: str) -> float:
        try:
            futures_symbol = self.fix_symbol(symbol)
            oi = await self.exchange.fetch_open_interest(futures_symbol)
            interest = float(oi.get('openInterest', 0.0))
            async with self.oi_lock:
                self.open_interest[symbol] = interest
            return interest
        except Exception as e:
            logger.error(f"Ошибка получения открытого интереса для {symbol}: {e}")
            return 0.0

    @retry(wait=wait_exponential(multiplier=1, min=2, max=5))
    async def fetch_orderbook(self, symbol: str) -> Dict:
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit=10)
            if not orderbook['bids'] or not orderbook['asks']:
                logger.warning(f"Пустая книга ордеров для {symbol}, повторная попытка")
                raise Exception("Пустой ордербук")
            return orderbook
        except Exception as e:
            logger.error(f"Ошибка получения книги ордеров для {symbol}: {e}")
            return {'bids': [], 'asks': []}

    async def synchronize_and_update(self, symbol: str, df: pd.DataFrame, funding_rate: float, open_interest: float, orderbook: dict, timeframe: str = 'primary'):
        try:
            if check_dataframe_empty(df, f"synchronize_and_update {symbol} {timeframe}"):
                logger.warning(f"Пустой DataFrame для {symbol} ({timeframe}), пропуск синхронизации")
                return
            if df['close'].isna().any() or (df['close'] <= 0).any():
                logger.warning(f"Некорректные данные для {symbol} ({timeframe}), пропуск")
                return
            if timeframe == 'primary':
                async with self.ohlcv_lock:
                    if isinstance(self.ohlcv.index, pd.MultiIndex):
                        base = self.ohlcv.drop(symbol, level='symbol', errors='ignore')
                    else:
                        base = self.ohlcv
                    self.ohlcv = pd.concat([base, df], ignore_index=False).sort_index()
            else:
                async with self.ohlcv_2h_lock:
                    if isinstance(self.ohlcv_2h.index, pd.MultiIndex):
                        base = self.ohlcv_2h.drop(symbol, level='symbol', errors='ignore')
                    else:
                        base = self.ohlcv_2h
                    self.ohlcv_2h = pd.concat([base, df], ignore_index=False).sort_index()
            async with self.funding_lock:
                self.funding_rates[symbol] = funding_rate
            async with self.oi_lock:
                self.open_interest[symbol] = open_interest
            async with self.orderbook_lock:
                orderbook_df = pd.DataFrame([orderbook | {'symbol': symbol, 'timestamp': time.time()}])
                self.orderbook = pd.concat([self.orderbook, orderbook_df], ignore_index=False)
            volatility = df['close'].pct_change().std() if not df.empty else 0.02
            cache_key = f"{symbol}_{timeframe}"
            if timeframe == 'primary':
                async with self.ohlcv_lock:
                    if cache_key not in self.indicators_cache:
                        obj_ref = calc_indicators.remote(
                            df.droplevel('symbol'), self.config, volatility, 'primary'
                        )
                        self.indicators_cache[cache_key] = await ray.get(obj_ref)
                    self.indicators[symbol] = self.indicators_cache[cache_key]
            else:
                async with self.ohlcv_2h_lock:
                    if cache_key not in self.indicators_cache_2h:
                        obj_ref = calc_indicators.remote(
                            df.droplevel('symbol'), self.config, volatility, 'secondary'
                        )
                        self.indicators_cache_2h[cache_key] = await ray.get(obj_ref)
                    self.indicators_2h[symbol] = self.indicators_cache_2h[cache_key]
            self.cache.save_cached_data(f"{timeframe}_{symbol}", timeframe, df)
        except Exception as e:
            logger.error(f"Ошибка синхронизации данных для {symbol} ({timeframe}): {e}")

    async def cleanup_old_data(self):
        while True:
            try:
                async with self.cleanup_lock:
                    current_time = pd.Timestamp.now(tz='UTC')
                    async with self.ohlcv_lock:
                        if not self.ohlcv.empty:
                            threshold = current_time - pd.Timedelta(seconds=self.config['forget_window'])
                            self.ohlcv = self.ohlcv[self.ohlcv.index.get_level_values('timestamp') >= threshold]
                    async with self.ohlcv_2h_lock:
                        if not self.ohlcv_2h.empty:
                            threshold = current_time - pd.Timedelta(seconds=self.config['forget_window'])
                            self.ohlcv_2h = self.ohlcv_2h[self.ohlcv_2h.index.get_level_values('timestamp') >= threshold]
                    async with self.orderbook_lock:
                        if not self.orderbook.empty and 'timestamp' in self.orderbook.columns:
                            self.orderbook = self.orderbook[
                                self.orderbook['timestamp'] >= time.time() - self.config['forget_window']
                            ]
                    async with self.ohlcv_lock:
                        for symbol in list(self.processed_timestamps.keys()):
                            if symbol not in self.usdt_pairs:
                                del self.processed_timestamps[symbol]
                    async with self.ohlcv_2h_lock:
                        for symbol in list(self.processed_timestamps_2h.keys()):
                            if symbol not in self.usdt_pairs:
                                del self.processed_timestamps_2h[symbol]
                    logger.info("Старые данные очищены")
                await asyncio.sleep(self.config['data_cleanup_interval'] * 2)
            except Exception as e:
                logger.error(f"Ошибка очистки данных: {e}")
                await asyncio.sleep(60)

    async def save_to_disk_buffer(self, priority, item):
        try:
            filename = os.path.join(self.buffer_dir, f"buffer_{time.time()}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump((priority, item), f)
            self.disk_buffer.put(filename)
            logger.info(f"Сообщение сохранено в дисковый буфер: {filename}")
        except Exception as e:
            logger.error(f"Ошибка сохранения в дисковый буфер: {e}")

    async def load_from_disk_buffer(self):
        while not self.disk_buffer.empty():
            try:
                filename = self.disk_buffer.get()
                with open(filename, 'rb') as f:
                    priority, item = pickle.load(f)
                await self.ws_queue.put((priority, item))
                os.remove(filename)
                logger.info(f"Сообщение загружено из дискового буфера: {filename}")
            except Exception as e:
                logger.error(f"Ошибка загрузки из дискового буфера: {e}")

    async def adjust_subscriptions(self):
        cpu_load = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_load = memory.percent
        current_rate = len(self.process_rate_timestamps) / self.process_rate_window if self.process_rate_timestamps else self.ws_min_process_rate
        if cpu_load > self.load_threshold * 100 or memory_load > self.load_threshold * 100:
            new_max = max(10, self.max_subscriptions // 2)
            logger.warning(f"Высокая нагрузка (CPU: {cpu_load}%, Memory: {memory_load}%), уменьшение подписок до {new_max}")
            self.max_subscriptions = new_max
        elif current_rate < self.ws_min_process_rate:
            new_max = max(10, int(self.max_subscriptions * 0.8))
            logger.warning(f"Низкая скорость обработки ({current_rate:.2f}/s), уменьшение подписок до {new_max}")
            self.max_subscriptions = new_max
        elif cpu_load < self.load_threshold * 50 and memory_load < self.load_threshold * 50 and current_rate > self.ws_min_process_rate * 1.5:
            new_max = min(100, self.max_subscriptions * 2)
            logger.info(f"Низкая нагрузка, увеличение подписок до {new_max}")
            self.max_subscriptions = new_max

    async def subscribe_to_klines(self, symbols: List[str]):
        try:
            self.cleanup_task = asyncio.create_task(self.cleanup_old_data())
            chunk_size = self.max_subscriptions
            tasks = []
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                tasks.append(self._subscribe_chunk(chunk, self.config['ws_url'], self.config['ws_reconnect_interval'], timeframe='primary'))
                tasks.append(self._subscribe_chunk(chunk, self.config['ws_url'], self.config['ws_reconnect_interval'], timeframe='secondary'))
            tasks.append(self._process_ws_queue())
            tasks.append(self.load_from_disk_buffer())
            tasks.append(self.monitor_load())
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Ошибка подписки на WebSocket: {e}")
            await self.telegram_logger.send_telegram_message(f"Ошибка WebSocket: {e}")

    async def monitor_load(self):
        while True:
            try:
                await self.adjust_subscriptions()
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Ошибка мониторинга нагрузки: {e}")
                await asyncio.sleep(60)

    def fix_symbol(self, symbol: str) -> str:
        return symbol.replace('/', '').replace('USDT', 'USDT:USDT')

    async def _subscribe_chunk(self, symbols, ws_url, connection_timeout, timeframe: str = 'primary'):
        reconnect_attempts = 0
        max_reconnect_attempts = self.config.get('max_reconnect_attempts', 10)
        urls = [ws_url] + self.backup_ws_urls
        current_url_index = 0
        selected_timeframe = self.config['timeframe'] if timeframe == 'primary' else self.config['secondary_timeframe']
        while True:
            current_url = urls[current_url_index % len(urls)]
            ws = None
            try:
                if current_url not in self.ws_pool:
                    self.ws_pool[current_url] = []
                if not self.ws_pool[current_url]:
                    ws = await websockets.connect(
                        current_url,
                        ping_interval=20,
                        ping_timeout=30,
                        open_timeout=connection_timeout
                    )
                    self.ws_pool[current_url].append(ws)
                else:
                    ws = self.ws_pool[current_url].pop(0)
                logger.info(f"Подключение к WebSocket {current_url} для {len(symbols)} символов ({timeframe})")
                subscription_tasks = []
                start_time = time.time()
                for symbol in symbols:
                    current_time = time.time()
                    self.ws_rate_timestamps.append(current_time)
                    self.ws_rate_timestamps = [t for t in self.ws_rate_timestamps if current_time - t < 1]
                    if len(self.ws_rate_timestamps) > self.config['ws_rate_limit']:
                        logger.warning(f"Превышен лимит подписок WebSocket, ожидание")
                        await asyncio.sleep(1)
                    subscription_tasks.append(ws.send(json.dumps({
                        'method': 'SUBSCRIBE',
                        'params': [f"kline.{selected_timeframe}.{self.fix_symbol(symbol).lower()}"],
                        'id': 1
                    })))
                await asyncio.gather(*subscription_tasks)
                reconnect_attempts = 0
                current_url_index = 0
                self.restart_attempts = 0
                self.active_subscriptions += len(symbols)
                # Проверка успешной подписки
                subscribed = False
                for _ in range(3):
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=5)
                        if json.loads(response).get('result') == 'ok':
                            subscribed = True
                            break
                    except asyncio.TimeoutError:
                        continue
                if not subscribed:
                    logger.warning(f"Не удалось подтвердить подписку для {symbols} ({timeframe}), повторная попытка")
                    raise Exception("Подписка не подтверждена")
                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=connection_timeout)
                        latency = time.time() - start_time
                        for symbol in symbols:
                            self.ws_latency[symbol] = latency
                        if latency > 5:
                            logger.warning(f"Высокая задержка WebSocket для {symbols} ({timeframe}): {latency:.2f} сек")
                            await self.telegram_logger.send_telegram_message(f"⚠️ Высокая задержка WebSocket для {symbols} ({timeframe}): {latency:.2f} сек")
                            for symbol in symbols:
                                symbol_df = await self.fetch_ohlcv_single(symbol, selected_timeframe, limit=1, cache_prefix='2h_' if timeframe == 'secondary' else '')
                                if isinstance(symbol_df, tuple) and len(symbol_df) == 2:
                                    _, df = symbol_df
                                    if not check_dataframe_empty(df, f"subscribe_to_klines {symbol} {timeframe}"):
                                        df['symbol'] = symbol
                                        df = df.set_index(['symbol', df.index])
                                        await self.synchronize_and_update(
                                            symbol, df,
                                            self.funding_rates.get(symbol, 0.0),
                                            self.open_interest.get(symbol, 0.0),
                                            {'imbalance': 0.0, 'timestamp': time.time()},
                                            timeframe=timeframe
                                        )
                            break
                        try:
                            data = json.loads(message)
                            symbol = data.get('data', {}).get('k', {}).get('s', '')
                            priority = self.symbol_priority.get(symbol, 0)
                            try:
                                await self.ws_queue.put((priority, (symbols, message, timeframe)), timeout=5)
                            except asyncio.TimeoutError:
                                logger.warning(f"Очередь WebSocket переполнена, сохранение в дисковый буфер")
                                await self.save_to_disk_buffer(priority, (symbols, message, timeframe))
                        except asyncio.TimeoutError:
                            logger.warning(f"Очередь WebSocket переполнена, сохранение в дисковый буфер")
                            await self.save_to_disk_buffer(priority, (symbols, message, timeframe))
                            continue
                    except asyncio.TimeoutError:
                        logger.warning(f"Тайм-аут WebSocket для {symbols} ({timeframe}), отправка пинга")
                        await ws.ping()
                        continue
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.error(f"WebSocket соединение закрыто для {symbols} ({timeframe}): {e}")
                        break
                    except Exception as e:
                        logger.error(f"Ошибка обработки WebSocket сообщения для {symbols} ({timeframe}): {e}")
                        break
            except Exception as e:
                reconnect_attempts += 1
                current_url_index += 1
                delay = min(2 ** reconnect_attempts, 60)
                logger.error(f"Ошибка WebSocket {current_url} для {symbols} ({timeframe}), попытка {reconnect_attempts}/{max_reconnect_attempts}, ожидание {delay} секунд: {e}")
                await asyncio.sleep(delay)
                if reconnect_attempts >= max_reconnect_attempts:
                    self.restart_attempts += 1
                    if self.restart_attempts >= self.max_restart_attempts:
                        logger.critical(f"Превышено максимальное количество перезапусков WebSocket для {symbols} ({timeframe})")
                        await self.telegram_logger.send_telegram_message(f"Критическая ошибка: Не удалось восстановить WebSocket для {symbols} ({timeframe})")
                        break
                    logger.info(f"Автоматический перезапуск WebSocket для {symbols} ({timeframe}), попытка {self.restart_attempts}/{self.max_restart_attempts}")
                    reconnect_attempts = 0
                    current_url_index = 0
                    await asyncio.sleep(60)
                else:
                    logger.info(f"Попытка загрузки данных через REST API для {symbols} ({timeframe})")
                    for symbol in symbols:
                        try:
                            symbol_df = await self.fetch_ohlcv_single(symbol, selected_timeframe, limit=1, cache_prefix='2h_' if timeframe == 'secondary' else '')
                            if isinstance(symbol_df, tuple) and len(symbol_df) == 2:
                                _, df = symbol_df
                                if not check_dataframe_empty(df, f"subscribe_to_klines {symbol} {timeframe}"):
                                    df['symbol'] = symbol
                                    df = df.set_index(['symbol', df.index])
                                    await self.synchronize_and_update(
                                        symbol, df,
                                        self.funding_rates.get(symbol, 0.0),
                                        self.open_interest.get(symbol, 0.0),
                                        {'imbalance': 0.0, 'timestamp': time.time()},
                                        timeframe=timeframe
                                    )
                        except Exception as rest_e:
                            logger.error(f"Ошибка REST API для {symbol} ({timeframe}): {rest_e}")
            finally:
                self.active_subscriptions -= len(symbols)
                if ws and ws.open:
                    self.ws_pool[current_url].append(ws)
                elif ws:
                    await ws.close()

    async def _process_ws_queue(self):
        last_latency_log = time.time()
        while True:
            try:
                priority, (symbols, message, timeframe) = await self.ws_queue.get()
                now = time.time()
                self.process_rate_timestamps.append(now)
                self.process_rate_timestamps = [t for t in self.process_rate_timestamps if now - t < self.process_rate_window]
                if len(self.process_rate_timestamps) > self.ws_min_process_rate and (len(self.process_rate_timestamps) / self.process_rate_window) < self.ws_min_process_rate:
                    await self.adjust_subscriptions()
                data = json.loads(message)
                if not isinstance(data, dict) or 'data' not in data or not isinstance(data['data'], dict) or 'k' not in data['data']:
                    logger.debug(f"Error in message data['data'] for {symbols}: {message}")
                    continue
                kline = data['data']['k']
                required_fields = ['s', 't', 'o', 'h', 'l', 'c', 'v']
                if not all(field in kline for field in required_fields):
                    logger.warning(f"Invalid kline data ({timeframe}): {kline}")
                    continue
                try:
                    kline_timestamp = pd.to_datetime(int(kline['t']), unit='ms', utc=True)
                    symbol = str(kline['s'])
                    open_price = float(kline['o'])
                    high_price = float(kline['h'])
                    low_price = float(kline['l'])
                    close_price = float(kline['c'])
                    volume = float(kline['v'])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Ошибка формата данных свечи для {symbol} ({timeframe}): {e}")
                    continue
                timestamp_dict = self.processed_timestamps if timeframe == 'primary' else self.processed_timestamps_2h
                lock = self.ohlcv_lock if timeframe == 'primary' else self.ohlcv_2h_lock
                async with lock:
                    if symbol not in timestamp_dict:
                        timestamp_dict[symbol] = set()
                    if kline['t'] in timestamp_dict[symbol]:
                        logger.debug(f"Дубликат сообщения для {symbol} ({timeframe}) с временной меткой {kline_timestamp}")
                        continue
                    timestamp_dict[symbol].add(kline['t'])
                    if len(timestamp_dict[symbol]) > 1000:
                        timestamp_dict[symbol] = set(list(timestamp_dict[symbol])[-500:])
                current_time = pd.Timestamp.now(tz='UTC')
                if (current_time - kline_timestamp).total_seconds() > 5:
                    logger.warning(f"Получены устаревшие данные для {symbol} ({timeframe}): {kline_timestamp}")
                    continue
                try:
                    df = pd.DataFrame([{
                        'timestamp': kline_timestamp,
                        'open': np.float32(open_price),
                        'high': np.float32(high_price),
                        'low': np.float32(low_price),
                        'close': np.float32(close_price),
                        'volume': np.float32(volume)
                    }])
                    df = filter_outliers_zscore(df, 'close')
                    if df.empty:
                        logger.warning(f"Данные для {symbol} ({timeframe}) отфильтрованы как аномалии")
                        continue
                    df['symbol'] = symbol
                    df = df.set_index(['symbol', 'timestamp'])
                    time_diffs = df.index.get_level_values('timestamp').to_series().diff().dt.total_seconds()
                    max_gap = pd.Timedelta(self.config['timeframe' if timeframe == 'primary' else 'secondary_timeframe']).total_seconds() * 2
                    if time_diffs.max() > max_gap:
                        logger.warning(f"Обнаружен разрыв в данных WebSocket для {symbol} ({timeframe}): {time_diffs.max()/60:.2f} минут")
                        await self.telegram_logger.send_telegram_message(f"⚠️ Разрыв в данных WebSocket для {symbol} ({timeframe}): {time_diffs.max()/60:.2f} минут")
                    await self.synchronize_and_update(
                        symbol, df,
                        self.funding_rates.get(symbol, 0.0),
                        self.open_interest.get(symbol, 0.0),
                        {'imbalance': 0.0, 'timestamp': time.time()},
                        timeframe=timeframe
                    )
                except Exception as e:
                    logger.error(f"Ошибка обработки данных для {symbol}: {e}")
                    continue
                if time.time() - last_latency_log > self.latency_log_interval:
                    rate = len(self.process_rate_timestamps) / self.process_rate_window
                    logger.info(f"Средняя задержка WebSocket: {sum(self.ws_latency.values()) / len(self.ws_latency):.2f} сек, скорость обработки: {rate:.2f}/с")
                    last_latency_log = time.time()
            except Exception as e:
                logger.error(f"Ошибка обработки очереди WebSocket: {e}")
                await asyncio.sleep(2)
            finally:
                self.ws_queue.task_done()
