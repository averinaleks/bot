import pandas as pd
import numpy as np
import optuna
import asyncio
import time
from utils import logger, check_dataframe_empty
from optuna.samplers import TPESampler

class ParameterOptimizer:
    def __init__(self, config, data_handler):
        self.config = config
        self.data_handler = data_handler
        self.base_optimization_interval = config.get('optimization_interval', 14400) // 2  # Уменьшен базовый интервал
        self.last_optimization = {symbol: 0 for symbol in data_handler.usdt_pairs}
        self.best_params_by_symbol = {symbol: {} for symbol in data_handler.usdt_pairs}
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.last_atr_update = {symbol: 0 for symbol in data_handler.usdt_pairs}
        self.atr_update_interval = 5  # Обновление ATR каждые 5 свечей
        self.last_volatility = {symbol: 0.0 for symbol in data_handler.usdt_pairs}  # Для отслеживания изменений волатильности
        self.max_trials = config.get('optuna_trials', 20)

    async def optimize(self, symbol):
        # Оптимизация гиперпараметров для символа
        try:
            ohlcv = self.data_handler.ohlcv
            if 'symbol' in ohlcv.index.names and symbol in ohlcv.index.get_level_values('symbol'):
                df = ohlcv.xs(symbol, level='symbol', drop_level=False)
            else:
                df = None
            if check_dataframe_empty(df, f"optimize {symbol}"):
                logger.warning(f"Нет данных для оптимизации {symbol}")
                return self.best_params_by_symbol.get(symbol, {}) or self.config
            volatility = df['close'].pct_change().std() if not df.empty else 0.02
            # Проверка значительного изменения волатильности
            volatility_change = abs(volatility - self.last_volatility.get(symbol, 0.0)) / max(self.last_volatility.get(symbol, 0.01), 0.01)
            self.last_volatility[symbol] = volatility
            optimization_interval = self.base_optimization_interval / (1 + volatility / self.volatility_threshold)
            optimization_interval = max(1800, min(self.base_optimization_interval * 2, optimization_interval))  # Минимум 30 минут
            if time.time() - self.last_optimization.get(symbol, 0) < optimization_interval and volatility_change < 0.5:
                logger.info(f"Оптимизация для {symbol} не требуется, следующая через {optimization_interval - (time.time() - self.last_optimization.get(symbol, 0)):.0f} секунд")
                return self.best_params_by_symbol.get(symbol, {}) or self.config
            # Использование TPESampler с multivariate=True для учета корреляций
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(
                    n_startup_trials=10,
                    multivariate=True,
                    warn_independent_sampling=False
                )
            )
            study.optimize(lambda trial: self.objective(trial, symbol, df), n_trials=self.max_trials)
            best_params = {**self.config, **study.best_params}
            if not self.validate_params(best_params):
                logger.warning(f"Некорректные параметры для {symbol}, использование предыдущих")
                return self.best_params_by_symbol.get(symbol, {}) or self.config
            self.best_params_by_symbol[symbol] = best_params
            self.last_optimization[symbol] = time.time()
            logger.info(f"Оптимизация для {symbol} завершена, лучшие параметры: {best_params}")
            return best_params
        except Exception as e:
            logger.error(f"Ошибка оптимизации для {symbol}: {e}")
            return self.best_params_by_symbol.get(symbol, {}) or self.config

    def validate_params(self, params):
        # Валидация оптимизированных параметров
        try:
            ema30 = params.get('ema30_period', self.config.get('ema30_period'))
            ema100 = params.get('ema100_period', self.config.get('ema100_period'))
            ema200 = params.get('ema200_period', self.config.get('ema200_period'))
            if ema30 >= ema100 or ema100 >= ema200:
                return False

            tp_mult = params.get('tp_multiplier', self.config.get('tp_multiplier'))
            sl_mult = params.get('sl_multiplier', self.config.get('sl_multiplier'))
            if tp_mult < sl_mult:
                return False

            base_prob = params.get('base_probability_threshold', self.config.get('base_probability_threshold'))
            if not (0.1 <= base_prob <= 0.9):
                return False

            if not (2 <= params.get('loss_streak_threshold', self.config.get('loss_streak_threshold', 2)) <= 5):
                return False

            if not (2 <= params.get('win_streak_threshold', self.config.get('win_streak_threshold', 2)) <= 5):
                return False

            if not (0.01 <= params.get('threshold_adjustment', self.config.get('threshold_adjustment', 0.05)) <= 0.1):
                return False

            return True
        except Exception as e:
            logger.error(f"Ошибка валидации параметров: {e}")
            return False

    def objective(self, trial, symbol, df):
        # Целевая функция для оптимизации
        try:
            ema_periods = [
                trial.suggest_int('ema30_period', 10, 50),
                trial.suggest_int('ema100_period', 50, 200),
                trial.suggest_int('ema200_period', 100, 300),
            ]
            ema_periods.sort()
            ema30_period, ema100_period, ema200_period = ema_periods
            atr_period_default = trial.suggest_int('atr_period_default', 5, 20)
            sl_multiplier = trial.suggest_float('sl_multiplier', 0.5, 2.0)
            tp_multiplier = trial.suggest_float('tp_multiplier', sl_multiplier, 3.0)
            n_splits = 5
            train_size = int(0.6 * len(df))
            test_size = int(0.2 * len(df))
            sharpe_ratios = []
            for i in range(n_splits):
                start = i * test_size
                end = start + train_size + test_size
                if end > len(df):
                    break
                test_df = df.iloc[start + train_size:end].droplevel('symbol')
                from data_handler import IndicatorsCache
                # Обновление ATR при необходимости
                current_candle_count = len(test_df)
                if current_candle_count - self.last_atr_update.get(symbol, 0) >= self.atr_update_interval:
                    indicators = IndicatorsCache(test_df, {
                        'ema30_period': ema30_period,
                        'ema100_period': ema100_period,
                        'ema200_period': ema200_period,
                        'atr_period_default': atr_period_default
                    }, test_df['close'].pct_change().std())
                    self.last_atr_update[symbol] = current_candle_count
                else:
                    indicators = self.data_handler.indicators_cache.get(f"{symbol}_{self.config['timeframe']}")
                    if not indicators:
                        indicators = IndicatorsCache(test_df, {
                            'ema30_period': ema30_period,
                            'ema100_period': ema100_period,
                            'ema200_period': ema200_period,
                            'atr_period_default': atr_period_default
                        }, test_df['close'].pct_change().std())
                if not indicators or check_dataframe_empty(indicators.df, f"objective {symbol}"):
                    return 0.0
                returns = []
                for j in range(1, len(test_df)):
                    close = test_df['close'].iloc[j]
                    prev_close = test_df['close'].iloc[j-1]
                    ema30 = indicators.ema30.iloc[j]
                    ema100 = indicators.ema100.iloc[j]
                    if indicators.volume_profile is not None and not indicators.volume_profile.empty:
                        price_bins = indicators.volume_profile.index.to_numpy()
                        idx = np.searchsorted(price_bins, close)
                        idx = np.clip(idx, 0, len(price_bins) - 1)
                        volume_profile = indicators.volume_profile.iloc[idx]
                    else:
                        volume_profile = 0.0
                    signal = 0
                    if ema30 > ema100 and close > ema30 and volume_profile > 0.02:
                        signal = 1
                    elif ema30 < ema100 and close < ema30 and volume_profile > 0.02:
                        signal = -1
                    if signal != 0:
                        ret = (close - prev_close) / prev_close * signal
                        returns.append(ret)
                if not returns:
                    return 0.0
                returns = np.array(returns, dtype=np.float32)
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(365 * 24 * 60 / pd.Timedelta(self.config['timeframe']).total_seconds())
                if np.isfinite(sharpe_ratio):
                    sharpe_ratios.append(sharpe_ratio)
            return np.mean(sharpe_ratios) if sharpe_ratios else 0.0
        except Exception as e:
            logger.error(f"Ошибка в objective для {symbol}: {e}")
            return 0.0

    async def optimize_all(self):
        # Оптимизация для всех символов
        try:
            tasks = [self.optimize(symbol) for symbol in self.data_handler.usdt_pairs]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for symbol, result in zip(self.data_handler.usdt_pairs, results):
                if not isinstance(result, Exception):
                    logger.info(f"Оптимизация завершена для {symbol}")
                else:
                    logger.error(f"Ошибка оптимизации для {symbol}: {result}")
            return self.best_params_by_symbol
        except Exception as e:
            logger.error(f"Ошибка в optimize_all: {e}")
            return self.best_params_by_symbol
