"""Hyperparameter optimizer for trading indicators and strategies.

Uses Optuna and Ray to search for the best indicator parameters based on
Sharpe ratio and supports an optional grid search refinement step.
"""

import pandas as pd
import numpy as np
import optuna
import asyncio
import time
import os
try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
import inspect
if os.getenv("TEST_MODE") == "1":
    import types
    import sys

    ray = types.ModuleType("ray")

    class _RayRemoteFunction:
        def __init__(self, func):
            self._function = func

        def remote(self, *args, **kwargs):
            return self._function(*args, **kwargs)

        def options(self, *args, **kwargs):
            return self

    def _ray_remote(func=None, **_kwargs):
        if func is None:
            def wrapper(f):
                return _RayRemoteFunction(f)
            return wrapper
        return _RayRemoteFunction(func)

    ray.remote = _ray_remote
    ray.get = lambda x: x
    ray.init = lambda *a, **k: None
    ray.is_initialized = lambda: False
else:
    import ray
from bot.utils import (
    logger,
    is_cuda_available,
    check_dataframe_empty_async as _check_df_async,
)
from bot.config import BotConfig
from optuna.samplers import TPESampler
try:
    from optuna.integration.mlflow import MLflowCallback
except ImportError:  # pragma: no cover - optional dependency may not be installed
    MLflowCallback = None  # type: ignore
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
try:
    from skopt import Optimizer as SkOptimizer
    from skopt.space import Integer, Real
except ImportError as exc:  # pragma: no cover - optional dependency
    logger.warning("skopt import failed: %s", exc)
    SkOptimizer = None  # type: ignore
    Integer = Real = None  # type: ignore


@ray.remote
def _objective_remote(
    df,
    symbol,
    ema30_period,
    ema100_period,
    ema200_period,
    atr_period_default,
    timeframe,
    n_splits=3,
):
    """Heavy part of objective executed remotely."""
    try:
        from bot.data_handler import IndicatorsCache

        train_size = int(0.6 * len(df))
        test_size = int(0.2 * len(df))
        sharpe_ratios = []
        last_atr_update = 0
        atr_update_interval = 5
        for i in range(n_splits):
            start = i * test_size
            end = start + train_size + test_size
            if end > len(df):
                break
            test_df = df.iloc[start + train_size : end].droplevel("symbol")
            # Ensure we have enough data for indicators like ADX that
            # require a minimum window size. Skip this split if the
            # resulting DataFrame is too small.
            if len(test_df) < 14:
                continue
            current_candle_count = len(test_df)
            if current_candle_count - last_atr_update >= atr_update_interval:
                indicators = IndicatorsCache(
                    test_df,
                    {
                        "ema30_period": ema30_period,
                        "ema100_period": ema100_period,
                        "ema200_period": ema200_period,
                        "atr_period_default": atr_period_default,
                    },
                    test_df["close"].pct_change().std(),
                )
                last_atr_update = current_candle_count
            else:
                indicators = IndicatorsCache(
                    test_df,
                    {
                        "ema30_period": ema30_period,
                        "ema100_period": ema100_period,
                        "ema200_period": ema200_period,
                        "atr_period_default": atr_period_default,
                    },
                    test_df["close"].pct_change().std(),
                )
            result = asyncio.run(
                _check_df_async(indicators.df, f"objective {symbol}")
            )
            if not indicators or result:
                return 0.0
            try:
                if is_cuda_available():
                    import cupy as cp  # type: ignore
                    use_gpu = True
                else:
                    cp = np  # type: ignore
                    use_gpu = False
            except ImportError as exc:  # pragma: no cover - fallback if CuPy unavailable
                logger.warning("CuPy import failed, falling back to NumPy: %s", exc)
                cp = np  # type: ignore
                use_gpu = False

            close_arr = cp.asarray(
                test_df["close"].to_numpy(dtype=np.float32)
            )
            ema30_arr = cp.asarray(
                indicators.ema30.to_numpy(dtype=np.float32)
            )
            ema100_arr = cp.asarray(
                indicators.ema100.to_numpy(dtype=np.float32)
            )
            if (
                indicators.volume_profile is not None
                and not indicators.volume_profile.empty
            ):
                price_bins = cp.asarray(
                    indicators.volume_profile.index.to_numpy(dtype=np.float32)
                )
                vp_values = cp.asarray(
                    indicators.volume_profile.to_numpy(dtype=np.float32)
                )
                idx = cp.searchsorted(price_bins, close_arr)
                idx = cp.clip(idx, 0, len(price_bins) - 1)
                volume_profile_arr = vp_values[idx]
            else:
                volume_profile_arr = cp.zeros_like(close_arr)

            long_cond = (
                (ema30_arr > ema100_arr)
                & (close_arr > ema30_arr)
                & (volume_profile_arr > 0.02)
            )
            short_cond = (
                (ema30_arr < ema100_arr)
                & (close_arr < ema30_arr)
                & (volume_profile_arr > 0.02)
            )
            signals = cp.where(long_cond, 1, cp.where(short_cond, -1, 0))

            price_diff = (close_arr[1:] - close_arr[:-1]) / close_arr[:-1]
            returns = price_diff * signals[1:]
            returns = returns[signals[1:] != 0]
            if returns.size == 0:
                return 0.0
            sharpe_ratio = (
                cp.mean(returns)
                / (cp.std(returns) + 1e-6)
                * np.sqrt(365 * 24 * 60 / pd.Timedelta(timeframe).total_seconds())
            )
            sharpe_ratio = (
                float(cp.asnumpy(sharpe_ratio)) if use_gpu else float(sharpe_ratio)
            )
            if np.isfinite(sharpe_ratio):
                sharpe_ratios.append(sharpe_ratio)
        return float(np.mean(sharpe_ratios)) if sharpe_ratios else 0.0
    except (ImportError, ValueError, RuntimeError, KeyError) as e:  # pragma: no cover - log and return
        logger.exception("Ошибка в _objective_remote для %s: %s", symbol, e)
        raise


class ParameterOptimizer:
    def __init__(self, config: BotConfig, data_handler):
        self.config = config
        self.data_handler = data_handler
        self.base_optimization_interval = max(
            float(config.get("optimization_interval", 14400)) / 2, 1.0
        )  # Уменьшен базовый интервал
        self.last_optimization = {symbol: 0 for symbol in data_handler.usdt_pairs}
        self.best_params_by_symbol = {symbol: {} for symbol in data_handler.usdt_pairs}
        self.volatility_threshold = config.get("volatility_threshold", 0.02)
        self.last_atr_update = {symbol: 0 for symbol in data_handler.usdt_pairs}
        self.atr_update_interval = 5  # Обновление ATR каждые 5 свечей
        self.last_volatility = {
            symbol: 0.0 for symbol in data_handler.usdt_pairs
        }  # Для отслеживания изменений волатильности
        self.max_trials = config.get("optuna_trials", 20)
        self.optimizer_method = config.get("optimizer_method", "tpe")
        self.holdout_warning_ratio = config.get("holdout_warning_ratio", 0.3)
        self.n_splits = config.get("n_splits", 3)
        self.mlflow_enabled = config.get("mlflow_enabled", False)
        self.mlflow_tracking_uri = config.get("mlflow_tracking_uri", "mlruns")
        if self.mlflow_enabled and MLflowCallback is None:
            logger.warning(
                "MLflow is enabled but optuna-integration is not installed; "
                "disabling MLflow integration"
            )
            self.mlflow_enabled = False
        self.enable_grid_search = config.get("enable_grid_search", False)

    def get_opt_interval(self, symbol: str, volatility: float) -> float:
        """Return optimization interval for a symbol based on its volatility."""
        try:
            threshold = max(self.volatility_threshold, 1e-6)
            interval = self.base_optimization_interval / (1 + volatility / threshold)
            lower = 1.0 if self.base_optimization_interval < 1800 else 1800
            upper = self.base_optimization_interval * 2
            interval = max(lower, min(upper, interval))
            return interval
        except (ZeroDivisionError, ValueError, TypeError) as e:
            logger.exception(
                "Ошибка расчёта интервала оптимизации для %s: %s",
                symbol,
                e,
            )
            raise

    async def optimize(self, symbol):
        # Оптимизация гиперпараметров для символа
        try:
            ohlcv = self.data_handler.ohlcv
            if (
                "symbol" in ohlcv.index.names
                and symbol in ohlcv.index.get_level_values("symbol")
            ):
                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
            else:
                df = None
            empty = await _check_df_async(df, f"optimize {symbol}")
            if empty:
                logger.warning("Нет данных для оптимизации %s", symbol)
                return self.best_params_by_symbol.get(symbol, {}) or self.config.asdict()
            volatility = df["close"].pct_change().std() if not df.empty else 0.02
            # Проверка значительного изменения волатильности
            volatility_change = abs(
                volatility - self.last_volatility.get(symbol, 0.0)
            ) / max(self.last_volatility.get(symbol, 0.01), 0.01)
            self.last_volatility[symbol] = volatility
            optimization_interval = self.get_opt_interval(symbol, volatility)
            if (
                time.time() - self.last_optimization.get(symbol, 0)
                < optimization_interval
                and volatility_change < 0.5
            ):
                logger.info(
                    "Оптимизация для %s не требуется, следующая через %.0f секунд",
                    symbol,
                    optimization_interval - (time.time() - self.last_optimization.get(symbol, 0)),
                )
                return self.best_params_by_symbol.get(symbol, {}) or self.config.asdict()
            method = self.optimizer_method
            best_value = 0.0
            if method == "gp" and SkOptimizer is not None:
                dims = [
                    Integer(10, 50, name="ema30_period"),
                    Integer(50, 200, name="ema100_period"),
                    Integer(100, 300, name="ema200_period"),
                    Integer(5, 20, name="atr_period_default"),
                    Integer(2, 5, name="loss_streak_threshold"),
                    Integer(2, 5, name="win_streak_threshold"),
                    Real(0.01, 0.1, name="threshold_adjustment"),
                    Real(0.1, 1.0, name="risk_sharpe_loss_factor"),
                    Real(1.0, 2.0, name="risk_sharpe_win_factor"),
                    Real(0.1, 1.0, name="risk_vol_min"),
                    Real(1.0, 3.0, name="risk_vol_max"),
                ]
                optimizer = SkOptimizer(dims)
                obj_refs = []
                params_list = []
                for _ in range(self.max_trials):
                    suggestion = optimizer.ask()
                    param_dict = {
                        dim.name: val for dim, val in zip(dims, suggestion)
                    }
                    ema_periods = [
                        param_dict["ema30_period"],
                        param_dict["ema100_period"],
                        param_dict["ema200_period"],
                    ]
                    ema_periods.sort()
                    (
                        param_dict["ema30_period"],
                        param_dict["ema100_period"],
                        param_dict["ema200_period"],
                    ) = ema_periods
                    obj_ref = self._evaluate_params(param_dict, symbol, df)
                    obj_refs.append(obj_ref)
                    params_list.append((suggestion, param_dict))
                results = await asyncio.to_thread(ray.get, obj_refs)
                for (sugg, _), val in zip(params_list, results):
                    optimizer.tell(sugg, val)
                idx = int(np.argmax(results)) if results else 0
                best_value = results[idx] if results else 0.0
                best_params = {**self.config.asdict(), **params_list[idx][1]}
            else:
                # Использование TPESampler с multivariate=True для учета корреляций
                study = optuna.create_study(
                    direction="maximize",
                    sampler=TPESampler(
                        n_startup_trials=10,
                        multivariate=True,
                        warn_independent_sampling=False,
                    ),
                )
                callbacks = []
                if self.mlflow_enabled:
                    callbacks.append(
                        MLflowCallback(
                            tracking_uri=self.mlflow_tracking_uri,
                            metric_name="sharpe_ratio",
                        )
                    )
                obj_refs = []
                trials = []
                for _ in range(self.max_trials):
                    trial = study.ask()
                    logger.debug(
                        "Dispatching _objective_remote for %s trial %s",
                        symbol,
                        trial.number,
                    )
                    obj_ref = self.objective(trial, symbol, df)
                    obj_refs.append(obj_ref)
                    trials.append(trial)
                results = await asyncio.to_thread(ray.get, obj_refs)
                for t in trials:
                    logger.debug(
                        "Received result for %s trial %s", symbol, t.number
                    )
                for trial, value in zip(trials, results):
                    study.tell(trial, value)
                    for cb in callbacks:
                        cb(study, trial)
                best_value = study.best_value
                best_params = {**self.config.asdict(), **study.best_params}

            if self.enable_grid_search:
                try:
                    best_params = self._grid_search(df, symbol, best_params)
                except (ValueError, RuntimeError) as e:
                    logger.exception("Ошибка GridSearchCV для %s: %s", symbol, e)
                    raise
            if not self.validate_params(best_params):
                logger.warning(
                    "Некорректные параметры для %s, использование предыдущих",
                    symbol,
                )
                return self.best_params_by_symbol.get(symbol, {}) or self.config.asdict()
            self.best_params_by_symbol[symbol] = best_params
            self.last_optimization[symbol] = time.time()

            # Hold-out evaluation
            try:
                start = int(len(df) * 0.8)
                holdout_df = df.iloc[start:]
                if not holdout_df.empty:
                    holdout_score = ray.get(
                        self._evaluate_params(best_params, symbol, holdout_df)
                    )
                    if best_value and holdout_score < best_value * (1 - self.holdout_warning_ratio):
                        logger.warning(
                            "Sharpe ratio on hold-out for %s dropped from %.3f to %.3f",
                            symbol,
                            best_value,
                            holdout_score,
                        )
                        tl = getattr(self.data_handler, "telegram_logger", None)
                        if tl is not None:
                            try:
                                await tl.send_telegram_message(
                                    f"Sharpe ratio drop for {symbol}: {holdout_score:.3f} vs {best_value:.3f}"
                                )
                            except (OSError, RuntimeError) as exc:
                                logger.exception("Failed to send Telegram warning: %s", exc)
            except (ValueError, RuntimeError, KeyError) as e:
                logger.exception("Ошибка hold-out проверки для %s: %s", symbol, e)

            logger.info(
                "Оптимизация для %s завершена, лучшие параметры: %s",
                symbol,
                best_params,
            )
            return best_params
        except (ValueError, RuntimeError, KeyError, ImportError) as e:
            logger.exception("Ошибка оптимизации для %s: %s", symbol, e)
            raise

    def validate_params(self, params):
        # Валидация оптимизированных параметров
        try:
            ema30 = params.get("ema30_period", self.config.get("ema30_period"))
            ema100 = params.get("ema100_period", self.config.get("ema100_period"))
            ema200 = params.get("ema200_period", self.config.get("ema200_period"))
            if ema30 >= ema100 or ema100 >= ema200:
                return False

            tp_mult = params.get("tp_multiplier", self.config.get("tp_multiplier"))
            sl_mult = params.get("sl_multiplier", self.config.get("sl_multiplier"))
            if tp_mult < sl_mult:
                return False

            base_prob = params.get(
                "base_probability_threshold",
                self.config.get("base_probability_threshold"),
            )
            if not (0.1 <= base_prob <= 0.9):
                return False

            if not (
                2
                <= params.get(
                    "loss_streak_threshold", self.config.get("loss_streak_threshold", 2)
                )
                <= 5
            ):
                return False

            if not (
                2
                <= params.get(
                    "win_streak_threshold", self.config.get("win_streak_threshold", 2)
                )
                <= 5
            ):
                return False

            if not (
                0.01
                <= params.get(
                    "threshold_adjustment",
                    self.config.get("threshold_adjustment", 0.05),
                )
                <= 0.1
            ):
                return False

            if not (
                0.1
                <= params.get(
                    "risk_sharpe_loss_factor",
                    self.config.get("risk_sharpe_loss_factor", 0.5),
                )
                <= 1.0
            ):
                return False

            if not (
                1.0
                <= params.get(
                    "risk_sharpe_win_factor",
                    self.config.get("risk_sharpe_win_factor", 1.5),
                )
                <= 2.0
            ):
                return False

            vol_min = params.get(
                "risk_vol_min", self.config.get("risk_vol_min", 0.5)
            )
            vol_max = params.get(
                "risk_vol_max", self.config.get("risk_vol_max", 2.0)
            )
            if not (0.1 <= vol_min < vol_max <= 3.0):
                return False

            return True
        except (KeyError, ValueError, TypeError) as e:
            logger.exception("Ошибка валидации параметров: %s", e)
            raise

    def _evaluate_params(self, params: dict, symbol: str, df: pd.DataFrame):
        """Helper to evaluate a parameter set using the remote objective."""
        if hasattr(_objective_remote, "options"):
            obj_fn = _objective_remote.options(
                num_gpus=1 if is_cuda_available() else 0
            ).remote
        else:
            obj_fn = getattr(_objective_remote, "remote", _objective_remote)
        return obj_fn(
            df,
            symbol,
            params["ema30_period"],
            params["ema100_period"],
            params["ema200_period"],
            params["atr_period_default"],
            self.config["timeframe"],
            self.n_splits,
        )

    def objective(self, trial, symbol, df):
        # Целевая функция для оптимизации
        try:
            ema_periods = [
                trial.suggest_int("ema30_period", 10, 50),
                trial.suggest_int("ema100_period", 50, 200),
                trial.suggest_int("ema200_period", 100, 300),
            ]
            ema_periods.sort()
            ema30_period, ema100_period, ema200_period = ema_periods
            atr_period_default = trial.suggest_int("atr_period_default", 5, 20)
            trial.suggest_int("loss_streak_threshold", 2, 5)
            trial.suggest_int("win_streak_threshold", 2, 5)
            trial.suggest_float("threshold_adjustment", 0.01, 0.1)
            trial.suggest_float("risk_sharpe_loss_factor", 0.1, 1.0)
            trial.suggest_float("risk_sharpe_win_factor", 1.0, 2.0)
            trial.suggest_float("risk_vol_min", 0.1, 1.0)
            trial.suggest_float("risk_vol_max", 1.0, 3.0)
            # Stop loss and take profit multipliers are now taken
            # directly from the configuration and are not optimized.
            if hasattr(_objective_remote, "options"):
                obj_fn = _objective_remote.options(
                    num_gpus=1 if is_cuda_available() else 0
                ).remote
            else:
                obj_fn = getattr(_objective_remote, "remote", _objective_remote)
            return obj_fn(
                df,
                symbol,
                ema30_period,
                ema100_period,
                ema200_period,
                atr_period_default,
                self.config["timeframe"],
                self.n_splits,
            )
        except (ValueError, RuntimeError, KeyError) as e:
            logger.exception("Ошибка в objective для %s: %s", symbol, e)
            raise

    def _grid_search(self, df, symbol, base_params):
        """Refine best parameters using GridSearchCV for reliability."""

        class _Estimator(BaseEstimator):
            def __init__(self, df, symbol, timeframe, n_splits):
                self.df = df
                self.symbol = symbol
                self.timeframe = timeframe
                self.n_splits = n_splits

            def fit(self, X=None, y=None):
                if hasattr(_objective_remote, "options"):
                    obj_fn = _objective_remote.options(
                        num_gpus=1 if is_cuda_available() else 0
                    ).remote
                else:
                    obj_fn = getattr(_objective_remote, "remote", _objective_remote)
                logger.debug(
                    "Dispatching _objective_remote for grid search %s", self.symbol
                )
                self.score_ = ray.get(
                    obj_fn(
                        self.df,
                        self.symbol,
                        self.ema30_period,
                        self.ema100_period,
                        self.ema200_period,
                        self.atr_period_default,
                        self.timeframe,
                        self.n_splits,
                    )
                )
                logger.debug("Received grid search result for %s", self.symbol)
                return self

            def score(self, X=None, y=None):
                return self.score_

        estimator = _Estimator(df, symbol, self.config["timeframe"], self.n_splits)
        param_grid = {
            "ema30_period": [
                max(10, base_params["ema30_period"] - 5),
                base_params["ema30_period"],
                min(50, base_params["ema30_period"] + 5),
            ],
            "ema100_period": [
                max(50, base_params["ema100_period"] - 20),
                base_params["ema100_period"],
                min(200, base_params["ema100_period"] + 20),
            ],
            "ema200_period": [
                max(100, base_params["ema200_period"] - 20),
                base_params["ema200_period"],
                min(300, base_params["ema200_period"] + 20),
            ],
            "atr_period_default": [
                max(5, base_params["atr_period_default"] - 2),
                base_params["atr_period_default"],
                min(20, base_params["atr_period_default"] + 2),
            ],
        }
        gs = GridSearchCV(estimator, param_grid, cv=[(slice(None), slice(None))])
        dummy_X = np.zeros((1, 1))
        dummy_y = np.zeros(1)
        gs.fit(dummy_X, dummy_y)
        return {**base_params, **gs.best_params_}

    async def optimize_all(self):
        # Оптимизация для всех символов
        try:
            tasks = [self.optimize(symbol) for symbol in self.data_handler.usdt_pairs]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for symbol, result in zip(self.data_handler.usdt_pairs, results):
                if not isinstance(result, Exception):
                    logger.info("Оптимизация завершена для %s", symbol)
                else:
                    logger.error("Ошибка оптимизации для %s: %s", symbol, result)
            return self.best_params_by_symbol
        except (RuntimeError, ValueError) as e:
            logger.exception("Ошибка в optimize_all: %s", e)
            raise
