"""Portfolio-level hyperparameter optimization."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import os
from itertools import product

try:
    import polars as pl  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pl = None  # type: ignore

try:
    import optuna  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore

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

from bot.utils import logger
from bot.config import BotConfig
from bot.portfolio_backtest import portfolio_backtest


def walk_forward_splits(n: int, n_splits: int, min_train: int, horizon: int):
    """Generate indices for walk-forward validation.

    Parameters
    ----------
    n: int
        Total number of samples.
    n_splits: int
        Number of walk-forward splits.
    min_train: int
        Minimum size of the initial training window.
    horizon: int
        Size of the test window.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        Arrays of train and test indices for each split.
    """
    if min_train <= 0 or horizon <= 0:
        raise ValueError("min_train and horizon must be positive")
    if min_train + horizon > n:
        raise ValueError("Not enough data for the first split")

    step = (n - min_train - horizon) // max(1, n_splits - 1)
    step = max(1, step)

    for i in range(n_splits):
        train_end = min_train + step * i
        test_start = train_end
        test_end = test_start + horizon
        if test_end > n:
            break
        train_idx = np.arange(train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx


@ray.remote
def _portfolio_backtest_remote(
    df_dict: dict[str, pd.DataFrame],
    params: dict,
    timeframe: str,
    metric: str = "sharpe",
    max_positions: int = 5,
) -> float:
    """Evaluate parameters on the whole portfolio."""
    try:
        return portfolio_backtest(
            df_dict, params, timeframe, metric=metric, max_positions=max_positions
        )
    except Exception as e:  # pragma: no cover - log
        logger.exception("Error in _portfolio_backtest_remote: %s", e)
        raise


class StrategyOptimizer:
    """Optimize parameters jointly for the whole portfolio."""

    def __init__(self, config: BotConfig, data_handler):
        self.config = config
        self.data_handler = data_handler
        self.max_trials = config.get("optuna_trials", 20)
        self.n_splits = config.get("n_splits", 3)
        self.metric = config.get("portfolio_metric", "sharpe")
        self.min_train = config.get("wf_min_train", 100)
        self.horizon = config.get("wf_horizon", 50)

    async def optimize(self) -> dict:
        """Optimize parameters for the portfolio."""
        df_dict: dict[str, pd.DataFrame] = {}
        ohlcv = self.data_handler.ohlcv
        for symbol in self.data_handler.usdt_pairs:
            if (
                "symbol" in ohlcv.index.names
                and symbol in ohlcv.index.get_level_values("symbol")
            ):
                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
                if not df.empty:
                    df_dict[symbol] = df
        if not df_dict:
            logger.warning("Нет данных для оптимизации стратегии")
            return self.config.asdict()

        n = min(len(df) for df in df_dict.values())
        min_train = max(1, min(self.min_train, n - 1))
        horizon = max(1, min(self.horizon, n - min_train))

        async def evaluate_params(params: dict) -> float:
            metrics: list[float] = []
            for i, (_train_idx, test_idx) in enumerate(
                walk_forward_splits(n, self.n_splits, min_train, horizon)
            ):
                test_df_dict = {
                    symbol: df.iloc[test_idx]
                    for symbol, df in df_dict.items()
                }
                result = _portfolio_backtest_remote.remote(
                    test_df_dict,
                    params,
                    self.config["timeframe"],
                    self.metric,
                    self.config.get("max_positions", 5),
                )
                if ray.is_initialized():
                    value = await asyncio.to_thread(ray.get, result)
                else:
                    value = result
                logger.info(
                    "WF split %d: metric=%.4f params=%s", i + 1, value, params
                )
                metrics.append(value)
            return float(np.mean(metrics)) if metrics else float("-inf")

        if optuna is None:
            logger.warning(
                "optuna не установлен, используется упрощённый перебор параметров"
            )
            param_grid = {
                "ema30_period": [10, 30],
                "ema100_period": [50, 100],
                "ema200_period": [200, 300],
            }
            best_value = float("-inf")
            best_params = self.config.asdict()
            for ema30, ema100, ema200 in product(
                param_grid["ema30_period"],
                param_grid["ema100_period"],
                param_grid["ema200_period"],
            ):
                if not (ema30 < ema100 < ema200):
                    continue
                params = {
                    **self.config.asdict(),
                    "ema30_period": ema30,
                    "ema100_period": ema100,
                    "ema200_period": ema200,
                }
                value = await evaluate_params(params)
                if value > best_value:
                    best_value = value
                    best_params = params
            return best_params

        study = optuna.create_study(direction="maximize")
        for _ in range(self.max_trials):
            trial = study.ask()
            params = {
                "ema30_period": trial.suggest_int("ema30_period", 10, 50),
                "ema100_period": trial.suggest_int("ema100_period", 50, 200),
                "ema200_period": trial.suggest_int("ema200_period", 100, 300),
                "tp_multiplier": trial.suggest_float("tp_multiplier", 1.0, 3.0),
                "sl_multiplier": trial.suggest_float("sl_multiplier", 0.5, 2.0),
                "base_probability_threshold": trial.suggest_float(
                    "base_probability_threshold", 0.1, 0.9
                ),
                "risk_sharpe_loss_factor": trial.suggest_float(
                    "risk_sharpe_loss_factor", 0.1, 1.0
                ),
                "risk_sharpe_win_factor": trial.suggest_float(
                    "risk_sharpe_win_factor", 1.0, 2.0
                ),
                "risk_vol_min": trial.suggest_float("risk_vol_min", 0.1, 1.0),
                "risk_vol_max": trial.suggest_float("risk_vol_max", 1.0, 3.0),
            }
            value = await evaluate_params(params)
            study.tell(trial, value)

        best_params = {**self.config.asdict(), **study.best_params}
        return best_params

    async def optimize_all(self) -> dict:
        """Convenience wrapper for compatibility."""
        return await self.optimize()
