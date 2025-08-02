"""Portfolio-level hyperparameter optimization."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import optuna
import os

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

        study = optuna.create_study(direction="maximize")
        obj_refs = []
        trials = []
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
            obj_ref = _portfolio_backtest_remote.remote(
                df_dict,
                params,
                self.config["timeframe"],
                self.metric,
                self.config.get("max_positions", 5),
            )
            obj_refs.append(obj_ref)
            trials.append((trial, params))

        results = await asyncio.to_thread(ray.get, obj_refs)
        for (trial, _), value in zip(trials, results):
            study.tell(trial, value)

        best_params = {**self.config.asdict(), **study.best_params}
        return best_params

    async def optimize_all(self) -> dict:
        """Convenience wrapper for compatibility."""
        return await self.optimize()
