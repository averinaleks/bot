"""Portfolio-level hyperparameter optimization."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import optuna
import ray

from utils import logger
from config import BotConfig


@ray.remote
def _portfolio_backtest_remote(
    df_dict: dict[str, pd.DataFrame],
    params: dict,
    timeframe: str,
    metric: str = "sharpe",
    n_splits: int = 5,
) -> float:
    """Evaluate parameters on the whole portfolio."""
    try:
        ratios = []
        for df in df_dict.values():
            if df is None or df.empty:
                continue
            train_size = int(0.6 * len(df))
            test_size = int(0.2 * len(df))
            for i in range(n_splits):
                start = i * test_size
                end = start + train_size + test_size
                if end > len(df):
                    break
                test_df = df.iloc[start + train_size : end].droplevel("symbol")
                if len(test_df) < 14:
                    continue
                ema_fast = test_df["close"].ewm(
                    span=params["ema30_period"], adjust=False
                ).mean()
                ema_slow = test_df["close"].ewm(
                    span=params["ema100_period"], adjust=False
                ).mean()
                returns = []
                for j in range(1, len(test_df)):
                    signal = 1 if ema_fast.iloc[j] > ema_slow.iloc[j] else -1
                    ret = (
                        test_df["close"].iloc[j] - test_df["close"].iloc[j - 1]
                    ) / test_df["close"].iloc[j - 1]
                    returns.append(ret * signal)
                if not returns:
                    continue
                returns_np = np.array(returns, dtype=np.float32)
                if metric == "sortino":
                    neg = returns_np[returns_np < 0]
                    denom = np.std(neg) + 1e-6
                else:
                    denom = np.std(returns_np) + 1e-6
                ratio = (
                    np.mean(returns_np)
                    / denom
                    * np.sqrt(
                        365 * 24 * 60
                        / pd.Timedelta(timeframe).total_seconds()
                    )
                )
                if np.isfinite(ratio):
                    ratios.append(ratio)
        return float(np.mean(ratios)) if ratios else 0.0
    except Exception as e:  # pragma: no cover - log
        logger.exception("Error in _portfolio_backtest_remote: %s", e)
        raise


class StrategyOptimizer:
    """Optimize parameters jointly for the whole portfolio."""

    def __init__(self, config: BotConfig, data_handler):
        self.config = config
        self.data_handler = data_handler
        self.max_trials = config.get("optuna_trials", 20)
        self.n_splits = config.get("n_splits", 5)
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
                df_dict, params, self.config["timeframe"], self.metric, self.n_splits
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
