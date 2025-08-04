"""Portfolio backtesting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot.utils import logger


def portfolio_backtest(
    df_dict: dict[str, pd.DataFrame],
    params: dict,
    timeframe: str,
    metric: str = "sharpe",
    max_positions: int = 5,
) -> float:
    """Simulate trading multiple symbols at once.

    Parameters
    ----------
    df_dict : dict[str, pd.DataFrame]
        Historical OHLCV data per symbol indexed by ``symbol`` and ``timestamp``.
    params : dict
        Strategy parameters such as EMA periods and TP/SL multipliers.
    timeframe : str
        Candle interval (e.g. ``"1m"``) used for annualization of metrics.
    metric : str, optional
        ``"sharpe"`` or ``"sortino``" to compute the fitness metric.
    max_positions : int, optional
        Maximum number of simultaneous open positions.
    """
    try:
        events = []
        for symbol, df in df_dict.items():
            if df is None or df.empty:
                continue
            if "symbol" in df.index.names:
                df_reset = df.reset_index()
                if "level_1" in df_reset.columns:
                    df_reset = df_reset.rename(columns={"level_1": "timestamp"})
            else:
                df_reset = df.copy()
                df_reset["timestamp"] = df_reset.index
                df_reset["symbol"] = symbol
            df_reset = df_reset.sort_values("timestamp")
            ema_fast = df_reset["close"].ewm(
                span=params.get("ema30_period", 30), adjust=False
            ).mean()
            ema_slow = df_reset["close"].ewm(
                span=params.get("ema100_period", 100), adjust=False
            ).mean()
            tr1 = df_reset["high"] - df_reset["low"]
            tr2 = (df_reset["high"] - df_reset["close"].shift()).abs()
            tr3 = (df_reset["low"] - df_reset["close"].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(params.get("atr_period", 14)).mean()
            df_reset = df_reset.assign(
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                atr=atr,
            )
            if "probability" not in df_reset.columns:
                df_reset["probability"] = 1.0
            events.append(
                df_reset[
                    [
                        "timestamp",
                        "symbol",
                        "close",
                        "high",
                        "low",
                        "probability",
                        "ema_fast",
                        "ema_slow",
                        "atr",
                    ]
                ]
            )
        if not events:
            return 0.0
        combined = pd.concat(events).sort_values("timestamp").reset_index(drop=True)

        positions: dict[str, dict] = {}
        returns: list[float] = []
        base_thr = params.get("base_probability_threshold", 0.6)
        sl_mult = params.get("sl_multiplier", 1.0)
        tp_mult = params.get("tp_multiplier", 2.0)

        for _, row in combined.iterrows():
            symbol = row["symbol"]
            price = row["close"]
            high = row["high"]
            low = row["low"]
            atr = row["atr"]
            probability = row.get("probability", 1.0)

            pos = positions.get(symbol)
            if pos is not None:
                if pos["side"] == "buy":
                    if high >= pos["tp"]:
                        returns.append((pos["tp"] - pos["entry"]) / pos["entry"])
                        del positions[symbol]
                    elif low <= pos["sl"]:
                        returns.append((pos["sl"] - pos["entry"]) / pos["entry"])
                        del positions[symbol]
                else:
                    if low <= pos["tp"]:
                        returns.append((pos["entry"] - pos["tp"]) / pos["entry"])
                        del positions[symbol]
                    elif high >= pos["sl"]:
                        returns.append((pos["entry"] - pos["sl"]) / pos["entry"])
                        del positions[symbol]

            if symbol not in positions and len(positions) < max_positions and pd.notna(atr):
                signal = None
                if row["ema_fast"] > row["ema_slow"] and probability >= base_thr:
                    signal = "buy"
                elif row["ema_fast"] < row["ema_slow"] and (1 - probability) >= base_thr:
                    signal = "sell"
                if signal:
                    if signal == "buy":
                        sl = price - sl_mult * atr
                        tp = price + tp_mult * atr
                    else:
                        sl = price + sl_mult * atr
                        tp = price - tp_mult * atr
                    positions[symbol] = {
                        "entry": price,
                        "side": signal,
                        "sl": sl,
                        "tp": tp,
                    }

        for symbol, pos in positions.items():
            last_close = combined[combined["symbol"] == symbol]["close"].iloc[-1]
            if pos["side"] == "buy":
                returns.append((last_close - pos["entry"]) / pos["entry"])
            else:
                returns.append((pos["entry"] - last_close) / pos["entry"])

        if not returns:
            return 0.0
        returns_np = np.array(returns, dtype=np.float32)
        if metric == "sortino":
            neg = returns_np[returns_np < 0]
            denom = np.std(neg) + 1e-6
        else:
            denom = np.std(returns_np) + 1e-6
        ratio = (
            np.mean(returns_np)
            / denom
            * np.sqrt(365 * 24 * 60 / pd.Timedelta(timeframe).total_seconds())
        )
        return float(ratio) if np.isfinite(ratio) else 0.0
    except Exception as e:  # pragma: no cover - log
        logger.exception("Error in portfolio_backtest: %s", e)
        raise

