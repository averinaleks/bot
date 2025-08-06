"""Portfolio backtesting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
try:  # pragma: no cover - optional dependency
    from numba import njit
except Exception:  # pragma: no cover - fallback if numba unavailable
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from bot.utils import logger


@njit(cache=True)
def _simulate_trades(
    symbol_ids: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    atr: np.ndarray,
    probability: np.ndarray,
    base_thr: float,
    sl_mult: float,
    tp_mult: float,
    max_positions: int,
    n_symbols: int,
):
    """Vectorized trade simulation using ``numba``.

    This function iterates over pre-converted ``numpy`` arrays but executes in
    compiled mode, eliminating Python-loop overhead.
    """

    positions_symbol = np.full(max_positions, -1, dtype=np.int32)
    positions_entry = np.zeros(max_positions, dtype=np.float32)
    positions_side = np.zeros(max_positions, dtype=np.int8)  # 1=buy, -1=sell
    positions_sl = np.zeros(max_positions, dtype=np.float32)
    positions_tp = np.zeros(max_positions, dtype=np.float32)
    last_close = np.zeros(n_symbols, dtype=np.float32)

    n = close.shape[0]
    returns = np.empty(n, dtype=np.float32)
    count = 0

    for i in range(n):
        sym = symbol_ids[i]
        price = close[i]
        hi = high[i]
        lo = low[i]
        a = atr[i]
        prob = probability[i]
        last_close[sym] = price

        pos_idx = -1
        for j in range(max_positions):
            if positions_symbol[j] == sym:
                pos_idx = j
                break

        if pos_idx != -1:
            side = positions_side[pos_idx]
            if side == 1:
                if hi >= positions_tp[pos_idx]:
                    returns[count] = (
                        (positions_tp[pos_idx] - positions_entry[pos_idx])
                        / positions_entry[pos_idx]
                    )
                    count += 1
                    positions_symbol[pos_idx] = -1
                elif lo <= positions_sl[pos_idx]:
                    returns[count] = (
                        (positions_sl[pos_idx] - positions_entry[pos_idx])
                        / positions_entry[pos_idx]
                    )
                    count += 1
                    positions_symbol[pos_idx] = -1
            else:
                if lo <= positions_tp[pos_idx]:
                    returns[count] = (
                        (positions_entry[pos_idx] - positions_tp[pos_idx])
                        / positions_entry[pos_idx]
                    )
                    count += 1
                    positions_symbol[pos_idx] = -1
                elif hi >= positions_sl[pos_idx]:
                    returns[count] = (
                        (positions_entry[pos_idx] - positions_sl[pos_idx])
                        / positions_entry[pos_idx]
                    )
                    count += 1
                    positions_symbol[pos_idx] = -1

        if pos_idx == -1:
            # count current open positions
            open_count = 0
            for j in range(max_positions):
                if positions_symbol[j] != -1:
                    open_count += 1
            if open_count < max_positions and not np.isnan(a):
                signal = 0
                if ema_fast[i] > ema_slow[i] and prob >= base_thr:
                    signal = 1
                elif ema_fast[i] < ema_slow[i] and (1 - prob) >= base_thr:
                    signal = -1
                if signal != 0:
                    # find empty slot
                    free_idx = -1
                    for j in range(max_positions):
                        if positions_symbol[j] == -1:
                            free_idx = j
                            break
                    positions_symbol[free_idx] = sym
                    positions_entry[free_idx] = price
                    positions_side[free_idx] = signal
                    if signal == 1:
                        positions_sl[free_idx] = price - sl_mult * a
                        positions_tp[free_idx] = price + tp_mult * a
                    else:
                        positions_sl[free_idx] = price + sl_mult * a
                        positions_tp[free_idx] = price - tp_mult * a

    # Close any remaining open positions at last seen price
    for j in range(max_positions):
        sym = positions_symbol[j]
        if sym != -1:
            entry = positions_entry[j]
            side = positions_side[j]
            last = last_close[sym]
            if side == 1:
                returns[count] = (last - entry) / entry
            else:
                returns[count] = (entry - last) / entry
            count += 1

    return returns[:count]


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

        base_thr = params.get("base_probability_threshold", 0.6)
        sl_mult = params.get("sl_multiplier", 1.0)
        tp_mult = params.get("tp_multiplier", 2.0)

        # Convert columns to numpy arrays for numba function
        symbol_codes, uniques = pd.factorize(combined["symbol"])
        returns_np = _simulate_trades(
            symbol_codes.astype(np.int32),
            combined["close"].to_numpy(np.float32),
            combined["high"].to_numpy(np.float32),
            combined["low"].to_numpy(np.float32),
            combined["ema_fast"].to_numpy(np.float32),
            combined["ema_slow"].to_numpy(np.float32),
            combined["atr"].to_numpy(np.float32),
            combined["probability"].to_numpy(np.float32),
            float(base_thr),
            float(sl_mult),
            float(tp_mult),
            int(max_positions),
            int(len(uniques)),
        )

        if returns_np.size == 0:
            return 0.0
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

