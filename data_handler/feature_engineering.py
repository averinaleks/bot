"""Вспомогательные функции для построения признаков.

Модуль предоставляет лёгкую реализацию расчёта индикаторов и целевой
переменной, чтобы использоваться в тестах.  Основная идея – исключить
look-ahead: все индикаторы рассчитываются только на основе прошлых
значений, а таргет определяется как знак доходности на горизонте ``t+h``
с лагом ``1``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import types
from typing import Any

try:  # pragma: no cover - optional dependency
    import ta  # type: ignore
except Exception:  # pragma: no cover - ta not installed
    ta = None  # type: ignore


def compute_indicators(df: pd.DataFrame, cfg: Any) -> types.SimpleNamespace:
    """Рассчитать набор индикаторов для ``df``.

    Если установлен пакет ``ta``, используется его реализация, иначе
    применяется упрощённый расчёт на ``pandas``.
    """

    close = df["close"]
    ema30 = close.ewm(span=getattr(cfg, "ema30_period", 30), adjust=False).mean()
    ema100 = close.ewm(span=getattr(cfg, "ema100_period", 100), adjust=False).mean()
    ema200 = close.ewm(span=getattr(cfg, "ema200_period", 200), adjust=False).mean()

    if ta is not None:  # pragma: no branch - import guard
        rsi = ta.momentum.RSIIndicator(
            close, window=getattr(cfg, "rsi_window", 14)
        ).rsi()
        adx = ta.trend.ADXIndicator(
            df["high"], df["low"], close, window=getattr(cfg, "adx_window", 14)
        ).adx()
        macd = ta.trend.MACD(
            close,
            window_slow=getattr(cfg, "macd_window_slow", 26),
            window_fast=getattr(cfg, "macd_window_fast", 12),
            window_sign=getattr(cfg, "macd_window_sign", 9),
        ).macd()
        atr = ta.volatility.AverageTrueRange(
            df["high"], df["low"], close, window=getattr(cfg, "atr_period_default", 14)
        ).average_true_range()
    else:
        window = getattr(cfg, "rsi_window", 14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        adx_window = getattr(cfg, "adx_window", 14)
        plus_dm = df["high"].diff().clip(lower=0)
        minus_dm = (-df["low"].diff()).clip(lower=0)
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - close.shift()).abs(),
                (df["low"] - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=getattr(cfg, "atr_period_default", 14), min_periods=1).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / adx_window, min_periods=adx_window).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1 / adx_window, min_periods=adx_window).mean() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.ewm(alpha=1 / adx_window, min_periods=adx_window).mean()

        fast = close.ewm(
            span=getattr(cfg, "macd_window_fast", 12), adjust=False
        ).mean()
        slow = close.ewm(
            span=getattr(cfg, "macd_window_slow", 26), adjust=False
        ).mean()
        macd = fast - slow

    return types.SimpleNamespace(
        ema30=ema30,
        ema100=ema100,
        ema200=ema200,
        rsi=rsi,
        adx=adx,
        macd=macd,
        atr=atr,
    )


def compute_features(df: pd.DataFrame, h: int, lag: int = 1, ema_span: int | None = None) -> pd.DataFrame:
    """Добавить индикаторы и таргет к исходному ``DataFrame``.

    Parameters
    ----------
    df: pd.DataFrame
        Таблица котировок, должна содержать колонку ``close``.
    h: int
        Горизонт прогнозирования, на который вычисляется доходность.
    lag: int, default=1
        Лаг, используемый для смещения индикаторов и таргета так, чтобы они
        опирались только на известные в прошлом данные.
    ema_span: int | None
        Период EMA. Если ``None``, EMA не рассчитывается.

    Returns
    -------
    pd.DataFrame
        Копия входного DataFrame с добавленным таргетом и, при необходимости,
        индикатором EMA. Первые ``h + lag`` строк с ``NaN`` удаляются, так как
        они не содержат полноценную информацию.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    result = df.copy()

    # EMA рассчитывается только на прошлых данных
    if ema_span is not None:
        ema_raw = result["close"].ewm(span=ema_span, adjust=False).mean()
        ema = ema_raw.shift(lag)
        # Проверяем, что индикатор действительно использует только прошлые значения
        if not ema.shift(-lag).equals(ema_raw):  # pragma: no cover - защитная проверка
            raise ValueError("EMA indicator uses future values")
        result["ema"] = ema

    # Доходность на горизонте t+h и её знак
    forward_return = result["close"].pct_change(periods=h).shift(-h)
    target = np.sign(forward_return).shift(lag)
    result["target"] = target

    # Удаляем первые h+lag строк с NaN
    result = result.iloc[h + lag :].copy()
    # Удаляем оставшиеся NaN (например, из-за сдвига в конец)
    result = result.dropna()

    return result


__all__ = ["compute_features", "compute_indicators"]
