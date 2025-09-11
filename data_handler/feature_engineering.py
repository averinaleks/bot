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


__all__ = ["compute_features"]
