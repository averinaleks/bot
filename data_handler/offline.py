from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

_SYMBOL_PERSONALISATION = b"botdho1"
_BLAKE2S_PERSON_SIZE = getattr(hashlib.blake2s, "PERSON_SIZE", 8)
if len(_SYMBOL_PERSONALISATION) > _BLAKE2S_PERSON_SIZE:
    raise ValueError(
        "Offline data handler personalisation exceeds blake2s PERSON_SIZE"
    )


@dataclass(frozen=True)
class Candle:
    """Immutable representation of a single OHLCV entry."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _rolling_average(values: Sequence[float], period: int) -> list[float]:
    if period <= 1:
        return [float(v) for v in values]
    result: list[float] = []
    window_sum = 0.0
    for index, value in enumerate(values):
        window_sum += value
        if index >= period:
            window_sum -= values[index - period]
        divisor = min(index + 1, period)
        result.append(window_sum / divisor)
    return result


def _true_range(high: Sequence[float], low: Sequence[float], close: Sequence[float]) -> list[float]:
    if not high or not low or not close:
        return []
    result: list[float] = [float(high[0] - low[0])]
    for index in range(1, len(close)):
        max_value = high[index]
        min_value = low[index]
        prev_close = close[index - 1]
        upper = max(max_value, prev_close)
        lower = min(min_value, prev_close)
        result.append(float(upper - lower))
    return result


def _average_true_range(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int,
) -> list[float]:
    tr = _true_range(high, low, close)
    if not tr:
        return []
    return _rolling_average(tr, max(1, period))


class OfflineDataHandler:
    """Deterministic, pandas-free data handler used in offline mode."""

    def __init__(
        self,
        cfg: Any | None = None,
        http_client: Any | None = None,
        optimizer: Any | None = None,
        exchange: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.http_client = http_client
        self.exchange = exchange
        self._optimizer = optimizer
        self.usdt_pairs = self._resolve_pairs(cfg)
        self.indicators: dict[str, SimpleNamespace] = {}
        self.indicators_2h: dict[str, SimpleNamespace] = {}
        self.ohlcv: dict[str, tuple[Candle, ...]] = {}
        self.ohlcv_2h: dict[str, tuple[Candle, ...]] = {}
        self.funding_rates: dict[str, float] = {}
        self.open_interest: dict[str, float] = {}
        self.parameter_optimizer = self._ensure_optimizer()
        self._last_refresh = 0.0
        self.refresh()

    @staticmethod
    def _resolve_pairs(cfg: Any | None) -> list[str]:
        candidates: Iterable[str] = ()
        if cfg is not None:
            symbols = getattr(cfg, "symbols", None)
            if isinstance(symbols, Iterable):
                candidates = (str(symbol).upper() for symbol in symbols)
        pairs = [symbol for symbol in candidates if symbol]
        if not pairs:
            pairs = ["BTCUSDT"]
        return list(dict.fromkeys(pairs))

    def _ensure_optimizer(self) -> SimpleNamespace:
        if self._optimizer is not None:
            return SimpleNamespace(optimize=self._wrap_optimizer(self._optimizer))

        async def _return_config(symbol: str) -> dict[str, Any]:  # noqa: ARG001
            if self.cfg is None:
                return {}
            asdict = getattr(self.cfg, "asdict", None)
            if callable(asdict):
                try:
                    return dict(asdict())
                except Exception:  # pragma: no cover - defensive
                    pass
            return dict(vars(self.cfg))

        return SimpleNamespace(optimize=_return_config)

    @staticmethod
    def _wrap_optimizer(optimizer: Any):
        optimise = getattr(optimizer, "optimize", None)
        if optimise is None:
            return lambda *_a, **_k: {}

        async def _wrapper(symbol: str):
            result = optimise(symbol)
            if hasattr(result, "__await__"):
                return await result  # type: ignore[func-returns-value]
            return result

        return _wrapper

    def refresh(self) -> None:
        """Populate deterministic OHLC and indicator fixtures."""

        now_ms = int(time.time() * 1000)
        for symbol in self.usdt_pairs:
            candles_primary = self._build_series(symbol, now_ms, step_seconds=60, length=60)
            candles_secondary = self._build_series(symbol, now_ms, step_seconds=7_200, length=30)
            self.ohlcv[symbol] = candles_primary
            self.ohlcv_2h[symbol] = candles_secondary
            self.indicators[symbol] = self._build_indicators(candles_primary)
            self.indicators_2h[symbol] = self._build_indicators(candles_secondary)
            self.funding_rates[symbol] = 0.0
            self.open_interest[symbol] = 0.0
        self._last_refresh = time.time()

    @staticmethod
    def _symbol_seed(symbol: str) -> int:
        digest = hashlib.blake2s(
            symbol.encode("utf-8", "surrogatepass"),
            digest_size=8,
            person=_SYMBOL_PERSONALISATION,
        ).digest()
        return int.from_bytes(digest, "big") % 10_000 + 10_000

    def _build_series(
        self,
        symbol: str,
        now_ms: int,
        *,
        step_seconds: int,
        length: int,
    ) -> tuple[Candle, ...]:
        base_price = float(self._symbol_seed(symbol))
        step_ms = step_seconds * 1_000
        candles: list[Candle] = []
        for index in range(length):
            timestamp = now_ms - (length - index) * step_ms
            angle = index / max(1.0, step_seconds / 60)
            delta = math.sin(angle) * 0.01 * base_price
            open_price = base_price + delta
            close_price = base_price - delta
            high_price = max(open_price, close_price) * 1.01
            low_price = min(open_price, close_price) * 0.99
            volume = 1_000.0 + index * 5.0
            candles.append(
                Candle(
                    timestamp=timestamp,
                    open=float(open_price),
                    high=float(high_price),
                    low=float(low_price),
                    close=float(close_price),
                    volume=float(volume),
                )
            )
        return tuple(candles)

    def _build_indicators(self, candles: tuple[Candle, ...]) -> SimpleNamespace:
        closes = [candle.close for candle in candles]
        highs = [candle.high for candle in candles]
        lows = [candle.low for candle in candles]
        atr = _average_true_range(highs, lows, closes, period=14)
        ema_short = self._ema(closes, span=12)
        ema_long = self._ema(closes, span=26)
        macd = [short - long for short, long in zip(ema_short, ema_long)]
        return SimpleNamespace(
            df=candles,
            atr=tuple(atr),
            ema_short=tuple(ema_short),
            ema_long=tuple(ema_long),
            macd=tuple(macd),
        )

    @staticmethod
    def _ema(values: Sequence[float], span: int) -> list[float]:
        if not values:
            return []
        if span <= 1:
            return [float(v) for v in values]
        alpha = 2.0 / (span + 1.0)
        ema_values: list[float] = [float(values[0])]
        for value in values[1:]:
            ema_values.append((float(value) - ema_values[-1]) * alpha + ema_values[-1])
        return ema_values

    async def is_data_fresh(self, symbol: str, timeframe: str = "primary", max_delay: float = 60.0) -> bool:  # noqa: D401
        _ = (symbol, timeframe)
        return (time.time() - self._last_refresh) <= max_delay

    async def get_atr(self, symbol: str) -> float:
        indicator = self.indicators.get(symbol)
        if not indicator:
            return 0.0
        atr_values = getattr(indicator, "atr", ())
        if not atr_values:
            return 0.0
        return float(atr_values[-1])
