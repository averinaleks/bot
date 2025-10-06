from __future__ import annotations

import math
from typing import Any, Literal

from bot.pydantic_compat import BaseModel, ConfigDict, Field

try:  # pragma: no cover - exercised when Pydantic v2 is available
    from pydantic import field_validator
except Exception:  # pragma: no cover - fallback for environments without Pydantic v2
    try:  # pragma: no cover - support for Pydantic v1
        from pydantic import validator as _legacy_validator  # type: ignore
    except Exception:  # pragma: no cover - executed when using the offline stub
        def field_validator(*_names: str, **_kwargs: Any):
            def decorator(func):
                return func

            return decorator
    else:
        def field_validator(*names: str, mode: str | None = None, **kwargs: Any):
            pre = mode == "before"
            kwargs.pop("mode", None)
            return _legacy_validator(*names, pre=pre, **kwargs)  # type: ignore


class OpenPositionRequest(BaseModel):
    """Request payload for the ``/open_position`` endpoint."""

    symbol: str
    side: Literal["buy", "sell"] = "buy"
    price: float | None = Field(default=None)
    amount: float | None = Field(default=None)
    tp: float | None = Field(default=None)
    sl: float | None = Field(default=None)
    trailing_stop: float | None = Field(default=None)

    model_config = ConfigDict(extra="forbid")

    @field_validator("symbol", mode="before")
    @classmethod
    def _validate_symbol(cls, value: Any) -> str:
        if isinstance(value, str):
            result = value.strip()
        else:
            result = str(value).strip() if value is not None else ""
        if not result:
            raise ValueError("symbol must be a non-empty string")
        return result

    @field_validator("side", mode="before")
    @classmethod
    def _validate_side(cls, value: Any) -> str:
        if value is None:
            return "buy"
        text = value.lower() if isinstance(value, str) else str(value).lower()
        if text not in {"buy", "sell"}:
            raise ValueError("side must be either 'buy' or 'sell'")
        return text

    @field_validator("price", "amount", "tp", "sl", "trailing_stop", mode="before")
    @classmethod
    def _validate_float(cls, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise ValueError("value must be a finite number")
            return float(value)
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("value must be a number") from exc
        if math.isnan(parsed) or math.isinf(parsed):
            raise ValueError("value must be a finite number")
        return parsed


__all__ = ["OpenPositionRequest"]
