"""Lightweight stub of the :mod:`polars` package used for tests.

The real Polars package requires CPU instructions that are unavailable in the
execution environment used for the kata, which leads to crashes when importing
it.  The tests exercise only a tiny portion of Polars' surface area, so this
module provides a minimal, pandas-backed implementation that mimics the methods
used in the project.  It is intentionally small but aims to behave close enough
for the unit tests.
"""
from __future__ import annotations

import datetime as _dt
import operator
from typing import Iterable, List

import pandas as _pd


class Datetime:
    """Simplified replacement for :class:`polars.Datetime`."""

    def __init__(self, time_zone: str | None = "UTC") -> None:
        self.time_zone = time_zone or "UTC"

    def convert(self, value: object) -> object:
        if isinstance(value, _pd.Timestamp):
            if value.tzinfo is None:
                value = value.tz_localize(self.time_zone)
            else:
                value = value.tz_convert(self.time_zone)
            return value.to_pydatetime()
        if isinstance(value, _dt.datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=_dt.timezone.utc)
            return value.astimezone(_dt.timezone.utc)
        return value


class Series:
    """Very small stand-in for :class:`polars.Series`."""

    def __init__(self, name: str, values: Iterable[object], dtype: Datetime | None = None) -> None:
        self.name = name
        if dtype is not None:
            converter = dtype.convert
            self._values = [converter(v) for v in values]
        else:
            self._values = list(values)

    def to_list(self) -> List[object]:
        return list(self._values)


class _ColumnExpression:
    def __init__(self, name: str, transform):
        self.name = name
        self._transform = transform

    def map_elements(self, func):
        def _wrapped(values):
            return [func(v) for v in self._transform(values)]

        return _ColumnExpression(self.name, _wrapped)

    def cast(self, dtype: Datetime, strict: bool = False):  # noqa: ARG002 - strict kept for API parity
        def _wrapped(values):
            return [dtype.convert(v) for v in self._transform(values)]

        return _ColumnExpression(self.name, _wrapped)

    def evaluate(self, frame: "DataFrame") -> List[object]:
        if self.name not in frame._df.columns:
            return []
        base = list(frame._df[self.name])
        return self._transform(base)

    def __ge__(self, other):
        return _BooleanExpression(self, operator.ge, other)


class _BooleanExpression:
    def __init__(self, column: _ColumnExpression, op, other):
        self.column = column
        self.op = op
        self.other = other

    def evaluate(self, frame: "DataFrame") -> List[bool]:
        values = self.column.evaluate(frame)
        return [self.op(value, self.other) for value in values]


def col(name: str) -> _ColumnExpression:
    return _ColumnExpression(name, lambda values: list(values))


class DataFrame:
    """Minimal pandas-backed DataFrame used in tests."""

    def __init__(self, data: object | None = None) -> None:
        if isinstance(data, DataFrame):
            self._df = data._df.copy()
        elif isinstance(data, _pd.DataFrame):
            self._df = data.copy()
        elif data is None:
            self._df = _pd.DataFrame()
        elif isinstance(data, dict):
            converted = {}
            for key, value in data.items():
                if isinstance(value, Series):
                    converted[key] = value.to_list()
                else:
                    converted[key] = list(value)
            self._df = _pd.DataFrame(converted)
        else:
            self._df = _pd.DataFrame(data)

    @property
    def height(self) -> int:
        return len(self._df)

    @property
    def columns(self) -> List[str]:
        return list(self._df.columns)

    def clone(self) -> "DataFrame":
        return DataFrame(self._df.copy())

    def with_columns(self, *exprs):
        if len(exprs) == 1 and isinstance(exprs[0], (list, tuple)):
            exprs = tuple(exprs[0])
        result = self._df.copy()
        for expr in exprs:
            if isinstance(expr, Series):
                result[expr.name] = expr.to_list()
            elif isinstance(expr, _ColumnExpression):
                result[expr.name] = expr.evaluate(self)
            elif isinstance(expr, tuple) and len(expr) == 2:
                name, values = expr
                result[name] = list(values)
            else:
                raise TypeError(f"Unsupported expression type: {type(expr)!r}")
        return DataFrame(result)

    def filter(self, predicate):
        if isinstance(predicate, _BooleanExpression):
            mask = predicate.evaluate(self)
        else:
            raise TypeError("filter expects a boolean expression")
        mask_series = _pd.Series(mask, index=self._df.index)
        filtered = self._df[mask_series].reset_index(drop=True)
        return DataFrame(filtered)

    def __getitem__(self, item):
        return self._df[item]

    def __iter__(self):
        return iter(self._df.columns)

    def __repr__(self) -> str:  # pragma: no cover - for debugging only
        return f"DataFrame({self._df!r})"


__all__ = [
    "DataFrame",
    "Series",
    "Datetime",
    "col",
]
