import pandas as pd
from typing import Any, Iterable


class DataFrame:
    def __init__(self, data: Any | None = None) -> None:
        if isinstance(data, pd.DataFrame):
            self._df = data.copy()
        elif data is None:
            self._df = pd.DataFrame()
        else:
            self._df = pd.DataFrame(data)

    @property
    def height(self) -> int:
        return len(self._df)

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    def clone(self) -> "DataFrame":
        return DataFrame(self._df.copy())

    def filter(self, expr) -> "DataFrame":
        if callable(expr):
            mask = expr(self._df)
        else:
            mask = expr
        return DataFrame(self._df[mask])

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()

    def __getitem__(self, item):
        return self._df[item]

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._df, name)


def from_dicts(rows: Iterable[dict]) -> DataFrame:
    return DataFrame(list(rows))


def col(name: str):
    class _Expr:
        def __init__(self, column: str) -> None:
            self.column = column

        def _cmp(self, other, op):
            def _inner(df: pd.DataFrame):
                return op(df[self.column], other)
            return _inner

        def __eq__(self, other):  # type: ignore[override]
            return self._cmp(other, lambda a, b: a == b)

        def __ge__(self, other):
            return self._cmp(other, lambda a, b: a >= b)

        def __le__(self, other):
            return self._cmp(other, lambda a, b: a <= b)

    return _Expr(name)


__all__ = ["DataFrame", "from_dicts", "col"]
