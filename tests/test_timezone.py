import importlib
import sys

import pandas as pd
import pandas.testing as pdt


def test_ensure_utc_converts_series_to_utc(monkeypatch):
    """ensure_utc должен конвертировать серию с локальным временем в UTC."""
    monkeypatch.setenv("TEST_MODE", "1")
    sys.modules.pop("data_handler", None)
    sys.modules.pop("data_handler.utils", None)
    ensure_utc = importlib.import_module("data_handler.utils").ensure_utc
    local_series = pd.Series([pd.Timestamp("2024-01-01 12:30:00")])

    utc_series = ensure_utc(local_series)

    expected = pd.Series(
        [pd.Timestamp("2024-01-01 12:30:00", tz="UTC")]
    )
    pdt.assert_series_equal(utc_series, expected)


def test_ensure_utc_converts_index_to_utc(monkeypatch):
    """ensure_utc должен корректно преобразовывать индекс в UTC."""
    monkeypatch.setenv("TEST_MODE", "1")
    sys.modules.pop("data_handler", None)
    sys.modules.pop("data_handler.utils", None)
    ensure_utc = importlib.import_module("data_handler.utils").ensure_utc
    local_index = pd.Index([pd.Timestamp("2024-02-15 08:00:00")])

    utc_index = ensure_utc(local_index)

    expected = pd.DatetimeIndex([pd.Timestamp("2024-02-15 08:00:00", tz="UTC")])
    pdt.assert_index_equal(utc_index, expected)
