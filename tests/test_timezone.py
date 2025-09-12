import pandas as pd


def test_local_time_converts_to_utc():
    """Локальное время должно корректно преобразовываться в UTC."""
    local_time = pd.Timestamp.now()
    utc_time = local_time.tz_localize("UTC").tz_convert("UTC")
    assert utc_time.tzname() == "UTC"
    assert utc_time.utcoffset() == pd.Timedelta(0)
