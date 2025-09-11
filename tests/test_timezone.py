import pandas as pd


def test_local_time_converts_to_utc():
    local_time = pd.Timestamp.now()
    utc_time = local_time.tz_localize('UTC').tz_convert('UTC')
    utc_now = pd.Timestamp.utcnow().tz_convert('UTC')
    # разница не должна превышать одну секунду
    assert abs((utc_time - utc_now).total_seconds()) < 1
    assert utc_time.tzname() == 'UTC'
