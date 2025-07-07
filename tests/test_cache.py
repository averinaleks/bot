import pickle
import pandas as pd

from utils import HistoricalDataCache, psutil


def _mock_virtual_memory():
    class Mem:
        percent = 0
        available = 1024 * 1024 * 1024
    return Mem


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    cache = HistoricalDataCache(cache_dir=str(tmp_path))
    df = pd.DataFrame({"close": [1, 2, 3]})
    cache.save_cached_data("BTC/USDT", "1m", df)
    file_path = tmp_path / "BTC_USDT_1m.pkl.gz"
    assert file_path.exists()
    loaded = cache.load_cached_data("BTC/USDT", "1m")
    assert loaded.equals(df)


def test_load_converts_old_format(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    cache = HistoricalDataCache(cache_dir=str(tmp_path))
    df = pd.DataFrame({"close": [1, 2, 3]})
    old_file = tmp_path / "BTCUSDT_1m.pkl"
    with open(old_file, "wb") as f:
        pickle.dump(df, f)
    loaded = cache.load_cached_data("BTCUSDT", "1m")
    new_file = tmp_path / "BTCUSDT_1m.pkl.gz"
    assert loaded.equals(df)
    assert new_file.exists()
    assert not old_file.exists()
    loaded_again = cache.load_cached_data("BTCUSDT", "1m")
    assert loaded_again.equals(df)
