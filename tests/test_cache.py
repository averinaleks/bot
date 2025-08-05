import pickle
import os
import pandas as pd

from bot.utils import HistoricalDataCache, psutil


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
    file_path = tmp_path / "BTC_USDT_1m.json.gz"
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
    new_file = tmp_path / "BTCUSDT_1m.json.gz"
    assert loaded.equals(df)
    assert new_file.exists()
    assert not old_file.exists()
    loaded_again = cache.load_cached_data("BTCUSDT", "1m")
    assert loaded_again.equals(df)


def test_save_skips_empty_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    cache = HistoricalDataCache(cache_dir=str(tmp_path))
    empty_df = pd.DataFrame()
    cache.save_cached_data("BTC/USDT", "1m", empty_df)
    assert not (tmp_path / "BTC_USDT_1m.json.gz").exists()


def test_cache_size_updates_without_walk(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    cache = HistoricalDataCache(cache_dir=str(tmp_path))
    # fail if _calculate_cache_size is used after init
    def fail(*a, **k):
        raise AssertionError("walk not called")
    monkeypatch.setattr(cache, "_calculate_cache_size", fail)
    cache.max_cache_size_mb = 10
    cache.max_buffer_size_mb = 10
    df = pd.DataFrame({"close": list(range(100))})
    cache.save_cached_data("BTC/USDT", "1m", df)
    file_path = tmp_path / "BTC_USDT_1m.json.gz"
    size_mb = file_path.stat().st_size / (1024 * 1024)
    assert abs(cache.current_cache_size_mb - size_mb) < 0.01
    cache.max_cache_size_mb = 0
    cache._aggressive_clean()
    assert cache.current_cache_size_mb == 0
    assert not file_path.exists()


def test_calculate_cache_size_skips_deleted_files(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    file_path = tmp_path / "del.json.gz"
    file_path.write_bytes(b"data")
    orig_getsize = os.path.getsize

    def fake_getsize(path):
        if path == str(file_path):
            file_path.unlink()
            raise FileNotFoundError
        return orig_getsize(path)

    monkeypatch.setattr(os.path, "getsize", fake_getsize)
    cache = HistoricalDataCache(cache_dir=str(tmp_path))
    assert cache.current_cache_size_mb == 0
