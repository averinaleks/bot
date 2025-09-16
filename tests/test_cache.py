import gzip
import os
import pickle
import pandas as pd
import pytest
from bot.cache import HistoricalDataCache

psutil = pytest.importorskip("psutil")


def _mock_virtual_memory():
    class Mem:
        percent = 0
        available = 1024 * 1024 * 1024
    return Mem


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    cache = HistoricalDataCache(cache_dir=str(tmp_path), min_free_disk_gb=0)
    df = pd.DataFrame({"close": [1, 2, 3]})
    cache.save_cached_data("BTC/USDT", "1m", df)
    file_path = tmp_path / "BTC_USDT_1m.parquet"
    assert file_path.exists()
    loaded = cache.load_cached_data("BTC/USDT", "1m")
    assert loaded.equals(df)


def test_load_multiindex_timestamps_localized(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    cache = HistoricalDataCache(cache_dir=str(tmp_path), min_free_disk_gb=0)
    # Build a MultiIndex with a naive ``timestamp`` level to ensure the cache
    # normalises it back to UTC when loading.
    timestamps = pd.date_range("2025-01-01", periods=3, freq="1h", tz=None)
    idx = pd.MultiIndex.from_product(
        [["BTCUSDT"], timestamps], names=["symbol", "timestamp"]
    )
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    cache.save_cached_data("BTCUSDT", "1m", df)
    loaded = cache.load_cached_data("BTCUSDT", "1m")
    ts_level = loaded.index.get_level_values("timestamp")
    assert getattr(ts_level, "tz", None) is not None
    expected_ts = pd.to_datetime(timestamps).tz_localize("UTC")
    expected_index = pd.MultiIndex.from_arrays(
        [idx.get_level_values("symbol"), expected_ts], names=idx.names
    )
    expected = df.copy()
    expected.index = expected_index
    pd.testing.assert_frame_equal(loaded.sort_index(), expected.sort_index())


@pytest.mark.parametrize("suffix", [".pkl", ".pkl.gz"])
def test_load_rejects_pickle_cache(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    cache = HistoricalDataCache(cache_dir=str(tmp_path), min_free_disk_gb=0)
    df = pd.DataFrame({"close": [1, 2, 3]})
    old_file = tmp_path / f"BTCUSDT_1m{suffix}"
    if suffix.endswith(".gz"):
        with gzip.open(old_file, "wb") as f:
            f.write(pickle.dumps(df))
    else:
        with open(old_file, "wb") as f:
            pickle.dump(df, f)
    loaded = cache.load_cached_data("BTCUSDT", "1m")
    assert loaded is None
    assert not old_file.exists()
    assert not (tmp_path / "BTCUSDT_1m.parquet").exists()


def test_save_skips_empty_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    cache = HistoricalDataCache(cache_dir=str(tmp_path), min_free_disk_gb=0)
    empty_df = pd.DataFrame()
    cache.save_cached_data("BTC/USDT", "1m", empty_df)
    assert not (tmp_path / "BTC_USDT_1m.parquet").exists()


def test_cache_size_updates_without_walk(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    cache = HistoricalDataCache(cache_dir=str(tmp_path), min_free_disk_gb=0)
    # fail if _calculate_cache_size is used after init
    def fail(*a, **k):
        raise AssertionError("walk not called")
    monkeypatch.setattr(cache, "_calculate_cache_size", fail)
    cache.max_cache_size_mb = 10
    cache.max_buffer_size_mb = 10
    df = pd.DataFrame({"close": list(range(100))})
    cache.save_cached_data("BTC/USDT", "1m", df)
    file_path = tmp_path / "BTC_USDT_1m.parquet"
    size_mb = file_path.stat().st_size / (1024 * 1024)
    assert abs(cache.current_cache_size_mb - size_mb) < 0.01
    cache.max_cache_size_mb = 0
    cache._aggressive_clean()
    assert cache.current_cache_size_mb == 0
    assert not file_path.exists()


def test_calculate_cache_size_skips_deleted_files(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    file_path = tmp_path / "del.parquet"
    file_path.write_bytes(b"data")
    orig_getsize = os.path.getsize

    def fake_getsize(path):
        if path == str(file_path):
            file_path.unlink()
            raise FileNotFoundError
        return orig_getsize(path)

    monkeypatch.setattr(os.path, "getsize", fake_getsize)
    cache = HistoricalDataCache(cache_dir=str(tmp_path), min_free_disk_gb=0)
    assert cache.current_cache_size_mb == 0


def test_delete_cache_file_never_negative(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", _mock_virtual_memory)
    cache = HistoricalDataCache(cache_dir=str(tmp_path), min_free_disk_gb=0)
    # simulate accounting mismatch by starting with tiny size
    cache.current_cache_size_mb = 0.001
    file_path = tmp_path / "dummy.parquet"
    file_path.write_bytes(b"data")
    cache._delete_cache_file(str(file_path))
    assert cache.current_cache_size_mb >= 0
