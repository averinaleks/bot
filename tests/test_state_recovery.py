import os
import types
import logging
import pandas as pd
import pytest
from bot.config import BotConfig


def _make_positions(symbol: str, ts: str) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [(symbol, pd.Timestamp(ts, tz="UTC"))], names=["symbol", "timestamp"]
    )
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "side": "long",
                "size": 1.0,
                "entry_price": 100.0,
                "tp_multiplier": 0.0,
                "sl_multiplier": 0.0,
                "stop_loss_price": 90.0,
                "highest_price": 100.0,
                "lowest_price": 100.0,
                "breakeven_triggered": False,
            }
        ],
        index=idx,
    )


def test_trade_manager_save_recovery(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("TEST_MODE", "1")
    from bot.trade_manager import TradeManager

    cfg = BotConfig(cache_dir=str(tmp_path))
    dh = types.SimpleNamespace(exchange=types.SimpleNamespace(), usdt_pairs=["BTCUSDT"])
    tm = TradeManager(cfg, dh, object(), None, None)

    tm.positions = _make_positions("BTCUSDT", "2020-01-01")
    tm.returns_by_symbol = {"BTCUSDT": [(0.0, 0.1)]}
    tm.positions_changed = True
    tm.last_save_time -= tm.save_interval + 1
    tm.save_state()

    tm.positions = _make_positions("BTCUSDT", "2020-01-02")
    tm.returns_by_symbol = {"BTCUSDT": [(1.0, 0.2)]}
    tm.positions_changed = True
    tm.last_save_time -= tm.save_interval + 1

    real_replace = os.replace

    def fail_replace(src, dst):
        raise OSError("boom")

    monkeypatch.setattr(os, "replace", fail_replace)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(OSError):
            tm.save_state()
        assert any("Failed to save state" in r.message for r in caplog.records)
    monkeypatch.setattr(os, "replace", real_replace)

    tm.positions = pd.DataFrame()
    tm.returns_by_symbol = {}
    tm.load_state()
    assert "BTCUSDT" in tm.positions.index.get_level_values("symbol")
    assert tm.returns_by_symbol["BTCUSDT"][0][1] == 0.1

    tm.positions = _make_positions("BTCUSDT", "2020-01-03")
    tm.returns_by_symbol = {"BTCUSDT": [(2.0, 0.3)]}
    tm.positions_changed = True
    tm.last_save_time -= tm.save_interval + 1
    tm.save_state()
    tm.positions = pd.DataFrame()
    tm.returns_by_symbol = {}
    tm.load_state()
    assert tm.returns_by_symbol["BTCUSDT"][0][1] == 0.3


def test_model_builder_save_recovery(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("TEST_MODE", "1")
    from bot.model_builder import ModelBuilder

    cfg = BotConfig(cache_dir=str(tmp_path))
    dh = types.SimpleNamespace(usdt_pairs=["BTCUSDT"])
    mb = ModelBuilder(cfg, dh, object())

    mb.base_thresholds["BTCUSDT"] = 0.5
    mb.last_save_time -= mb.save_interval + 1
    mb.save_state()

    mb.base_thresholds["BTCUSDT"] = 0.6
    mb.last_save_time -= mb.save_interval + 1

    real_replace = os.replace

    def fail_replace(src, dst):
        raise OSError("boom")

    monkeypatch.setattr(os, "replace", fail_replace)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(OSError):
            mb.save_state()
        assert any(
            "Ошибка сохранения состояния ModelBuilder" in r.message for r in caplog.records
        )
    monkeypatch.setattr(os, "replace", real_replace)

    mb.base_thresholds.clear()
    mb.load_state()
    assert mb.base_thresholds["BTCUSDT"] == 0.5

    mb.base_thresholds["BTCUSDT"] = 0.7
    mb.last_save_time -= mb.save_interval + 1
    mb.save_state()
    mb.base_thresholds.clear()
    mb.load_state()
    assert mb.base_thresholds["BTCUSDT"] == 0.7

