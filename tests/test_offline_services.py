import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "expected_key, target",
    [
        ("exchange", "services.offline:OfflineBybit"),
        ("telegram_logger", "services.offline:OfflineTelegram"),
        ("gpt_client", "services.offline:OfflineGPT"),
        ("model_builder", "model_builder.offline:OfflineModelBuilder"),
        ("trade_manager", "services.offline:OfflineTradeManager"),
    ],
)
def test_offline_factories_mapping(expected_key, target):
    from services.offline import OFFLINE_SERVICE_FACTORIES

    assert OFFLINE_SERVICE_FACTORIES.get(expected_key) == target


def test_offline_bybit_generates_deterministic_data():
    from services.offline import OfflineBybit

    client = OfflineBybit()
    ticker_first = client.fetch_ticker("BTCUSDT")
    ticker_second = client.fetch_ticker("BTCUSDT")

    assert ticker_first["last"] == ticker_second["last"] > 0

    candles = client.fetch_ohlcv("BTCUSDT", limit=3)
    assert len(candles) == 3

    timestamps = [candle[0] for candle in candles]
    assert timestamps == sorted(timestamps)


def test_offline_telegram_logs_message(caplog):
    from services.offline import OfflineTelegram

    caplog.set_level("INFO")
    telegram = OfflineTelegram(chat_id="123")
    asyncio.run(telegram.send_telegram_message("test message"))

    assert any("OFFLINE TELEGRAM" in record.message for record in caplog.records)


def test_offline_trade_manager_persists_state(tmp_path):
    from services.offline import OfflineBybit, OfflineTradeManager

    config = SimpleNamespace(cache_dir=str(tmp_path), offline_iterations=1)
    data_handler = SimpleNamespace(exchange=OfflineBybit(), usdt_pairs=["BTCUSDT"])
    model_builder = SimpleNamespace(update_models=lambda: None)
    telegram_bot = SimpleNamespace()

    trade_manager = OfflineTradeManager(
        config,
        data_handler,
        model_builder,
        telegram_bot,
        chat_id="chat",
    )

    trade_manager.positions_changed = True
    trade_manager.save_state()

    df = pd.read_json(tmp_path / "trade_manager_state.json")
    assert df.empty or "symbol" in df.columns
