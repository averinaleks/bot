from collections import deque

import pytest
from bot import trading_bot
from bot.gpt_client import GPTClientError, GPTClientJSONError


@pytest.fixture(autouse=True)
def reset_gpt_state():
    trading_bot.GPT_ADVICE = trading_bot.GPTAdviceModel()
    trading_bot._GPT_ADVICE_ERROR_COUNT = 0
    trading_bot._GPT_SAFE_MODE = False
    yield
    trading_bot.GPT_ADVICE = trading_bot.GPTAdviceModel()
    trading_bot._GPT_ADVICE_ERROR_COUNT = 0
    trading_bot._GPT_SAFE_MODE = False


@pytest.mark.asyncio
async def test_refresh_gpt_advice_parses(monkeypatch):
    async def fake_query(prompt):
        return {"signal": 0.5, "tp_mult": 1.2, "sl_mult": 0.8}

    alerts = []

    async def fake_alert(msg):
        alerts.append(msg)

    monkeypatch.setattr(trading_bot, "query_gpt_json_async", fake_query)
    monkeypatch.setattr(trading_bot, "send_telegram_alert", fake_alert)

    await trading_bot.refresh_gpt_advice()
    assert trading_bot.GPT_ADVICE.signal == 0.5
    assert not alerts


@pytest.mark.asyncio
async def test_refresh_gpt_advice_invalid_json(monkeypatch):
    async def fake_query(prompt):
        raise GPTClientJSONError("bad json")

    alerts = []

    async def fake_alert(msg):
        alerts.append(msg)

    monkeypatch.setattr(trading_bot, "query_gpt_json_async", fake_query)
    monkeypatch.setattr(trading_bot, "send_telegram_alert", fake_alert)

    await trading_bot.refresh_gpt_advice()
    assert alerts
    assert trading_bot.GPT_ADVICE.signal == "hold"


@pytest.mark.asyncio
async def test_refresh_gpt_advice_safe_mode_after_retries(monkeypatch):
    async def fake_features(symbol, price):
        return [price, 0.0, price, 0.0, 50.0]

    attempts = {"count": 0}

    async def failing_query(prompt):
        attempts["count"] += 1
        raise GPTClientError("boom")

    alerts: list[str] = []

    async def fake_alert(msg: str) -> None:
        alerts.append(msg)

    toggles: list[bool] = []

    async def fake_toggle(value: bool) -> None:
        toggles.append(value)

    monkeypatch.setattr(trading_bot, "build_feature_vector", fake_features)
    monkeypatch.setattr(trading_bot, "SYMBOLS", ["BTCUSDT"])
    monkeypatch.setattr(
        trading_bot,
        "_PRICE_HISTORY",
        {"BTCUSDT": deque([1.0], maxlen=200)},
    )
    monkeypatch.setattr(trading_bot, "_load_env", lambda: {})
    monkeypatch.setattr(trading_bot, "query_gpt_json_async", failing_query)
    monkeypatch.setattr(trading_bot, "send_telegram_alert", fake_alert)
    monkeypatch.setattr(trading_bot, "set_trading_enabled", fake_toggle)

    for _ in range(trading_bot.GPT_ADVICE_MAX_ATTEMPTS):
        await trading_bot.refresh_gpt_advice()

    assert trading_bot.GPT_ADVICE.signal == "hold"
    assert trading_bot._GPT_SAFE_MODE is True
    assert toggles == [False]
    assert any("safe mode" in msg.lower() for msg in alerts)

    await trading_bot.refresh_gpt_advice()
    assert attempts["count"] == trading_bot.GPT_ADVICE_MAX_ATTEMPTS
    assert toggles == [False]
