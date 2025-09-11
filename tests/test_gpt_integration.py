import pytest
from bot import trading_bot
from bot.gpt_client import GPTClientJSONError


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
