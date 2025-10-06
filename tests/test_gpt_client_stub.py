import asyncio
import importlib
import json
import logging
import sys

import pytest

from bot.gpt_client import Signal, SignalAction, parse_signal


def test_parse_signal_valid():
    payload = json.dumps({"signal": "buy", "tp_mult": 1.2, "sl_mult": 0.8})
    result = parse_signal(payload)
    assert result.signal == SignalAction.buy
    assert result.tp_mult == 1.2
    assert result.sl_mult == 0.8


def test_parse_signal_invalid_json(caplog):
    with caplog.at_level(logging.WARNING):
        result = parse_signal("not json")
    assert result.model_dump() == Signal().model_dump()
    assert "Failed to parse signal" in caplog.text


@pytest.mark.parametrize(
    "field,value",
    [("tp_mult", -0.1), ("tp_mult", 11), ("sl_mult", -0.5), ("sl_mult", 12)],
)
def test_parse_signal_out_of_range(field, value, caplog):
    payload = json.dumps({field: value})
    with caplog.at_level(logging.WARNING):
        result = parse_signal(payload)
    assert result.model_dump() == Signal().model_dump()
    assert "Failed to parse signal" in caplog.text


def test_parse_signal_boundary_values():
    payload = json.dumps({"tp_mult": 0, "sl_mult": 10})
    result = parse_signal(payload)
    assert result.tp_mult == 0
    assert result.sl_mult == 10


def test_gpt_client_imports_offline_stub(monkeypatch):
    from bot import config as bot_config

    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setattr(bot_config, "OFFLINE_MODE", True)
    monkeypatch.delitem(sys.modules, "bot.gpt_client", raising=False)
    monkeypatch.delitem(sys.modules, "httpx", raising=False)

    gpt_client = importlib.import_module("bot.gpt_client")

    assert getattr(gpt_client, "__offline_stub__", False) is True
    result = asyncio.run(gpt_client.query_gpt_json_async("prompt"))
    assert result == {"signal": "hold"}
