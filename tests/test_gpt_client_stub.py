import json
import logging
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
