import json
import pytest

from bot.gpt_client import (
    GPTClientJSONError,
    GPTClientResponseError,
    Signal,
    parse_signal,
)


def test_parse_signal_valid():
    payload = json.dumps({"signal": "buy", "tp_mult": 1.2, "sl_mult": 0.8})
    assert parse_signal(payload) == Signal(signal="buy", tp_mult=1.2, sl_mult=0.8)


def test_parse_signal_invalid_json():
    with pytest.raises(GPTClientJSONError):
        parse_signal("not json")


@pytest.mark.parametrize("field,value", [("tp_mult", -0.1), ("tp_mult", 11), ("sl_mult", -0.5), ("sl_mult", 12)])
def test_parse_signal_out_of_range(field, value):
    payload = json.dumps({field: value})
    with pytest.raises(GPTClientResponseError):
        parse_signal(payload)


def test_parse_signal_boundary_values():
    payload = json.dumps({"tp_mult": 0, "sl_mult": 10})
    result = parse_signal(payload)
    assert result.tp_mult == 0
    assert result.sl_mult == 10
