import sys

import pytest

import run_bot


def test_parse_symbols_strips_and_uppercases():
    assert run_bot.parse_symbols(None) is None
    assert run_bot.parse_symbols("") is None
    assert run_bot.parse_symbols(" ,btc ,, eth ") == ["BTC", "ETH"]


def test_parse_args_runtime_normalization(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", ".")
    monkeypatch.setattr(sys, "argv", ["run_bot.py", "--runtime", "0"])

    args = run_bot.parse_args()

    assert args.command == "trade"
    assert args.runtime is None


@pytest.mark.parametrize(
    "argv,expected_command",
    [
        (["run_bot.py"], "trade"),
        (["run_bot.py", "simulate", "--start", "2024-01-01", "--end", "2024-01-02"], "simulate"),
    ],
)
def test_parse_args_defaults(monkeypatch, argv, expected_command):
    monkeypatch.setattr(sys, "argv", argv)
    args = run_bot.parse_args()
    assert args.command == expected_command
    if expected_command != "trade":
        assert args.runtime is None
