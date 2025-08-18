import logging
from bot.config import load_config


def test_load_config_logs_invalid_env(monkeypatch, caplog):
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "oops")
    with caplog.at_level(logging.WARNING):
        load_config()
    assert "Failed to convert 'oops' to int" in caplog.text
