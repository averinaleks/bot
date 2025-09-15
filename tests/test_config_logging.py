import importlib
import logging
import sys

from bot.config import load_config


def test_load_config_logs_invalid_env(monkeypatch, caplog):
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "oops")
    with caplog.at_level(logging.WARNING):
        load_config()
    assert "Failed to convert 'oops' to int" in caplog.text


def test_config_handles_missing_dotenv():
    module_names = ("dotenv", "bot.dotenv_utils", "bot.config")
    saved_modules = {name: sys.modules.get(name) for name in module_names}

    for name in module_names:
        sys.modules.pop(name, None)
    sys.modules["dotenv"] = None  # simulate missing dependency

    try:
        config = importlib.import_module("bot.config")
        assert hasattr(config, "load_defaults")
        assert isinstance(config.load_defaults(), dict)
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module

        helpers_module = saved_modules.get("bot.dotenv_utils")
        if helpers_module is not None:
            importlib.reload(helpers_module)
        else:
            importlib.import_module("bot.dotenv_utils")

        config_module = saved_modules.get("bot.config")
        if config_module is not None:
            importlib.reload(config_module)
        else:
            importlib.import_module("bot.config")
