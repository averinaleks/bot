import sys

import pytest

_MissingModule = object()


@pytest.mark.usefixtures("_clean_model_builder_modules")
def test_offline_model_builder_import_without_numpy_pandas(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("OFFLINE_MODE", "0")

    for name in [
        "bot.model_builder",
        "model_builder",
        "model_builder.core",
        "model_builder.storage",
        "bot.model_builder.core",
        "bot.model_builder.storage",
        "bot.config",
        "config",
    ]:
        sys.modules.pop(name, None)

    original_numpy = sys.modules.get("numpy", _MissingModule)
    original_pandas = sys.modules.get("pandas", _MissingModule)
    sys.modules["numpy"] = None
    sys.modules["pandas"] = None
    try:
        from bot.model_builder import ModelBuilder  # noqa: PLC0415
        from bot.model_builder.offline import OfflineModelBuilder  # noqa: PLC0415

        assert ModelBuilder is OfflineModelBuilder
    finally:
        if original_numpy is _MissingModule:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = original_numpy
        if original_pandas is _MissingModule:
            sys.modules.pop("pandas", None)
        else:
            sys.modules["pandas"] = original_pandas

    monkeypatch.setenv("OFFLINE_MODE", "1")
    for name in ["bot.config", "config", "bot.trade_manager", "trade_manager"]:
        sys.modules.pop(name, None)

    import run_bot
    from config import BotConfig  # noqa: PLC0415

    cfg = BotConfig()
    _, mb, _ = run_bot._build_components(cfg, offline=True, symbols=None)

    assert isinstance(mb, OfflineModelBuilder)

    for name in [
        "bot.model_builder",
        "model_builder",
        "model_builder.core",
        "model_builder.storage",
        "bot.model_builder.core",
        "bot.model_builder.storage",
    ]:
        sys.modules.pop(name, None)


@pytest.fixture(name="_clean_model_builder_modules")
def _clean_model_builder_modules_fixture():
    modules = {
        name: sys.modules.get(name)
        for name in [
            "bot.model_builder",
            "model_builder",
            "model_builder.core",
            "model_builder.storage",
            "bot.model_builder.core",
            "bot.model_builder.storage",
        ]
    }
    for name in modules:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name, module in modules.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)
