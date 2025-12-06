import logging

import pytest

import config


def test_validate_service_factories_rejects_external_module():
    with pytest.raises(ValueError, match="Factory modules must reside"):
        config._validate_service_factories({"exchange": "os:system"})


def test_validate_service_factories_warns_unknown_key(caplog):
    caplog.set_level(logging.WARNING, logger="config")

    factories = config._validate_service_factories(
        {"custom": "services.service_factories:build_exchange"}
    )

    assert factories["custom"] == "services.service_factories:build_exchange"
    assert "Допустимые ключи" in caplog.text
