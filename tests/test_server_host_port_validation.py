import importlib
import logging
import sys
import types

import pytest


@pytest.mark.parametrize(
    "env_port, env_host, expected_message",
    [
        ("notaport", "127.0.0.1", "Invalid PORT value"),
        ("8000", "0.0.0.0", "Invalid HOST"),
    ],
)
def test_server_invalid_host_or_port(monkeypatch, caplog, env_port, env_host, expected_message):
    stub_pydantic = types.SimpleNamespace(
        BaseModel=object,
        Field=lambda default=None, *_, **__: default,
        ValidationError=Exception,
    )
    monkeypatch.setitem(sys.modules, "pydantic", stub_pydantic)
    monkeypatch.setenv("CSRF_SECRET", "unit-test-secret")
    monkeypatch.setenv("PORT", env_port)
    monkeypatch.setenv("HOST", env_host)
    caplog.set_level(logging.ERROR)

    sys.modules.pop("server", None)

    with pytest.raises(RuntimeError) as excinfo:
        importlib.import_module("server")

    assert expected_message in str(excinfo.value)
    assert any(expected_message in record.getMessage() for record in caplog.records)

    sys.modules.pop("server", None)
