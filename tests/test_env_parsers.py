import logging
import pytest
import password_utils as pu


def test_default_when_env_missing(monkeypatch):
    monkeypatch.delenv("BCRYPT_ROUNDS", raising=False)
    assert pu.get_bcrypt_rounds() == pu.DEFAULT_BCRYPT_ROUNDS


def test_default_when_env_not_numeric(monkeypatch, caplog):
    monkeypatch.setenv("BCRYPT_ROUNDS", "not-a-number")
    with caplog.at_level(logging.WARNING):
        assert pu.get_bcrypt_rounds() == pu.DEFAULT_BCRYPT_ROUNDS
        assert "not an integer" in caplog.text


@pytest.mark.parametrize("value", ["-1", "0"])
def test_default_when_env_non_positive(monkeypatch, caplog, value):
    monkeypatch.setenv("BCRYPT_ROUNDS", value)
    with caplog.at_level(logging.WARNING):
        assert pu.get_bcrypt_rounds() == pu.DEFAULT_BCRYPT_ROUNDS
        assert "between 4 and 31" in caplog.text
