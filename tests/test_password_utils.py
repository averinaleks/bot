import logging

import pytest
import bcrypt

from password_utils import (
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    validate_password_length,
    hash_password,
    verify_password,
    get_bcrypt_rounds,
)


VALID_PASSWORD = "Aa1!" + "a" * (MIN_PASSWORD_LENGTH - 4)
VALID_PASSWORD_NON_ASCII = "Aa1€" + "a" * (MIN_PASSWORD_LENGTH - 4)
WEAK_PASSWORDS = [
    ("aaaaaaaa", ["uppercase", "digit", "special"]),
    ("AAAAAAAA", ["lowercase", "digit", "special"]),
    ("AAAAaaaa", ["digit", "special"]),
    ("AAAAaaa1", ["special"]),
    ("AAAA!aaa", ["digit"]),
]

WEAK_PASSWORD_STRINGS = [p for p, _ in WEAK_PASSWORDS]


def test_validate_password_length_accepts_valid_length():
    validate_password_length("a" * MIN_PASSWORD_LENGTH)


@pytest.mark.parametrize(
    "password, message",
    [
        ("a" * (MIN_PASSWORD_LENGTH - 1), "Password too short"),
        ("a" * (MAX_PASSWORD_LENGTH + 1), "Password exceeds maximum length"),
    ],
)
def test_validate_password_length_rejects_invalid_length(password, message):
    with pytest.raises(ValueError, match=message):
        validate_password_length(password)


def test_hash_password_allows_short_password():
    password = "Aa1!" + "a" * (MAX_PASSWORD_LENGTH - 4)
    hashed = hash_password(password)
    rounds = get_bcrypt_rounds()
    assert hashed.startswith(f"$2b${rounds:02d}$")
    assert verify_password(password, hashed)


def test_verify_password_success():
    hashed = hash_password(VALID_PASSWORD)
    assert verify_password(VALID_PASSWORD, hashed)


def test_hash_password_accepts_non_ascii_special_char():
    hashed = hash_password(VALID_PASSWORD_NON_ASCII)
    assert verify_password(VALID_PASSWORD_NON_ASCII, hashed)


def test_hash_password_rejects_long_password():
    long_password = "Aa1!" + "a" * (MAX_PASSWORD_LENGTH - 3)
    with pytest.raises(ValueError, match="Password exceeds maximum length"):
        hash_password(long_password)


def test_verify_password_rejects_long_password():
    hashed = hash_password(VALID_PASSWORD)
    long_password = "Aa1!" + "a" * (MAX_PASSWORD_LENGTH - 3)
    assert not verify_password(long_password, hashed)


@pytest.mark.parametrize("password", ["", "a" * (MIN_PASSWORD_LENGTH - 1)])
def test_hash_password_rejects_short_password(password):
    with pytest.raises(ValueError, match="Password too short"):
        hash_password(password)


@pytest.mark.parametrize("password", ["", "a" * (MIN_PASSWORD_LENGTH - 1)])
def test_verify_password_rejects_short_password(password):
    hashed = hash_password(VALID_PASSWORD)
    assert not verify_password(password, hashed)


def test_hash_password_generates_unique_hashes():
    assert hash_password(VALID_PASSWORD) != hash_password(VALID_PASSWORD)


@pytest.mark.parametrize("weak_password, violations", WEAK_PASSWORDS)
def test_hash_password_rejects_weak_passwords(weak_password, violations):
    msg = ", ".join(violations)
    with pytest.raises(
        ValueError,
        match=f"Пароль не соответствует требованиям сложности: {msg}",
    ):
        hash_password(weak_password)


@pytest.mark.parametrize("weak_password", WEAK_PASSWORD_STRINGS)
def test_verify_password_accepts_existing_weak_hashes(weak_password):
    stored_hash = bcrypt.hashpw(weak_password.encode(), bcrypt.gensalt()).decode()
    assert verify_password(weak_password, stored_hash)


def test_invalid_bcrypt_rounds_env_uses_default(monkeypatch, caplog):
    import password_utils as pu

    monkeypatch.setenv("BCRYPT_ROUNDS", "32")
    with caplog.at_level(logging.WARNING):
        assert pu.get_bcrypt_rounds() == pu.DEFAULT_BCRYPT_ROUNDS
        pu.get_bcrypt_rounds()
    assert caplog.text.count("BCRYPT_ROUNDS") == 2
    monkeypatch.delenv("BCRYPT_ROUNDS", raising=False)


def test_env_change_affects_rounds(monkeypatch):
    import password_utils as pu

    monkeypatch.setenv("BCRYPT_ROUNDS", "04")
    hashed_low = pu.hash_password(VALID_PASSWORD)
    assert hashed_low.startswith("$2b$04$")

    monkeypatch.setenv("BCRYPT_ROUNDS", "05")
    hashed_high = pu.hash_password(VALID_PASSWORD)
    assert hashed_high.startswith("$2b$05$")

