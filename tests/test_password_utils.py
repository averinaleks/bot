import pytest

from password_utils import (
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    hash_password,
    verify_password,
)


def test_hash_password_allows_short_password():
    password = "a" * MAX_PASSWORD_LENGTH
    hashed = hash_password(password)
    assert hashed.startswith("$2b$")
    assert verify_password(password, hashed)


def test_hash_password_rejects_long_password():
    with pytest.raises(ValueError):
        hash_password("a" * (MAX_PASSWORD_LENGTH + 1))


def test_verify_password_rejects_long_password():
    hashed = hash_password("a" * MIN_PASSWORD_LENGTH)
    with pytest.raises(ValueError):
        verify_password("a" * (MAX_PASSWORD_LENGTH + 1), hashed)


@pytest.mark.parametrize("password", ["", "a" * (MIN_PASSWORD_LENGTH - 1)])
def test_hash_password_rejects_short_password(password):
    with pytest.raises(ValueError):
        hash_password(password)


@pytest.mark.parametrize("password", ["", "a" * (MIN_PASSWORD_LENGTH - 1)])
def test_verify_password_rejects_short_password(password):
    hashed = hash_password("a" * MIN_PASSWORD_LENGTH)
    with pytest.raises(ValueError):
        verify_password(password, hashed)


def test_hash_password_generates_unique_hashes():
    password = "a" * MIN_PASSWORD_LENGTH
    assert hash_password(password) != hash_password(password)

