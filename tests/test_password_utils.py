import pytest
import bcrypt

from password_utils import (
    BCRYPT_ROUNDS,
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    hash_password,
    verify_password,
)


VALID_PASSWORD = "Aa1!" + "a" * (MIN_PASSWORD_LENGTH - 4)
WEAK_PASSWORDS = [
    "aaaaaaaa",  # no uppercase, digit, special
    "AAAAAAAA",  # no lowercase, digit, special
    "AAAAaaaa",  # no digit, special
    "AAAAaaa1",  # no special
    "AAAA!aaa",  # no digit
]


def test_hash_password_allows_short_password():
    password = "Aa1!" + "a" * (MAX_PASSWORD_LENGTH - 4)
    hashed = hash_password(password)
    assert hashed.startswith(f"$2b${BCRYPT_ROUNDS:02d}$")
    assert verify_password(password, hashed)


def test_verify_password_success():
    hashed = hash_password(VALID_PASSWORD)
    assert verify_password(VALID_PASSWORD, hashed)


def test_hash_password_rejects_long_password():
    long_password = "Aa1!" + "a" * (MAX_PASSWORD_LENGTH - 3)
    with pytest.raises(ValueError):
        hash_password(long_password)


def test_verify_password_rejects_long_password():
    hashed = hash_password(VALID_PASSWORD)
    long_password = "Aa1!" + "a" * (MAX_PASSWORD_LENGTH - 3)
    with pytest.raises(ValueError):
        verify_password(long_password, hashed)


@pytest.mark.parametrize("password", ["", "a" * (MIN_PASSWORD_LENGTH - 1)])
def test_hash_password_rejects_short_password(password):
    with pytest.raises(ValueError):
        hash_password(password)


@pytest.mark.parametrize("password", ["", "a" * (MIN_PASSWORD_LENGTH - 1)])
def test_verify_password_rejects_short_password(password):
    hashed = hash_password(VALID_PASSWORD)
    with pytest.raises(ValueError):
        verify_password(password, hashed)


def test_hash_password_generates_unique_hashes():
    assert hash_password(VALID_PASSWORD) != hash_password(VALID_PASSWORD)


@pytest.mark.parametrize("weak_password", WEAK_PASSWORDS)
def test_hash_password_rejects_weak_passwords(weak_password):
    with pytest.raises(ValueError):
        hash_password(weak_password)


@pytest.mark.parametrize("weak_password", WEAK_PASSWORDS)
def test_verify_password_accepts_existing_weak_hashes(weak_password):
    stored_hash = bcrypt.hashpw(weak_password.encode(), bcrypt.gensalt()).decode()
    assert verify_password(weak_password, stored_hash)

