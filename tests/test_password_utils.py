import crypt
import pytest

from password_utils import hash_password, MAX_PASSWORD_LENGTH


def test_hash_password_allows_short_password():
    salt = crypt.mksalt(crypt.METHOD_SHA512)
    result = hash_password("a" * MAX_PASSWORD_LENGTH, salt)
    assert result.startswith("$6$")


def test_hash_password_rejects_long_password():
    salt = crypt.mksalt(crypt.METHOD_SHA512)
    with pytest.raises(ValueError):
        hash_password("a" * (MAX_PASSWORD_LENGTH + 1), salt)
