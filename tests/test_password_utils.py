import base64
import pytest

from password_utils import (
    SALT_SIZE,
    MAX_PASSWORD_LENGTH,
    hash_password,
    verify_password,
)


def test_hash_password_allows_short_password():
    password = "a" * MAX_PASSWORD_LENGTH
    result = hash_password(password)
    data = base64.b64decode(result)
    assert len(data) == SALT_SIZE + 32
    assert verify_password(password, result)


def test_hash_password_rejects_long_password():
    with pytest.raises(ValueError):
        hash_password("a" * (MAX_PASSWORD_LENGTH + 1))
