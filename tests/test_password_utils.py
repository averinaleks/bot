import pytest

from password_utils import (
    MIN_PASSWORD_LENGTH,
    hash_password,
    verify_password,
)


@pytest.fixture
def valid_password() -> str:
    """Возвращает пароль, удовлетворяющий всем требованиям."""
    padding = max(MIN_PASSWORD_LENGTH - 4, 0)
    return "Aa1!" + "b" * padding


def test_hash_password_rejects_short_password():
    """Пароль короче минимальной длины приводит к ValueError."""
    padding = max(MIN_PASSWORD_LENGTH - 5, 0)
    too_short = "Aa1!" + "c" * padding
    if len(too_short) >= MIN_PASSWORD_LENGTH:
        too_short = too_short[: MIN_PASSWORD_LENGTH - 1]
    with pytest.raises(ValueError, match="Password too short"):
        hash_password(too_short)


def test_hash_password_requires_special_character():
    """Отсутствие спецсимвола фиксируется как нарушение сложности."""
    padding = max(MIN_PASSWORD_LENGTH - 3, 0)
    no_special = "Aa1" + "d" * padding
    if len(no_special) < MIN_PASSWORD_LENGTH:
        no_special += "d" * (MIN_PASSWORD_LENGTH - len(no_special))
    with pytest.raises(ValueError) as excinfo:
        hash_password(no_special)
    assert "отсутствует спецсимвол" in str(excinfo.value)


def test_hash_password_accepts_valid_password(valid_password: str):
    """Корректный пароль успешно хэшируется и проходит проверку."""
    hashed = hash_password(valid_password)
    assert hashed != valid_password
    assert verify_password(valid_password, hashed)


def test_verify_password_handles_incorrect_value(valid_password: str):
    """`verify_password` возвращает `False` для неверного пароля."""
    hashed = hash_password(valid_password)
    assert verify_password(valid_password, hashed)
    assert not verify_password(valid_password + "x", hashed)
