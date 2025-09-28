import pytest

import password_utils
from password_utils import (
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    PBKDF2_PREFIX,
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
    with pytest.raises(ValueError, match="Пароль слишком короткий"):
        hash_password(too_short)


def test_hash_password_rejects_long_password():
    """Пароль длиннее максимально допустимого вызывает ожидаемую ошибку."""
    too_long = "Aa1!" + "d" * MAX_PASSWORD_LENGTH
    with pytest.raises(ValueError, match="Пароль превышает максимально допустимую длину"):
        hash_password(too_long)


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


def test_hash_password_uses_pbkdf2_when_bcrypt_missing(
    valid_password: str, monkeypatch: pytest.MonkeyPatch
):
    """При отсутствии bcrypt используется PBKDF2 c корректной верификацией."""
    monkeypatch.setattr(password_utils, "BCRYPT_AVAILABLE", False)
    monkeypatch.setattr(password_utils, "bcrypt", None)

    hashed = hash_password(valid_password)
    assert hashed.startswith(f"{PBKDF2_PREFIX}$")
    assert verify_password(valid_password, hashed)
    assert not verify_password(valid_password + "z", hashed)


def test_verify_password_rejects_corrupted_pbkdf2_hash(valid_password: str):
    """Повреждённый PBKDF2-хэш приводит к ValueError."""
    corrupted = f"{PBKDF2_PREFIX}$bad$ff"
    with pytest.raises(ValueError, match="PBKDF2"):
        verify_password(valid_password, corrupted)


def test_verify_password_rejects_unknown_hash_format(valid_password: str):
    """Неизвестный формат хэша приводит к ValueError с понятным сообщением."""
    with pytest.raises(ValueError, match="Неизвестный формат"):
        verify_password(valid_password, "not-a-valid-bcrypt-hash")


def test_verify_password_rejects_malformed_bcrypt_hash(valid_password: str):
    """Повреждённый bcrypt-хэш распознаётся и сообщает об ошибке."""
    malformed = "$2b$12$short"
    with pytest.raises(ValueError, match="bcrypt"):
        verify_password(valid_password, malformed)
