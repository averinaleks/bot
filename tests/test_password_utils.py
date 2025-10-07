import hmac

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


@pytest.mark.parametrize("length", [MIN_PASSWORD_LENGTH, MAX_PASSWORD_LENGTH])
@pytest.mark.parametrize("mode", ["bcrypt", "pbkdf2"])
def test_hash_password_accepts_boundary_lengths(
    length: int, mode: str, monkeypatch: pytest.MonkeyPatch
):
    """Пароли минимальной и максимальной длины хэшируются в обоих режимах."""

    def build_password(target_length: int) -> str:
        base = "Aa1!"
        if target_length <= len(base):
            return base[:target_length]
        return base + "b" * (target_length - len(base))

    password = build_password(length)

    if mode == "bcrypt":

        class FakeBcrypt:
            prefix = "$2b$12$"

            @staticmethod
            def gensalt(rounds: int) -> bytes:  # pragma: no cover - deterministic salt
                assert rounds == password_utils.DEFAULT_BCRYPT_ROUNDS
                return FakeBcrypt.prefix.encode()

            @staticmethod
            def hashpw(password_bytes: bytes, salt: bytes) -> bytes:
                assert salt.decode() == FakeBcrypt.prefix
                return (FakeBcrypt.prefix + password_bytes.decode()).encode()

            @staticmethod
            def checkpw(password_bytes: bytes, stored_bytes: bytes) -> bool:
                expected = (FakeBcrypt.prefix + password_bytes.decode()).encode()
                return hmac.compare_digest(expected, stored_bytes)

        monkeypatch.setattr(password_utils, "bcrypt", FakeBcrypt)
        monkeypatch.setattr(password_utils, "BCRYPT_AVAILABLE", True)
    else:
        monkeypatch.setattr(password_utils, "bcrypt", None)
        monkeypatch.setattr(password_utils, "BCRYPT_AVAILABLE", False)

    hashed = hash_password(password)
    assert hashed.startswith("$2") if mode == "bcrypt" else hashed.startswith(
        f"{PBKDF2_PREFIX}$"
    )
    assert verify_password(password, hashed)


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


def test_verify_password_rejects_malformed_bcrypt_hash(
    valid_password: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Повреждённый bcrypt-хэш распознаётся и сообщает подробную причину."""

    class FakeBcrypt:
        @staticmethod
        def checkpw(password: bytes, stored: bytes) -> bool:
            raise ValueError("invalid salt")

    monkeypatch.setattr(password_utils, "bcrypt", FakeBcrypt)
    monkeypatch.setattr(password_utils, "BCRYPT_AVAILABLE", True)

    bcrypt_hash = "$2b$12$" + "a" * 53

    caplog.set_level("ERROR", logger=password_utils.__name__)

    with pytest.raises(ValueError) as excinfo:
        verify_password(valid_password, bcrypt_hash)

    message = str(excinfo.value)
    assert "bcrypt" in message
    assert "invalid salt" in message
    assert any("invalid salt" in record.getMessage() for record in caplog.records)


def test_verify_password_preserves_type_error_context(
    valid_password: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """TypeError от bcrypt сохраняет текст ошибки в исключении и логах."""

    class FakeBcrypt:
        @staticmethod
        def checkpw(password: bytes, stored: bytes) -> bool:
            raise TypeError("bytes-like object required")

    monkeypatch.setattr(password_utils, "bcrypt", FakeBcrypt)
    monkeypatch.setattr(password_utils, "BCRYPT_AVAILABLE", True)

    bcrypt_hash = "$2b$12$" + "b" * 53

    caplog.set_level("ERROR", logger=password_utils.__name__)

    with pytest.raises(ValueError) as excinfo:
        verify_password(valid_password, bcrypt_hash)

    message = str(excinfo.value)
    assert "bcrypt" in message
    assert "bytes-like object required" in message
    assert any("bytes-like object required" in record.getMessage() for record in caplog.records)


def test_verify_password_treats_pyo3_runtime_error_as_corrupted(
    valid_password: str, monkeypatch: pytest.MonkeyPatch
):
    """Ошибки PyO3 при проверке bcrypt считаются повреждённым хэшем."""

    class PyO3RuntimeError(RuntimeError):
        __module__ = "pyo3_runtime"

    class FakeBcrypt:
        @staticmethod
        def checkpw(password: bytes, stored: bytes) -> bool:
            raise PyO3RuntimeError("panic")

    monkeypatch.setattr(password_utils, "bcrypt", FakeBcrypt)
    monkeypatch.setattr(password_utils, "BCRYPT_AVAILABLE", True)

    bcrypt_hash = "$2b$12$" + "a" * 53

    with pytest.raises(ValueError, match="bcrypt"):
        verify_password(valid_password, bcrypt_hash)


def test_verify_password_does_not_swallow_system_exit(
    valid_password: str, monkeypatch: pytest.MonkeyPatch
):
    """Системные исключения не подавляются при проверке bcrypt."""

    class FakeBcrypt:
        @staticmethod
        def checkpw(password: bytes, stored: bytes) -> bool:
            raise SystemExit("terminate")

    monkeypatch.setattr(password_utils, "bcrypt", FakeBcrypt)
    monkeypatch.setattr(password_utils, "BCRYPT_AVAILABLE", True)

    bcrypt_hash = "$2b$12$" + "b" * 53

    with pytest.raises(SystemExit):
        verify_password(valid_password, bcrypt_hash)
