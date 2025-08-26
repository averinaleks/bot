import os
import bcrypt
import logging
import re

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 64
DEFAULT_BCRYPT_ROUNDS = 12
logger = logging.getLogger(__name__)


def get_bcrypt_rounds() -> int:
    """Возвращает число раундов bcrypt из переменной окружения.

    Допустимый диапазон значений: от 4 до 31. При недопустимом значении или
    отсутствии переменной используется ``DEFAULT_BCRYPT_ROUNDS`` и
    выводится предупреждение.
    """

    env_value = os.getenv("BCRYPT_ROUNDS")
    if env_value is not None:
        try:
            rounds = int(env_value)
            if 4 <= rounds <= 31:
                return rounds
            logger.warning(
                "BCRYPT_ROUNDS must be between 4 and 31; using default %d",
                DEFAULT_BCRYPT_ROUNDS,
            )
        except ValueError:
            logger.warning(
                "BCRYPT_ROUNDS is not an integer; using default %d",
                DEFAULT_BCRYPT_ROUNDS,
            )
    return DEFAULT_BCRYPT_ROUNDS


def validate_password_complexity(password: str) -> None:
    """Проверяет наличие цифр, верхнего/нижнего регистра и спецсимволов."""
    violations = []
    if not re.search(r"[A-Z]", password):
        violations.append("uppercase")
    if not re.search(r"[a-z]", password):
        violations.append("lowercase")
    if not re.search(r"\d", password):
        violations.append("digit")
    if not re.search(r"[^\w\s]", password):
        violations.append("special")
    if violations:
        raise ValueError("Пароль не соответствует требованиям сложности")


def validate_password_length(password: str) -> None:
    """Проверяет, что длина пароля находится в допустимых пределах."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError("Password too short")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password exceeds maximum length")


def hash_password(password: str) -> str:
    """Хэширует пароль, используя bcrypt."""
    validate_password_length(password)
    validate_password_complexity(password)
    rounds = get_bcrypt_rounds()
    return bcrypt.hashpw(
        password.encode(), bcrypt.gensalt(rounds=rounds)
    ).decode()


def verify_password(password: str, stored_hash: str) -> bool:
    """Проверяет пароль по сохранённому bcrypt-хэшу."""

    try:
        validate_password_length(password)
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    except ValueError:
        # ``bcrypt.checkpw`` raises ``ValueError`` when ``stored_hash`` is not a
        # valid bcrypt hash. Historically this bubbled up to callers, but for a
        # verification helper it's more convenient to treat such cases as a
        # failed password check.  We also treat length violations the same way
        # by reusing the existing ``ValueError`` from ``validate_password_length``.
        return False

