import os
import bcrypt
import logging
import re

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 64
DEFAULT_BCRYPT_ROUNDS = 12
logger = logging.getLogger(__name__)


_bcrypt_rounds_env = os.getenv("BCRYPT_ROUNDS")
if _bcrypt_rounds_env is not None:
    try:
        _rounds = int(_bcrypt_rounds_env)
        if 4 <= _rounds <= 31:
            BCRYPT_ROUNDS = _rounds
        else:
            logger.warning(
                "BCRYPT_ROUNDS must be between 4 and 31; using default %d",
                DEFAULT_BCRYPT_ROUNDS,
            )
            BCRYPT_ROUNDS = DEFAULT_BCRYPT_ROUNDS
    except ValueError:
        logger.warning(
            "BCRYPT_ROUNDS is not an integer; using default %d",
            DEFAULT_BCRYPT_ROUNDS,
        )
        BCRYPT_ROUNDS = DEFAULT_BCRYPT_ROUNDS
else:
    BCRYPT_ROUNDS = DEFAULT_BCRYPT_ROUNDS


def validate_password_complexity(password: str) -> None:
    """Проверяет наличие цифр, верхнего/нижнего регистра и спецсимволов."""
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain an uppercase letter")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain a lowercase letter")
    if not re.search(r"\d", password):
        raise ValueError("Password must contain a digit")
    if not re.search(r"[^\w\s]", password):
        raise ValueError("Password must contain a special character")


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
    rounds = BCRYPT_ROUNDS
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

