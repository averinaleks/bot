import hashlib
import hmac
import logging
import os
import re

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised indirectly via monkeypatch in tests
    import bcrypt  # type: ignore[import]

    BCRYPT_AVAILABLE = True
except ImportError:  # pragma: no cover - explicitly simulated in tests
    bcrypt = None  # type: ignore[assignment]
    BCRYPT_AVAILABLE = False
    logger.warning(
        "Пакет `bcrypt` недоступен. Включён fallback на PBKDF2 из `hashlib`."
    )

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 64
DEFAULT_BCRYPT_ROUNDS = 12
PBKDF2_PREFIX = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 240_000
PBKDF2_SALT_BYTES = 16
PBKDF2_DIGEST = "sha256"


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
        violations.append("отсутствует символ верхнего регистра")
    if not re.search(r"[a-z]", password):
        violations.append("отсутствует символ нижнего регистра")
    if not re.search(r"\d", password):
        violations.append("отсутствует цифра")
    if not re.search(r"[^\w\s]", password):
        violations.append("отсутствует спецсимвол")
    if violations:
        raise ValueError(
            "Пароль не соответствует требованиям сложности: "
            + ", ".join(violations)
        )


def validate_password_length(password: str) -> None:
    """Проверяет, что длина пароля находится в допустимых пределах."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError("Password too short")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password exceeds maximum length")


def _hash_with_bcrypt(password: str) -> str:
    if not BCRYPT_AVAILABLE or bcrypt is None:
        raise RuntimeError(
            "bcrypt недоступен для хэширования, используйте PBKDF2 fallback."
        )
    rounds = get_bcrypt_rounds()
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=rounds)).decode()


def _hash_with_pbkdf2(password: str) -> str:
    salt = os.urandom(PBKDF2_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        PBKDF2_DIGEST, password.encode(), salt, PBKDF2_ITERATIONS
    )
    return (
        f"{PBKDF2_PREFIX}${PBKDF2_ITERATIONS}${salt.hex()}${derived.hex()}"
    )


def hash_password(password: str) -> str:
    """Хэширует пароль, используя bcrypt или PBKDF2 fallback."""
    validate_password_length(password)
    validate_password_complexity(password)
    if BCRYPT_AVAILABLE and bcrypt is not None:
        return _hash_with_bcrypt(password)
    return _hash_with_pbkdf2(password)


def verify_password(password: str, stored_hash: str) -> bool:
    """Проверяет пароль по сохранённому bcrypt- или PBKDF2-хэшу."""

    try:
        validate_password_length(password)
    except ValueError:
        return False

    if stored_hash.startswith(f"{PBKDF2_PREFIX}$"):
        return _verify_with_pbkdf2(password, stored_hash)

    if stored_hash.startswith("pbkdf2_"):
        raise ValueError("Неподдерживаемый формат PBKDF2-хэша")

    if stored_hash.startswith("$2"):
        if not BCRYPT_AVAILABLE or bcrypt is None:
            raise RuntimeError(
                "bcrypt-хэш не может быть проверен без установленного пакета `bcrypt`."
            )
        try:
            return bcrypt.checkpw(password.encode(), stored_hash.encode())
        except ValueError as exc:
            raise ValueError("Повреждён bcrypt-хэш пароля") from exc

    raise ValueError("Неизвестный формат хэша пароля")


def _verify_with_pbkdf2(password: str, stored_hash: str) -> bool:
    try:
        prefix, iterations_str, salt_hex, derived_hex = stored_hash.split("$")
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError("PBKDF2-хэш имеет неверный формат") from exc

    if prefix != PBKDF2_PREFIX:
        raise ValueError("Неподдерживаемый тип PBKDF2-хэша")

    try:
        iterations = int(iterations_str)
    except ValueError as exc:
        raise ValueError("PBKDF2-хэш содержит некорректное число итераций") from exc

    try:
        salt = bytes.fromhex(salt_hex)
        derived = bytes.fromhex(derived_hex)
    except ValueError as exc:
        raise ValueError("PBKDF2-хэш содержит некорректные шестнадцатеричные данные") from exc

    recalculated = hashlib.pbkdf2_hmac(
        PBKDF2_DIGEST, password.encode(), salt, iterations
    )
    return hmac.compare_digest(recalculated, derived)

