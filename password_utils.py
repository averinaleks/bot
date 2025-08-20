import bcrypt
import re
import string

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 64
BCRYPT_ROUNDS = 12


def validate_password_complexity(password: str) -> None:
    """Проверяет наличие цифр, верхнего/нижнего регистра и спецсимволов."""
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain an uppercase letter")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain a lowercase letter")
    if not re.search(r"\d", password):
        raise ValueError("Password must contain a digit")
    if not any(ch in string.punctuation for ch in password):
        raise ValueError("Password must contain a special character")


def hash_password(password: str) -> str:
    """Хэширует пароль, используя bcrypt."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError("Password too short")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password exceeds maximum length")
    validate_password_complexity(password)
    return bcrypt.hashpw(
        password.encode(), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    ).decode()


def verify_password(password: str, stored_hash: str) -> bool:
    """Проверяет пароль по сохранённому bcrypt-хэшу."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError("Password too short")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password exceeds maximum length")
    return bcrypt.checkpw(password.encode(), stored_hash.encode())

