import bcrypt

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 64


def hash_password(password: str) -> str:
    """Хэширует пароль, используя bcrypt."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError("Password too short")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password exceeds maximum length")
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, stored_hash: str) -> bool:
    """Проверяет пароль по сохранённому bcrypt-хэшу."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError("Password too short")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password exceeds maximum length")
    return bcrypt.checkpw(password.encode(), stored_hash.encode())

