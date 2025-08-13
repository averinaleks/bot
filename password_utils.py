import crypt

MAX_PASSWORD_LENGTH = 64

def hash_password(password: str, salt: str) -> str:
    """Hash a password using system crypt with a length limit.

    Raises:
        ValueError: if the password exceeds MAX_PASSWORD_LENGTH.
    """
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password exceeds maximum length")
    return crypt.crypt(password, salt)
