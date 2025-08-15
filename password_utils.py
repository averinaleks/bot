import base64
import hashlib
import hmac
import os

MAX_PASSWORD_LENGTH = 64
SALT_SIZE = 16
PBKDF2_ITERATIONS = 100_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random salt.

    Raises:
        ValueError: if the password exceeds MAX_PASSWORD_LENGTH.

    Returns:
        str: base64 encoded salt and hash.
    """
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password exceeds maximum length")
    if salt is None:
        salt = os.urandom(SALT_SIZE)
    if isinstance(salt, str):
        salt = salt.encode()
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return base64.b64encode(salt + dk).decode()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash using constant-time comparison."""
    data = base64.b64decode(stored_hash)
    salt, hashed = data[:SALT_SIZE], data[SALT_SIZE:]
    new_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return hmac.compare_digest(hashed, new_hash)
