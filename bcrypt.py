"""Minimal bcrypt stub using SHA256 for tests."""

from __future__ import annotations

import base64
import hashlib
import os

def gensalt(rounds: int = 12) -> bytes:
    prefix = f"$2b${rounds:02d}$".encode()
    salt = base64.b64encode(os.urandom(16))[:22]
    return prefix + salt

def _salt_from_hash(hashed: bytes) -> bytes:
    return hashed[:29]

def hashpw(password: bytes, salt: bytes) -> bytes:
    if isinstance(password, str):
        password = password.encode()
    if isinstance(salt, str):
        salt = salt.encode()
    prefix = _salt_from_hash(salt)
    # Use PBKDF2-HMAC-SHA256 for computational expense instead of SHA256 directly
    # Use the 'rounds' parameter encoded in the salt to determine iterations
    try:
        rounds = int(prefix[4:6])
    except Exception:
        rounds = 12
    # NIST recommends at least 10,000 iterations, bcrypt default work factor is 12 (2**12 = 4096), so scale as 2**rounds
    iterations = 2 ** rounds
    # Use prefix as salt, lengthen digest for demonstration only (bcrypt's output is 31 bytes; pbkdf2 defaults to 32)
    digest = hashlib.pbkdf2_hmac('sha256', password, prefix, iterations, dklen=32)
    # Base64 encode digest to get ASCII (bcrypt uses a custom alphabet in reality, but for a stub, std base64 is okay)
    digest_b64 = base64.b64encode(digest)[:43]  # bcrypt hashes are 31 chars base64 for the hash part, but use 43 here for compatibility
    return prefix + digest_b64

def checkpw(password: bytes, hashed: bytes) -> bool:
    if isinstance(hashed, str):
        hashed = hashed.encode()
    salt = _salt_from_hash(hashed)
    return hashpw(password, salt) == hashed

__all__ = ["gensalt", "hashpw", "checkpw"]
