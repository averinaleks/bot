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
    digest = hashlib.sha256(password + prefix).hexdigest().encode()
    return prefix + digest

def checkpw(password: bytes, hashed: bytes) -> bool:
    if isinstance(hashed, str):
        hashed = hashed.encode()
    salt = _salt_from_hash(hashed)
    return hashpw(password, salt) == hashed

__all__ = ["gensalt", "hashpw", "checkpw"]
