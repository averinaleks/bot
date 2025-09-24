"""Minimal bcrypt stub using PBKDF2-HMAC-SHA256 for testing purposes.

This module emulates a subset of the :mod:`bcrypt` API so that the project can
run in environments where the real ``bcrypt`` dependency is unavailable.  The
implementation intentionally prioritises determinism and basic security
properties over perfect compatibility.  It provides ``gensalt``, ``hashpw`` and
``checkpw`` helpers that behave similarly to their counterparts in the genuine
library while relying solely on the Python standard library.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Final

_BCRYPT_PREFIX: Final[bytes] = b"$2b$"
_DEFAULT_ROUNDS: Final[int] = 12
_MIN_ROUNDS: Final[int] = 4
_MAX_ROUNDS: Final[int] = 20  # guard against impractically high iteration counts
_SALT_BYTES: Final[int] = 16
_HASH_DK_LEN: Final[int] = 32


def _ensure_bytes(value: bytes | str, *, name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    raise TypeError(f"{name} must be bytes or str")


def _bcrypt64(data: bytes) -> bytes:
    """Return URL-safe base64 without padding, similar to bcrypt's alphabet."""

    return base64.b64encode(data).rstrip(b"=")


def _bcrypt64_decode(data: bytes) -> bytes:
    padding_length = (4 - len(data) % 4) % 4
    return base64.b64decode(data + b"=" * padding_length)


def _format_rounds(rounds: int) -> bytes:
    return f"{rounds:02d}".encode()


def gensalt(rounds: int | None = None) -> bytes:
    """Generate a bcrypt-like salt string.

    The number of rounds is clamped to keep the resulting PBKDF2 iterations
    within a sane range for a testing stub.
    """

    if rounds is None:
        rounds = _DEFAULT_ROUNDS
    if not isinstance(rounds, int):  # pragma: no cover - parity with real bcrypt
        raise TypeError("rounds must be an integer")

    clamped_rounds = max(_MIN_ROUNDS, min(rounds, _MAX_ROUNDS))
    salt = _bcrypt64(os.urandom(_SALT_BYTES))[:22]
    return _BCRYPT_PREFIX + _format_rounds(clamped_rounds) + b"$" + salt


def _salt_from_hash(value: bytes) -> bytes:
    value = _ensure_bytes(value, name="salt")
    if not value.startswith(_BCRYPT_PREFIX) or len(value) < 29:
        raise ValueError("Invalid bcrypt salt format")
    return value[:29]


def _salt_bytes_from_prefix(prefix: bytes) -> bytes:
    salt_b64 = prefix[-22:]
    return _bcrypt64_decode(salt_b64)


def hashpw(password: bytes | str, salt: bytes | str) -> bytes:
    password_bytes = _ensure_bytes(password, name="password")
    prefix = _salt_from_hash(_ensure_bytes(salt, name="salt"))

    try:
        rounds = int(prefix[4:6])
    except Exception:  # pragma: no cover - defensive fallback
        rounds = _DEFAULT_ROUNDS

    clamped_rounds = max(_MIN_ROUNDS, min(rounds, _MAX_ROUNDS))
    iterations = max(1 << clamped_rounds, 10_000)

    salt_material = _salt_bytes_from_prefix(prefix)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password_bytes, salt_material, iterations, dklen=_HASH_DK_LEN
    )
    digest_b64 = _bcrypt64(digest)
    return prefix + digest_b64


def checkpw(password: bytes | str, hashed: bytes | str) -> bool:
    hashed_bytes = _ensure_bytes(hashed, name="hashed")
    recalculated = hashpw(password, hashed_bytes)
    return hmac.compare_digest(recalculated, hashed_bytes)


__all__ = ["gensalt", "hashpw", "checkpw"]
