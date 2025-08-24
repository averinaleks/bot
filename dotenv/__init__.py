"""Minimal stub of ``python-dotenv`` for tests.

The real project exposes utilities like :func:`load_dotenv`,
:func:`find_dotenv` and :func:`dotenv_values`.  The tests in this
repository only need these symbols to exist; they do not actually load
any environment files.  This module therefore provides no-op
implementations that satisfy imports without performing any I/O.
"""

from __future__ import annotations

from typing import Mapping


def dotenv_values(*args, **kwargs) -> Mapping[str, str]:
    """Return an empty mapping of environment variables."""

    return {}


def load_dotenv(*args, **kwargs) -> bool:
    """Pretend to load environment variables and return ``True``.

    The actual ``python-dotenv`` returns whether a dotenv file was found.
    For our purposes the distinction is irrelevant, so the function simply
    reports success without reading anything from disk.
    """

    return True


def find_dotenv(*args, **kwargs) -> str:
    """Return an empty string indicating that no dotenv file was found.

    Flask's CLI calls this function and expects a string path.  Returning an
    empty string signals that there is no file to load, which is adequate for
    the tests here.
    """

    return ""



