"""Minimal stub of :mod:`python-dotenv` for tests.

The integration tests spawn subprocesses that import ``python-dotenv`` to look
for ``.env`` files. The real library exposes both :func:`load_dotenv` and
``find_dotenv``; our previous stub only implemented ``load_dotenv`` which led to
an ``AttributeError`` when the child processes attempted to call
``dotenv.find_dotenv``. The additional ``find_dotenv`` definition below returns
an empty string, which is sufficient for the tests that merely ensure the
function exists.
"""


def load_dotenv(*args, **kwargs):  # pragma: no cover - trivial
    """Pretend to load environment variables from a ``.env`` file."""
    return True


def find_dotenv(*args, **kwargs):  # pragma: no cover - trivial
    """Return an empty path, indicating no ``.env`` file was found."""
    return ""


__all__ = ["load_dotenv", "find_dotenv"]
