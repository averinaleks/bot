"""Minimal colorama stub for tests."""

class _Ansi:
    def __getattr__(self, name):  # pragma: no cover - trivial
        return ""

Fore = _Ansi()
Style = _Ansi()

def init(*args, **kwargs):  # pragma: no cover - trivial
    return None

__all__ = ["Fore", "Style", "init"]
