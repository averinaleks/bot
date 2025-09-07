"""Minimal colorama stub for tests."""

# Provide a version attribute so libraries that perform version checks (e.g.
# `numba`) treat this stub as sufficiently recent.  The value matches a widely
# used colorama release.
__version__ = "0.4.6"


class _Ansi:
    def __getattr__(self, name):  # pragma: no cover - trivial
        return ""


Fore = _Ansi()
Style = _Ansi()


def init(*args, **kwargs):  # pragma: no cover - trivial
    return None


__all__ = ["Fore", "Style", "init", "__version__"]
