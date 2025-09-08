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


class AnsiToWin32:  # pragma: no cover - trivial
    """No-op replacement for :class:`colorama.AnsiToWin32`.

    The real class wraps a stream to convert ANSI escape sequences on Windows.
    Our tests only require the attribute to exist so importing libraries such as
    ``click`` succeeds. The implementation here simply stores the provided
    stream and exposes a ``write`` method that forwards to it if present.
    """

    def __init__(self, stream=None, *_, **__):
        self.stream = stream

    def write(self, text):
        if self.stream is not None:
            self.stream.write(text)
        else:  # pragma: no cover - not used
            pass


def init(*args, **kwargs):  # pragma: no cover - trivial
    return None


__all__ = ["Fore", "Style", "AnsiToWin32", "init", "__version__"]
