"""Utilities to sanitize values before logging."""

from __future__ import annotations

from typing import Any

_CONTROL_MAP: dict[int, str] = {
    ord("\n"): "\\n",
    ord("\r"): "\\r",
    ord("\t"): "\\t",
}
for _code_point in range(32):
    if _code_point in (10, 13, 9):  # already handled newlines, carriage return, tab
        continue
    _CONTROL_MAP[_code_point] = "?"
_CONTROL_MAP[0x7F] = "?"


def sanitize_log_value(value: Any) -> str:
    """Return a printable representation safe for log output.

    The function replaces newline and carriage return characters with their
    escaped counterparts (``"\\n"`` and ``"\\r"``) and substitutes other control
    characters with ``"?"``. This prevents log injection attacks where an
    attacker could smuggle extra log lines or terminal escape codes by
    providing crafted input.
    """

    text = str(value)
    return text.translate(_CONTROL_MAP)

