"""Utilities to sanitize values before logging."""

from __future__ import annotations

import logging
import os

from typing import Any

logger = logging.getLogger(__name__)

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


def configure_service_logging() -> None:
    """Configure basic logging for containerized microservices.

    This setup keeps logs flowing to stderr (which Gunicorn redirects into the
    configured ``error.log`` file) while honoring the ``LOG_LEVEL`` environment
    variable and applying a consistent, timestamped format. It also updates any
    existing handlers that may have been installed by the runtime so messages
    remain formatted uniformly.
    """

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelName(level_name)
    invalid_level = isinstance(level, str)
    if invalid_level:
        level = logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(level)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(level)
            try:
                handler.setFormatter(formatter)
            except Exception:
                continue

    if invalid_level:
        logger.warning(
            "Некорректное значение LOG_LEVEL=%s: используется INFO по умолчанию",
            level_name,
        )

