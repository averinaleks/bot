"""Тесты для утилит очистки логов."""

from services.logging_utils import sanitize_log_value


def test_sanitize_log_value_replaces_control_characters() -> None:
    """Невидимые символы управления заменяются безопасными эквивалентами."""

    original = "start\nline\twith\x00controls\rend"
    sanitized = sanitize_log_value(original)

    assert sanitized == "start\\nline\\twith?controls\\rend"
