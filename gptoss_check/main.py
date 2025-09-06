from __future__ import annotations

from pathlib import Path
import os
import logging

from typing import Optional


logger = logging.getLogger(__name__)


def _load_skip_flag(config_path: Path) -> bool:
    """Return True if the check should be skipped based on the config."""
    if not config_path.exists():
        logger.warning("Файл конфигурации %s не найден, проверка запущена", config_path)
        return False
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        if key == "skip_gptoss_check":
            # allow inline comments after the value, e.g. "true  # comment"
            value = value.split("#", 1)[0].strip().lower()
            return value in {"1", "true", "yes"}
    # By default run the check when the flag is absent
    return False


def main(config_path: Optional[Path] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "gptoss_check.config"
    skip = _load_skip_flag(config_path)
    if skip:
        logger.info("GPT-OSS check skipped via configuration")
        return
    api_url = os.getenv("GPT_OSS_API")
    if not api_url:
        logger.warning("Переменная окружения GPT_OSS_API не установлена, проверка пропущена")
        return
    try:
        from . import check_code  # package execution
    except ImportError:  # script execution
        import check_code  # type: ignore
    logger.info("Running GPT-OSS check...")
    check_code.run()
    logger.info("GPT-OSS check completed")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
