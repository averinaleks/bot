from __future__ import annotations

from pathlib import Path

from typing import Optional


def _load_skip_flag(config_path: Path) -> bool:
    """Return True if the check should be skipped based on the config."""
    if not config_path.exists():
        return True
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if key.strip() == "skip_gptoss_check":
            return value.strip().lower() in {"1", "true", "yes"}
    return True


def main(config_path: Optional[Path] = None) -> None:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "gptoss_check.config"
    skip = _load_skip_flag(config_path)
    if skip:
        print("GPT-OSS check skipped via configuration")
        return
    print("Running GPT-OSS check...")
    from . import check_code
    check_code.run()
    print("GPT-OSS check completed")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
