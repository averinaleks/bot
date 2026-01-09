from __future__ import annotations

import argparse
import json
import os
import sys
from json import JSONDecodeError
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parent.parent
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

from scripts._filesystem import write_secure_text


DOCKER_CONFIG_ENV = "DOCKER_CONFIG"
DEFAULT_CONFIG_DIR = Path.home() / ".docker"
CONFIG_FILENAME = "config.json"
BACKUP_SUFFIX = "%Y%m%d-%H%M%S"


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc


def _serialize_config(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, indent=2)
    return f"{payload}\n"


def save_config(config_path: Path, data: Dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_secure_text(config_path, _serialize_config(data))


def backup_config(config_path: Path) -> Path | None:
    if not config_path.exists():
        return None
    timestamp = datetime.now().strftime(BACKUP_SUFFIX)
    backup_path = config_path.with_suffix(config_path.suffix + f".backup-{timestamp}")
    payload = config_path.read_text(encoding="utf-8", errors="replace")
    write_secure_text(backup_path, payload)
    return backup_path


def strip_desktop_helpers(data: Dict[str, Any]) -> bool:
    """Remove Windows-only credential helpers from a Docker config.

    The function returns True if the config was changed. It removes:
    - credsStore entries that reference docker-credential-desktop
    - credHelpers entries whose value points to a desktop helper
    - a legacy "credStore" (singular) entry that some guides suggest as a
      workaround; Docker CLI ignores it, but we remove it to avoid confusion.
    """

    changed = False

    creds_store = data.get("credsStore")
    if isinstance(creds_store, str) and "desktop" in creds_store.lower():
        data.pop("credsStore", None)
        changed = True

    legacy_cred_store = data.get("credStore")
    if isinstance(legacy_cred_store, str) and "desktop" in legacy_cred_store.lower():
        data.pop("credStore", None)
        changed = True

    cred_helpers = data.get("credHelpers")
    if isinstance(cred_helpers, dict):
        bad_keys = [key for key, value in cred_helpers.items() if isinstance(value, str) and "desktop" in value.lower()]
        for key in bad_keys:
            cred_helpers.pop(key, None)
            changed = True
        if not cred_helpers:
            data.pop("credHelpers", None)
            changed = changed or bool(bad_keys)

    return changed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove docker-credential-desktop entries from a Docker config. "
            "Useful in WSL where the Windows helper is unavailable."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to the Docker config to rewrite. Defaults to "
            "$DOCKER_CONFIG/config.json or ~/.docker/config.json."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Optional source config to read from. If provided and different from "
            "--config, the sanitized data will be written to --config without "
            "modifying the source file."
        ),
    )
    parser.add_argument(
        "--create-empty",
        action="store_true",
        help=(
            "Create a minimal config when the source file is missing. This is "
            "useful for wrapper scripts that want a clean DOCKER_CONFIG directory."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or [])

    env_config_dir = Path(os.environ.get(DOCKER_CONFIG_ENV, DEFAULT_CONFIG_DIR))
    config_path = args.config if args.config else env_config_dir / CONFIG_FILENAME
    source_path = args.source if args.source else config_path

    try:
        config_data = load_config(source_path)
    except ValueError as exc:
        backup_path = backup_config(source_path)
        save_config(config_path, {})
        if backup_path:
            print(f"Backed up invalid Docker config to {backup_path}.")
        print(f"Invalid Docker config at {source_path}: {exc}")
        print("Rewrote Docker config without credential helpers; rerun your compose command.")
        return

    if not config_data and not args.create_empty:
        print(f"No existing Docker config found at {source_path}; nothing to update.")
        return

    changed = strip_desktop_helpers(config_data)

    if not config_data and args.create_empty:
        config_data = {"auths": {}, "credHelpers": {}}
        changed = True

    if not changed:
        print(f"No desktop credential helper entries found in {source_path}; no changes made.")
        if config_path != source_path and not config_path.exists():
            print(f"Copying existing config from {source_path} to {config_path} without changes.")
            save_config(config_path, config_data)
        return

    if config_path.exists():
        backup_path = backup_config(config_path)
    else:
        backup_path = None

    save_config(config_path, config_data)

    if backup_path:
        print(f"Updated {config_path} and created backup at {backup_path}.")
    else:
        print(f"Updated {config_path} (no previous file to back up).")

    if config_path != source_path and source_path.exists():
        print(f"Source config {source_path} left untouched; sanitized copy written to {config_path}.")

    print("Removed docker-credential-desktop entries; try rerunning your docker compose commands.")


if __name__ == "__main__":
    main(sys.argv[1:])
