from __future__ import annotations

import json
import os
from json import JSONDecodeError
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


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


def save_config(config_path: Path, data: Dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
        fp.write("\n")


def backup_config(config_path: Path) -> Path | None:
    if not config_path.exists():
        return None
    timestamp = datetime.now().strftime(BACKUP_SUFFIX)
    backup_path = config_path.with_suffix(config_path.suffix + f".backup-{timestamp}")
    backup_path.write_bytes(config_path.read_bytes())
    return backup_path


def strip_desktop_helpers(data: Dict[str, Any]) -> bool:
    changed = False

    creds_store = data.get("credsStore")
    if isinstance(creds_store, str) and "desktop" in creds_store.lower():
        data.pop("credsStore", None)
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


def main() -> None:
    config_dir = Path(os.environ.get(DOCKER_CONFIG_ENV, DEFAULT_CONFIG_DIR))
    config_path = config_dir / CONFIG_FILENAME

    try:
        config_data = load_config(config_path)
    except ValueError as exc:
        backup_path = backup_config(config_path)
        save_config(config_path, {})
        if backup_path:
            print(f"Backed up invalid Docker config to {backup_path}.")
        print(f"Invalid Docker config at {config_path}: {exc}")
        print("Rewrote Docker config without credential helpers; rerun your compose command.")
        return

    if not config_data:
        print(f"No existing Docker config found at {config_path}; nothing to update.")
        return

    changed = strip_desktop_helpers(config_data)
    if not changed:
        print(f"No desktop credential helper entries found in {config_path}; no changes made.")
        return

    backup_path = backup_config(config_path)
    save_config(config_path, config_data)

    if backup_path:
        print(f"Updated {config_path} and created backup at {backup_path}.")
    else:
        print(f"Updated {config_path} (no previous file to back up).")

    print("Removed docker-credential-desktop entries; try rerunning your docker compose commands.")


if __name__ == "__main__":
    main()
