#!/usr/bin/env bash

set -euo pipefail

# Run docker compose with a local Docker config that does not rely on
# Windows-only credential helpers. This avoids the
# `docker-credential-desktop.exe` error inside WSL.

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DOCKER_CONFIG="$ROOT_DIR/docker/config-wsl"

CONFIG_FILE="$DOCKER_CONFIG/config.json"
SOURCE_CONFIG="$HOME/.docker/config.json"

python "$ROOT_DIR/scripts/disable_desktop_credential_helper.py" \
  --config "$CONFIG_FILE" \
  --source "$SOURCE_CONFIG" \
  --create-empty

exec docker compose "$@"
