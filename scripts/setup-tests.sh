#!/usr/bin/env sh
set -e
# Install packages required for running the unit tests.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python -m pip install -r "$REPO_ROOT/requirements.txt"
