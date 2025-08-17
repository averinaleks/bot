#!/usr/bin/env sh
set -e
# Install Python packages needed for running the test suite from the pinned
# requirement files generated via `pip-compile`.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Install core requirements by default. Pass --gpu to include GPU packages.
if [ "$1" = "--gpu" ]; then
    python -m pip install -r "$REPO_ROOT/requirements-core.txt" -r "$REPO_ROOT/requirements-gpu.txt"
elif [ -z "$1" ]; then
    python -m pip install -r "$REPO_ROOT/requirements-core.txt"
else
    echo "Usage: $0 [--gpu]" >&2
    exit 1
fi
