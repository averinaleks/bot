#!/usr/bin/env sh
set -e
# Install only the packages required for running the unit tests on a CPU.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python -m pip install -r "$REPO_ROOT/requirements-cpu.txt"
