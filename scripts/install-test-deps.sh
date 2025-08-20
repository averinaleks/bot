#!/usr/bin/env sh
set -e
# Install Python packages needed for running the test suite.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"


# Install GPU requirements only when explicitly requested.
if [ "${INSTALL_GPU_DEPS:-0}" -eq 1 ]; then
  python -m pip install --no-cache-dir -r "$REPO_ROOT/requirements-gpu.txt"
fi
