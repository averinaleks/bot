#!/usr/bin/env sh
set -e
# Install Python packages needed for running the test suite.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Ensure gymnasium is available before importing model_builder
python -m pip install --no-cache-dir gymnasium==1.2.0
python -m pip install --no-cache-dir -r "$REPO_ROOT/requirements-ci.txt"

# Install GPU requirements only when explicitly requested.
if [ "${INSTALL_GPU_DEPS:-0}" -eq 1 ]; then
  python -m pip install --no-cache-dir -r "$REPO_ROOT/requirements-gpu.txt"
fi
