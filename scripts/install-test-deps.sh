#!/usr/bin/env sh
set -e
# Install Python packages needed for running the test suite.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
pip install -r "$REPO_ROOT/requirements-cpu.txt"
