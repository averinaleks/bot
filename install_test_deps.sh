#!/usr/bin/env sh
# Install packages required for running the test suite.
# This script installs everything listed in requirements.txt.

set -e
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt"
