#!/usr/bin/env sh
set -e
# Install packages required for running the unit tests.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python -m pip install --upgrade 'pip>=24.0' 'setuptools>=78.1.1,<81' wheel
python -m pip install -r "$REPO_ROOT/requirements.txt"
