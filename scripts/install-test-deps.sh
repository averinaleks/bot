#!/usr/bin/env sh
set -e
# Install Python packages needed for running the test suite from the pinned
# requirement files generated via `pip-compile --strip-extras` to avoid extras.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Install all requirements from the unified requirements.txt.
python -m pip install -r "$REPO_ROOT/requirements.txt"
python -m pip install flake8
python -m pip install pytest
