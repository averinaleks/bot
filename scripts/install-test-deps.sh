#!/usr/bin/env sh
set -e
# Install Python packages needed for running the test suite from the pinned
# requirement files generated via `pip-compile`.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Default to the lightweight CPU requirements. Pass --full to install the
# complete list from requirements.txt (includes GPU packages).
REQ_FILE="requirements-cpu.txt"
if [ "$1" = "--full" ]; then
    REQ_FILE="requirements.txt"
elif [ -n "$1" ]; then
    echo "Usage: $0 [--full]" >&2
    exit 1
fi

python -m pip install -r "$REPO_ROOT/$REQ_FILE"
