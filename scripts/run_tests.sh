#!/usr/bin/env bash
# Run the full Forge test suite from the repo root on CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
echo "Running Forge tests with $($PYTHON --version 2>&1)..."
"$PYTHON" -m pytest "$@"
