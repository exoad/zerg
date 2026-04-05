#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the Discord Council Orchestrator project.
# This script creates a Python virtual environment, installs build tools,
# and installs the project in editable mode.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
fi

if [ -n "${PYTHON}" ]; then
  PYTHON_VERSION=$($PYTHON -c 'import sys; print(sys.version_info[:3])')
  if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' >/dev/null 2>&1; then
    PYTHON=python3
  else
    PYTHON=""
  fi
fi

if [ -z "${PYTHON}" ] && [ -x "/opt/homebrew/bin/python3" ]; then
  if /opt/homebrew/bin/python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' >/dev/null 2>&1; then
    PYTHON="/opt/homebrew/bin/python3"
  fi
fi

if [ -z "${PYTHON}" ]; then
  echo "ERROR: Python 3.11+ is required. Install it and retry."
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment with ${PYTHON}..."
  "$PYTHON" -m venv venv
fi

echo "Activating virtual environment..."
# shellcheck source=/dev/null
source venv/bin/activate

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing project dependencies..."
pip install -e .

echo "Bootstrap complete. Starting the bot now..."
exec python -m src.app.main
