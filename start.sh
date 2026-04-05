#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the Discord Council Orchestrator project.
# This script creates a Python virtual environment, installs build tools,
# installs the project in editable mode, starts Docker sandboxing via Colima,
# and runs the bot. On exit, it tears down all Docker resources.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# --- Docker sandbox lifecycle ---
DOCKER_STARTED_BY_SCRIPT=false

cleanup_docker() {
  echo ""
  echo "Shutting down Docker sandbox..."

  # Kill any running containers spawned by the sandbox tools
  if [ "$DOCKER_STARTED_BY_SCRIPT" = true ]; then
    docker ps -q --filter "ancestor=python:3.11-alpine" --filter "ancestor=node:20-alpine" 2>/dev/null \
      | xargs -r docker stop 2>/dev/null || true
    docker ps -aq --filter "ancestor=python:3.11-alpine" --filter "ancestor=node:20-alpine" 2>/dev/null \
      | xargs -r docker rm -f 2>/dev/null || true

    # Stop the Colima VM
    echo "Stopping Colima VM..."
    colima stop 2>/dev/null || true
  fi

  echo "Docker sandbox cleaned up."
}

trap cleanup_docker EXIT

echo "Checking Docker sandbox (Colima)..."
if command -v colima >/dev/null 2>&1 && command -v docker >/dev/null 2>&1; then
  if ! colima status >/dev/null 2>&1; then
    echo "Starting Colima VM for sandboxed code execution..."
    colima start --cpu 2 --memory 2 --disk 10 2>&1 | grep -v "^\[" || true
    DOCKER_STARTED_BY_SCRIPT=true
    echo "Docker sandbox ready."
  else
    echo "Colima is already running. Using existing instance."
  fi

  # Pre-pull sandbox images if missing
  docker image inspect python:3.11-alpine >/dev/null 2>&1 || docker pull python:3.11-alpine >/dev/null 2>&1 || true
  docker image inspect node:20-alpine >/dev/null 2>&1 || docker pull node:20-alpine >/dev/null 2>&1 || true
else
  echo "WARNING: Colima or Docker not found. Sandboxed code execution tools will be disabled."
fi

# --- Python bootstrap ---
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
