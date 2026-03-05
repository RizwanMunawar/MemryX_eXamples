#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

if [[ -d "${VENV_DIR}" ]]; then
  . "${VENV_DIR}/bin/activate"
  cd "${REPO_ROOT}"
  example_launcher
  exit 0
fi

if command -v example_launcher >/dev/null 2>&1; then
  cd "${REPO_ROOT}"
  example_launcher
  exit 0
fi

echo "ERROR: launcher is not installed."
echo "Run: pip install --extra-index-url https://developer.memryx.com/pip -e \".[memryx-sdk]\""
exit 1
