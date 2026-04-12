#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

python3 -m venv "${VENV_DIR}"
. "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel

if ! pip install --extra-index-url https://developer.memryx.com/pip -e "${REPO_ROOT}[memryx-sdk]"; then
  echo "WARNING: memryx-sdk extra install failed; installing launcher dependencies only."
  pip install -e "${REPO_ROOT}"
fi

echo
echo "Environment is ready."
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Launch with:   example_launcher"
