#!/usr/bin/env bash
# Start a LocalMesh AI node on this device (macOS / Linux).
#   ./start.sh                      normal
#   ./start.sh --demo               add simulated devices (for presentations)
#   ./start.sh --token SECRET       require the same secret on every device
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || { echo "Python 3.9+ is required (macOS: xcode-select --install)"; exit 1; }
if [ ! -d .venv ]; then
  echo "→ First run: creating a virtual environment"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q --disable-pip-version-check -r requirements.txt
exec python -m localmesh "$@"
