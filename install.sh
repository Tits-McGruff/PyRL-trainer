#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "python not found in PATH"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  "$PY" -m venv .venv
fi

. ".venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  pip install numpy websockets torch
  pip install tomli
fi

echo "Done, venv is .venv"
