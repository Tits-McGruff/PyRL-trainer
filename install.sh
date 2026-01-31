#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

# Choose python command
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "python not found in PATH"
  exit 1
fi

# Create venv if missing
if [ ! -x ".venv/bin/python" ]; then
  "$PY" -m venv .venv
fi

# Activate venv
. ".venv/bin/activate"

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  pip install numpy websockets torch

  # Optional, only needed if you run on Python < 3.11 and want TOML parsing
  pip install toml
fi

echo "Done. Virtual environment is in .venv"
