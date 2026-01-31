#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo ".venv not found, run install.sh first."
  exit 1
fi

. ".venv/bin/activate"

TRAINER="-m pyrl_trainer"

# If first arg ends with .py, treat it as the trainer script
if [ "${1-}" != "" ]; then
  case "$1" in
    *.py)
      TRAINER="$1"
      shift
      ;;
  esac
fi

python $TRAINER "$@"
