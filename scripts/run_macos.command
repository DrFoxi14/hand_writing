#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[training,test]'
exec hie-server --host 127.0.0.1 --port 8000
