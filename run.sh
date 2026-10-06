#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8765}"
HOST="${HOST:-127.0.0.1}"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

if [ -d "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--urchade--gliner_multi_pii-v1/snapshots" ]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi

exec .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT"
