#!/usr/bin/env bash
# Render start script: install deps, build frontend, start single process.
set -euo pipefail
cd "$(dirname "$0")"

pip install --no-cache-dir -e .

cd frontend
npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund
npm run build
cd ..

python bootstrap_models.py || echo "[start] bootstrap failed; continuing (app will show empty registry)"

exec uvicorn deploy_server:app --host 0.0.0.0 --port "${PORT:-8000}"
