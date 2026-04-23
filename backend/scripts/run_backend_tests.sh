#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BACKEND_ROOT}"

if command -v uv >/dev/null 2>&1; then
  uv run --extra dev pytest tests/ -q
  uv run python scripts/smoke_api.py
else
  python -m pytest tests/ -q
  python scripts/smoke_api.py
fi

echo "Backend tests and smoke checks passed."
