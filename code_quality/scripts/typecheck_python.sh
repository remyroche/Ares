#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if command -v mypy >/dev/null 2>&1; then
  mypy . || true
else
  echo "mypy not installed; skipping."
fi

