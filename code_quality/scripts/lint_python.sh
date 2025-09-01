#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if command -v ruff >/dev/null 2>&1; then
  ruff check . || true
elif command -v flake8 >/dev/null 2>&1; then
  flake8 . || true
else
  echo "ruff/flake8 not installed; skipping."
fi

