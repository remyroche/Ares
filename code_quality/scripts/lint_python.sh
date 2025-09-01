#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXCLUSIONS="$(dirname "$0")/../exclusions.txt"
cd "$ROOT"

if command -v ruff >/dev/null 2>&1; then
  ruff check . --extend-exclude="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' '|' | sed 's/|$//')" || true
elif command -v flake8 >/dev/null 2>&1; then
  flake8 . --extend-exclude="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' ',')" || true
else
  echo "ruff/flake8 not installed; skipping."
fi

