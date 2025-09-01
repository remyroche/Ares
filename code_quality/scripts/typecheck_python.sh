#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXCLUSIONS="$(dirname "$0")/../exclusions.txt"
cd "$ROOT"

if command -v mypy >/dev/null 2>&1; then
  mypy . --exclude="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' '|' | sed 's/|$//')" || true
else
  echo "mypy not installed; skipping."
fi

