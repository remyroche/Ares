#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXCLUSIONS="$(dirname "$0")/../exclusions.txt"
cd "$ROOT"

if command -v black >/dev/null 2>&1; then
  black . --extend-exclude="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' '|' | sed 's/|$//')"
else
  echo "black not installed; skipping."
fi

