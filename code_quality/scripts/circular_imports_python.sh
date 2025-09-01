#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXCLUSIONS="$(dirname "$0")/../exclusions.txt"
cd "$ROOT"

if command -v pylint >/dev/null 2>&1; then
  EXCLUDE_PATTERNS="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' ',')"
  pylint --disable=all --enable=cyclic-import . --ignore="$EXCLUDE_PATTERNS" || true
else
  echo "pylint not installed; skipping circular import detection."
fi