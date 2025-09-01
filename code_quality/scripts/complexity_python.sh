#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXCLUSIONS="$(dirname "$0")/../exclusions.txt"
cd "$ROOT"

if command -v radon >/dev/null 2>&1; then
  EXCLUDE_PATTERNS="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' ',')"
  radon cc -s -a . --exclude="$EXCLUDE_PATTERNS" || true
  radon mi . --exclude="$EXCLUDE_PATTERNS" || true
else
  echo "radon not installed; skipping."
fi

