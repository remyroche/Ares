#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXCLUSIONS="$(dirname "$0")/../exclusions.txt"
cd "$ROOT"

if command -v vulture >/dev/null 2>&1; then
  vulture . --exclude="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' ',')" || true
else
  echo "vulture not installed; skipping."
fi

