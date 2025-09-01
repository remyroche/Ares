#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if command -v vulture >/dev/null 2>&1; then
  vulture . || true
else
  echo "vulture not installed; skipping."
fi

