#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ -f package.json ] && command -v npx >/dev/null 2>&1; then
  npx --yes prettier . --write || true
else
  echo "JS prettier not available; skipping."
fi

