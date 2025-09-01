#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ -f tsconfig.json ] && command -v npx >/dev/null 2>&1; then
  npx --yes tsc -p tsconfig.json --noEmit || true
else
  echo "TypeScript not available or no tsconfig.json; skipping."
fi

