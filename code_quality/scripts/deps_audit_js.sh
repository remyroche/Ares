#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ -f package.json ] && command -v npm >/dev/null 2>&1; then
  npm audit || true
else
  echo "npm audit not available; skipping."
fi

