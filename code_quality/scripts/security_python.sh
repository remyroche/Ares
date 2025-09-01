#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if command -v bandit >/dev/null 2>&1; then
  bandit -q -r . || true
else
  echo "bandit not installed; skipping."
fi

