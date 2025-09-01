#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if command -v black >/dev/null 2>&1; then
  black .
else
  echo "black not installed; skipping."
fi

