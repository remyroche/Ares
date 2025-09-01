#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if command -v pip-audit >/dev/null 2>&1; then
  pip-audit || true
else
  echo "pip-audit not installed; skipping."
fi

