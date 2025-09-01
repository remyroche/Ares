#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if command -v detect-secrets >/dev/null 2>&1; then
  detect-secrets scan --all-files || true
else
  echo "detect-secrets not installed; skipping."
fi

