#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if command -v radon >/dev/null 2>&1; then
  radon cc -s -a . || true
  radon mi . || true
else
  echo "radon not installed; skipping."
fi

