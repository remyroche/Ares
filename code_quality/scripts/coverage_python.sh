#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXCLUSIONS="$(dirname "$0")/../exclusions.txt"
cd "$ROOT"

if command -v pytest >/dev/null 2>&1; then
  IGNORE_PATTERNS="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | sed 's/^/--ignore=/' | tr '\n' ' ')"
  if python3 -c "import pytest_cov" >/dev/null 2>&1; then
    pytest -q --maxfail=1 --disable-warnings $IGNORE_PATTERNS --cov=. --cov-report=term-missing || true
  elif command -v coverage >/dev/null 2>&1; then
    coverage run -m pytest -q --maxfail=1 --disable-warnings $IGNORE_PATTERNS || true
    coverage report -m || true
  else
    echo "pytest found but no coverage plugin; install pytest-cov or coverage to enable reports."
    pytest -q --maxfail=1 --disable-warnings $IGNORE_PATTERNS || true
  fi
else
  echo "pytest not installed; skipping coverage."
fi

