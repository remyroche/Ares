#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/artifacts"
mkdir -p "$OUT_DIR"
cd "$ROOT"

if command -v pydeps >/dev/null 2>&1; then
  # Generate a repo-level dependency graph PNG if possible
  pydeps . --noshow --max-bacon=2 --no-config --pylib-all --externals --output "$OUT_DIR/repo_deps.svg" || true
else
  echo "pydeps not installed; skipping graph generation."
fi

if command -v pipdeptree >/dev/null 2>&1; then
  pipdeptree --warn silence > "$OUT_DIR/pipdeptree.txt" || true
fi

echo "Artifacts in $OUT_DIR"

