#!/usr/bin/env bash
set -euo pipefail

# Best-effort installer for optional tools. Skips if unavailable.

ensure() {
  local name="$1"; shift
  if ! command -v "$name" >/dev/null 2>&1; then
    return 1
  fi
}

pip_install() {
  local pkg="$1"
  if python3 -m pip install --upgrade "$pkg" >/dev/null 2>&1; then
    echo "Installed/updated: $pkg"
  else
    echo "[WARN] Failed to install $pkg"
  fi
}

# Python tooling
pip_install ruff
pip_install black
pip_install mypy
pip_install vulture
pip_install radon
pip_install pip-audit
pip_install pydeps
pip_install pipdeptree

# JS tooling (optional, only if npm present and package.json exists)
if command -v npm >/dev/null 2>&1 && [ -f "$(cd "$(dirname "$0")/.." && pwd)/package.json" ]; then
  npm ls --depth=0 >/dev/null 2>&1 || true
  npx -y --yes eslint -v >/dev/null 2>&1 || npm i -D eslint || true
  npx -y --yes prettier -v >/dev/null 2>&1 || npm i -D prettier || true
  npx -y --yes typescript -v >/dev/null 2>&1 || npm i -D typescript || true
  npm audit >/dev/null 2>&1 || true
fi

echo "Bootstrap complete."

