#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXCLUSIONS="$(dirname "$0")/../exclusions.txt"
cd "$ROOT"

echo "🔧 Running auto-fixes..."

# Python formatting fixes
if command -v black >/dev/null 2>&1; then
  echo "  📝 Formatting Python code with black..."
  black . --extend-exclude="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' '|' | sed 's/|$//')" || true
else
  echo "  ⚠️  black not installed; skipping formatting."
fi

# Python linting fixes
if command -v ruff >/dev/null 2>&1; then
  echo "  🧹 Auto-fixing Python linting issues with ruff..."
  ruff check . --extend-exclude="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' '|' | sed 's/|$//')" --fix || true
  echo "  🔍 Running ruff check again to verify fixes..."
  ruff check . --extend-exclude="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' '|' | sed 's/|$//')" || true
elif command -v autopep8 >/dev/null 2>&1; then
  echo "  🧹 Auto-fixing Python linting issues with autopep8..."
  autopep8 --in-place --recursive --aggressive --aggressive . || true
else
  echo "  ⚠️  ruff/autopep8 not installed; skipping linting fixes."
fi

# Import sorting and organization
if command -v isort >/dev/null 2>&1; then
  echo "  📦 Organizing imports with isort..."
  isort . --skip-glob="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' '|' | sed 's/|$//')" || true
elif command -v ruff >/dev/null 2>&1; then
  echo "  📦 Organizing imports with ruff..."
  ruff check . --extend-exclude="$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' '|' | sed 's/|$//')" --select=I --fix || true
else
  echo "  ⚠️  isort/ruff not installed; skipping import organization."
fi

# JS/TS formatting fixes (if applicable)
if [ -f package.json ] && command -v npx >/dev/null 2>&1; then
  echo "  📝 Formatting JavaScript/TypeScript code with prettier..."
  npx --yes prettier . --write || true
  
  echo "  🧹 Auto-fixing JavaScript/TypeScript linting issues with eslint..."
  npx --yes eslint . --fix || true
else
  echo "  ℹ️  No JS/TS files detected or tools not available; skipping JS/TS fixes."
fi

# Remove trailing whitespace and fix line endings
echo "  🧹 Cleaning up whitespace and line endings..."
find . -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.md" -o -name "*.txt" -o -name "*.sh" | \
  grep -v -E "$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' '|' | sed 's/|$//')" | \
  xargs -I {} sed -i 's/[[:space:]]*$//' {} 2>/dev/null || true

# Fix common Python issues
echo "  🔧 Applying common Python fixes..."
find . -name "*.py" | \
  grep -v -E "$(cat "$EXCLUSIONS" | grep -v '^#' | grep -v '^$' | tr '\n' '|' | sed 's/|$//')" | \
  while read -r file; do
    # Remove unused imports (basic cleanup)
    if command -v autoflake >/dev/null 2>&1; then
      autoflake --in-place --remove-all-unused-imports --remove-unused-variables "$file" 2>/dev/null || true
    fi
  done

echo "✅ Auto-fixes completed!"