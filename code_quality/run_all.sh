#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running Code Quality Suite in: $REPO_ROOT"

run() {
  local label="$1"; shift
  echo "\n=== $label ==="
  if ! "$@"; then
    echo "[WARN] $label failed"
  fi
}

run "Bootstrap tools" "$SCRIPT_DIR/bootstrap_tools.sh"
run "Formatting (Python)" "$SCRIPT_DIR/scripts/format_python.sh"
run "Linting (Python)" "$SCRIPT_DIR/scripts/lint_python.sh"
run "Type Check (Python)" "$SCRIPT_DIR/scripts/typecheck_python.sh"
run "Dead Code (Python)" "$SCRIPT_DIR/scripts/deadcode_python.sh"
run "Circular Imports (Python)" "$SCRIPT_DIR/scripts/circular_imports_python.sh"
run "Complexity (Python)" "$SCRIPT_DIR/scripts/complexity_python.sh"
run "Dependency Audit (Python)" "$SCRIPT_DIR/scripts/deps_audit_python.sh"
run "Dependency Map (Python)" "$SCRIPT_DIR/scripts/deps_map_python.sh"
run "Test Coverage (Python)" "$SCRIPT_DIR/scripts/coverage_python.sh"

# Optional JS/TS
if [ -f "$REPO_ROOT/package.json" ]; then
  run "Formatting (JS)" "$SCRIPT_DIR/scripts/format_js.sh"
  run "Linting (JS)" "$SCRIPT_DIR/scripts/lint_js.sh"
  run "Type Check (TS)" "$SCRIPT_DIR/scripts/typecheck_js.sh"
  run "Deps Audit (JS)" "$SCRIPT_DIR/scripts/deps_audit_js.sh"
fi

echo "\nAll checks completed. Review warnings above."

