#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXCLUSIONS="$(cd "$(dirname "$0")/.." && pwd)/exclusions.txt"
TOOL="$(cd "$(dirname "$0")/.." && pwd)/tools/placeholder_finder.py"
cd "$ROOT"

if [ -f "$TOOL" ]; then
  python3 "$TOOL" . --exclusions="$EXCLUSIONS" --output="placeholder_report.txt"
  echo "Placeholder finder report generated: placeholder_report.txt"
  echo "Summary:"
  head -20 placeholder_report.txt
else
  echo "Placeholder finder tool not found at $TOOL"
  exit 1
fi