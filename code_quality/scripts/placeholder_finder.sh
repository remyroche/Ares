#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXCLUSIONS="$(cd "$(dirname "$0")/.." && pwd)/exclusions.txt"
TOOL="$(cd "$(dirname "$0")/.." && pwd)/tools/placeholder_finder.py"
cd "$ROOT"

if [ -f "$TOOL" ]; then
  echo "🔍 Running Enhanced Placeholder Finder..."
  echo "📁 Analyzing: $ROOT"
  echo "🚫 Exclusions: $EXCLUSIONS"
  echo ""
  
  # Run the enhanced placeholder finder with both text and JSON output
  python3 "$TOOL" . \
    --exclusions="$EXCLUSIONS" \
    --output="placeholder_report.txt" \
    --json="placeholder_report.json" \
    --verbose
  
  echo ""
  echo "✅ Enhanced Placeholder Finder completed successfully!"
  echo "📄 Text report: placeholder_report.txt"
  echo "🔧 JSON report: placeholder_report.json"
  echo ""
  echo "📊 Summary:"
  head -25 placeholder_report.txt
  echo ""
  echo "💡 Tip: Use '--json' flag for programmatic access to results"
  echo "💡 Tip: Use '--verbose' flag for detailed logging"
else
  echo "❌ Enhanced Placeholder Finder tool not found at $TOOL"
  exit 1
fi