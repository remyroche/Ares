#!/bin/bash

# Ray Session Cleanup Script
# This script cleans up old Ray sessions to free up disk space

echo "🧹 Cleaning up old Ray sessions..."

# Default Ray temp directory
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray}"

if [ -d "$RAY_TEMP_DIR" ]; then
    echo "📁 Checking Ray temp directory: $RAY_TEMP_DIR"

    # Count sessions before cleanup
    SESSION_COUNT=$(find "$RAY_TEMP_DIR" -name "session_*" -type d | wc -l)
    echo "📊 Found $SESSION_COUNT Ray sessions"

    if [ "$SESSION_COUNT" -gt 1 ]; then
        # Keep the latest session, remove older ones
        find "$RAY_TEMP_DIR" -name "session_*" -type d -not -name "$(basename $(readlink $RAY_TEMP_DIR/session_latest 2>/dev/null || echo "session_latest"))" -exec rm -rf {} + 2>/dev/null
        echo "✅ Cleaned up old Ray sessions"
    else
        echo "ℹ️  Only one Ray session found, nothing to clean up"
    fi
else
    echo "⚠️  Ray temp directory not found: $RAY_TEMP_DIR"
fi

# Also check and clean /tmp/ray if it exists and is different
if [ "$RAY_TEMP_DIR" != "/tmp/ray" ] && [ -d "/tmp/ray" ]; then
    echo "🧽 Also cleaning /tmp/ray..."
    find /tmp/ray -name "session_*" -type d -mtime +1 -exec rm -rf {} + 2>/dev/null
    echo "✅ Cleaned up old sessions in /tmp/ray"
fi

echo "🎉 Cleanup complete!"
