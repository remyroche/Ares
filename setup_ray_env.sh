#!/bin/bash

# Ray Environment Setup Script
# This script configures Ray to avoid disk space issues

echo "🔧 Setting up Ray environment..."

# Create a directory for Ray temp files in user's home (more space available)
RAY_TEMP_DIR="${HOME}/ray_temp"
mkdir -p "$RAY_TEMP_DIR"

# Set environment variables
export RAY_TEMP_DIR="$RAY_TEMP_DIR"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=0
export RAY_DISABLE_MEMORY_MONITOR=1

echo "✅ Ray temp directory: $RAY_TEMP_DIR"
echo "✅ Environment variables set"
echo ""
echo "📝 To use this setup, run:"
echo "   source setup_ray_env.sh"
echo "   python your_script.py"
echo ""
echo "🧹 To clean up old Ray sessions:"
echo "   ./cleanup_ray_sessions.sh"
