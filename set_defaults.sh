#!/bin/bash

# Default environment variables for Ares trading system
# Source this file to set default values for debug and data preview

# Debug mode - enabled by default
export DEBUG=true

# Data preview settings - enabled by default with specified values
export ENABLE_DATA_PREVIEW=true
export DATA_PREVIEW_MAX_ROWS=5
export DATA_PREVIEW_MAX_COLS=10
export DATA_PREVIEW_LARGE_THRESHOLD=10000

echo "✅ Default environment variables set:"
echo "   DEBUG=$DEBUG"
echo "   ENABLE_DATA_PREVIEW=$ENABLE_DATA_PREVIEW"
echo "   DATA_PREVIEW_MAX_ROWS=$DATA_PREVIEW_MAX_ROWS"
echo "   DATA_PREVIEW_MAX_COLS=$DATA_PREVIEW_MAX_COLS"
echo "   DATA_PREVIEW_LARGE_THRESHOLD=$DATA_PREVIEW_LARGE_THRESHOLD"