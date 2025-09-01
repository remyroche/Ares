#!/bin/bash

# Ares Trading Bot GUI Startup Script
# This script starts both the API server and the frontend

set -euo pipefail

# Resolve script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "🚀 Starting Ares Trading Bot GUI..."

API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
VITE_API_BASE_URL="${VITE_API_BASE_URL:-}"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ and try again."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm and try again."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install frontend dependencies if node_modules doesn't exist
if [ ! -d "${SCRIPT_DIR}/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    (cd "${SCRIPT_DIR}" && npm install)
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install frontend dependencies"
        exit 1
    fi
fi

# Check if required Python packages are installed
echo "🔍 Checking Python dependencies..."
if ! python3 -c "import fastapi, uvicorn, psutil, prometheus_client" 2>/dev/null; then
    echo "📦 Installing Python dependencies (fastapi, uvicorn, psutil, prometheus-client)..."
    # Fallback installs for constrained environments
    pip3 install fastapi uvicorn psutil prometheus-client || pip3 install --break-system-packages fastapi uvicorn psutil prometheus-client || { echo "❌ Failed to install Python dependencies"; exit 1; }
fi

echo "✅ Dependencies check passed"

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down..."
    kill ${API_PID:-} ${FRONTEND_PID:-} 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the API server in the background (run as module from project root)
echo "🔧 Starting API server on port ${API_PORT}..."
(
  cd "${ROOT_DIR}" && API_PORT="${API_PORT}" python3 -m GUI.api_server &
)
API_PID=$!

# Wait a moment for the API server to start
sleep 3

# Check if API server started successfully
if ! curl -s "http://localhost:${API_PORT}" > /dev/null; then
    echo "❌ API server failed to start (port ${API_PORT})"
    kill ${API_PID:-} 2>/dev/null || true
    exit 1
fi

echo "✅ API server started on http://localhost:${API_PORT}"

# Start the frontend in the background
echo "🌐 Starting frontend on port ${FRONTEND_PORT}..."
if [ -n "$VITE_API_BASE_URL" ]; then
  echo "↪ Using VITE_API_BASE_URL=$VITE_API_BASE_URL"
  (cd "${SCRIPT_DIR}" && API_PORT="${API_PORT}" VITE_API_BASE_URL="$VITE_API_BASE_URL" npm run dev -- --port ${FRONTEND_PORT} &)
else
  # Use proxy to API if no explicit base URL
  (cd "${SCRIPT_DIR}" && API_PORT="${API_PORT}" npm run dev -- --port ${FRONTEND_PORT} &)
fi
FRONTEND_PID=$!

# Wait a moment for the frontend to start
sleep 5

# Check if frontend started successfully
if ! curl -s "http://localhost:${FRONTEND_PORT}" > /dev/null; then
    echo "❌ Frontend failed to start (port ${FRONTEND_PORT})"
    kill ${API_PID:-} ${FRONTEND_PID:-} 2>/dev/null || true
    exit 1
fi

echo "✅ Frontend started on http://localhost:${FRONTEND_PORT}"
[ -z "$VITE_API_BASE_URL" ] && echo "🔗 Proxying /api to http://localhost:${API_PORT} via Vite" || true

echo ""
echo "🎉 Ares Trading Bot GUI is now running!"
echo "📊 Dashboard: http://localhost:${FRONTEND_PORT}"
echo "📚 API Docs: http://localhost:${API_PORT}/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to stop the servers
wait