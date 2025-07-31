#!/bin/bash

# Ares Trading Bot GUI Startup Script
# This script starts both the API server and the frontend

echo "🚀 Starting Ares Trading Bot GUI..."

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
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install frontend dependencies"
        exit 1
    fi
fi

# Check if required Python packages are installed
echo "🔍 Checking Python dependencies..."
python3 -c "import fastapi, uvicorn, psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing Python dependencies..."
    pip3 install fastapi uvicorn psutil
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Python dependencies"
        exit 1
    fi
fi

echo "✅ Dependencies check passed"

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down..."
    kill $API_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the API server in the background
echo "🔧 Starting API server..."
python3 api_server.py &
API_PID=$!

# Wait a moment for the API server to start
sleep 3

# Check if API server started successfully
if ! curl -s http://localhost:8000 > /dev/null; then
    echo "❌ API server failed to start"
    kill $API_PID 2>/dev/null
    exit 1
fi

echo "✅ API server started on http://localhost:8000"

# Start the frontend in the background
echo "🌐 Starting frontend..."
npm run dev &
FRONTEND_PID=$!

# Wait a moment for the frontend to start
sleep 5

# Check if frontend started successfully
if ! curl -s http://localhost:3000 > /dev/null; then
    echo "❌ Frontend failed to start"
    kill $API_PID $FRONTEND_PID 2>/dev/null
    exit 1
fi

echo "✅ Frontend started on http://localhost:3000"
echo ""
echo "🎉 Ares Trading Bot GUI is now running!"
echo "📊 Dashboard: http://localhost:3000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to stop the servers
wait 