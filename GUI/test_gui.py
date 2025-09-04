#!/usr/bin/env python3
"""
GUI Test Script

This script tests the GUI functionality by starting the API server
and checking if all endpoints are working correctly.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:8000"
    
    endpoints = [
        "/",
        "/api/dashboard-data",
        "/api/kill-switch/status",
        "/api/system/status",
        "/api/launcher/status",
        "/api/training/modes",
        "/api/training/status",
        "/api/data/status",
        "/api/tokens",
        "/api/models/available",
        "/api/monitoring/dashboard"
    ]
    
    print("🧪 Testing API endpoints...")
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint} - OK")
            else:
                print(f"❌ {endpoint} - Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint} - Error: {e}")
    
    print("\n🧪 Testing POST endpoints...")
    
    # Test launcher start
    try:
        response = requests.post(
            f"{base_url}/api/launcher/start",
            json={"mode": "blank", "symbol": "ETHUSDT", "exchange": "BINANCE"},
            timeout=5
        )
        if response.status_code == 200:
            print("✅ /api/launcher/start - OK")
        else:
            print(f"❌ /api/launcher/start - Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ /api/launcher/start - Error: {e}")
    
    # Test training start
    try:
        response = requests.post(
            f"{base_url}/api/training/start",
            json={"mode": "blank", "symbol": "ETHUSDT", "exchange": "BINANCE"},
            timeout=5
        )
        if response.status_code == 200:
            print("✅ /api/training/start - OK")
        else:
            print(f"❌ /api/training/start - Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ /api/training/start - Error: {e}")

def test_launcher_integration():
    """Test launcher integration"""
    print("\n🧪 Testing launcher integration...")
    
    try:
        # Import and test launcher integration
        sys.path.insert(0, str(Path(__file__).parent))
        from launcher_integration import (
            get_available_modes, get_available_training_modes, 
            get_available_exchanges, launcher_integration
        )
        
        print(f"✅ Available modes: {get_available_modes()}")
        print(f"✅ Available training modes: {get_available_training_modes()}")
        print(f"✅ Available exchanges: {get_available_exchanges()}")
        print(f"✅ Launcher exists: {launcher_integration.validate_launcher_exists()}")
        
    except ImportError as e:
        print(f"❌ Launcher integration import failed: {e}")
    except Exception as e:
        print(f"❌ Launcher integration test failed: {e}")

def check_dependencies():
    """Check if all dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check Python packages
    python_packages = [
        "fastapi", "uvicorn", "psutil", "prometheus_client", "requests"
    ]
    
    for package in python_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
    
    # Check Node.js packages
    gui_dir = Path(__file__).parent
    package_json = gui_dir / "package.json"
    node_modules = gui_dir / "node_modules"
    
    if package_json.exists():
        print("✅ package.json exists")
    else:
        print("❌ package.json not found")
    
    if node_modules.exists():
        print("✅ node_modules exists")
    else:
        print("❌ node_modules not found - run 'npm install' in GUI directory")

def main():
    """Main test function"""
    print("🚀 Ares GUI Test Suite")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Test launcher integration
    test_launcher_integration()
    
    # Test API endpoints (requires server to be running)
    print("\n🌐 Testing API endpoints...")
    print("Note: This requires the API server to be running on localhost:8000")
    print("Start the server with: python GUI/api_server.py")
    
    try:
        test_api_endpoints()
    except Exception as e:
        print(f"❌ API test failed: {e}")
        print("Make sure the API server is running first")
    
    print("\n✅ Test suite completed!")
    print("\nTo start the GUI:")
    print("1. Start API server: python GUI/api_server.py")
    print("2. Start frontend: cd GUI && npm run dev")
    print("3. Or use the unified script: bash GUI/start.sh")

if __name__ == "__main__":
    main()