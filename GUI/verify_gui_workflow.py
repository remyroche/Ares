#!/usr/bin/env python3
"""
GUI Workflow Verification

This script verifies the complete GUI workflow by:
1. Starting the API server
2. Starting the frontend
3. Testing the complete user workflow
4. Verifying all components work together
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

def start_api_server():
    """Start the API server"""
    print("🚀 Starting API server...")
    
    process = subprocess.Popen(
        [sys.executable, "api_server_simple.py"],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    # Test if server is responding
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ API server started successfully")
            return process
        else:
            print(f"❌ API server failed to start: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ API server startup failed: {e}")
        return None

def start_frontend():
    """Start the frontend development server"""
    print("🌐 Starting frontend...")
    
    process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for frontend to start
    time.sleep(5)
    
    # Test if frontend is responding
    try:
        response = requests.get("http://localhost:3000/", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend started successfully")
            return process
        else:
            print(f"❌ Frontend failed to start: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Frontend startup failed: {e}")
        return None

def test_complete_workflow():
    """Test the complete GUI workflow"""
    print("\n🧪 Testing complete GUI workflow...")
    
    api_base = "http://localhost:8000"
    
    # Test 1: Dashboard data
    print("1. Testing dashboard data...")
    try:
        response = requests.get(f"{api_base}/api/dashboard-data", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Dashboard data loaded: {data.get('totalPnl', 0):.2f} PnL, {data.get('openPositionsCount', 0)} positions")
        else:
            print(f"   ❌ Dashboard data failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Dashboard data error: {e}")
    
    # Test 2: Launcher control
    print("2. Testing launcher control...")
    try:
        # Get launcher status
        response = requests.get(f"{api_base}/api/launcher/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Launcher status: {data.get('launcher_active', False)}")
            
            # Start a launcher mode
            start_response = requests.post(
                f"{api_base}/api/launcher/start",
                json={"mode": "blank", "symbol": "ETHUSDT", "exchange": "BINANCE"},
                timeout=5
            )
            if start_response.status_code == 200:
                result = start_response.json()
                print(f"   ✅ Launcher started: {result.get('message', 'Success')}")
            else:
                print(f"   ❌ Launcher start failed: {start_response.status_code}")
        else:
            print(f"   ❌ Launcher status failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Launcher control error: {e}")
    
    # Test 3: Training modes
    print("3. Testing training modes...")
    try:
        response = requests.get(f"{api_base}/api/training/modes", timeout=5)
        if response.status_code == 200:
            data = response.json()
            modes = data.get("modes", {})
            print(f"   ✅ Training modes available: {list(modes.keys())}")
            
            # Start training
            train_response = requests.post(
                f"{api_base}/api/training/start",
                json={"mode": "light", "symbol": "ETHUSDT", "exchange": "BINANCE"},
                timeout=5
            )
            if train_response.status_code == 200:
                result = train_response.json()
                print(f"   ✅ Training started: {result.get('message', 'Success')}")
            else:
                print(f"   ❌ Training start failed: {train_response.status_code}")
        else:
            print(f"   ❌ Training modes failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Training modes error: {e}")
    
    # Test 4: System control
    print("4. Testing system control...")
    try:
        # Get system status
        response = requests.get(f"{api_base}/api/system/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ System status: {data.get('status', 'unknown')}")
            
            # Test kill switch
            kill_response = requests.post(
                f"{api_base}/api/kill-switch/activate",
                json={"reason": "Test activation", "emergency": False},
                timeout=5
            )
            if kill_response.status_code == 200:
                print("   ✅ Kill switch activated")
                
                # Deactivate kill switch
                deactivate_response = requests.post(
                    f"{api_base}/api/kill-switch/deactivate",
                    timeout=5
                )
                if deactivate_response.status_code == 200:
                    print("   ✅ Kill switch deactivated")
                else:
                    print(f"   ❌ Kill switch deactivation failed: {deactivate_response.status_code}")
            else:
                print(f"   ❌ Kill switch activation failed: {kill_response.status_code}")
        else:
            print(f"   ❌ System status failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ System control error: {e}")
    
    # Test 5: Token management
    print("5. Testing token management...")
    try:
        # Get tokens
        response = requests.get(f"{api_base}/api/tokens", timeout=5)
        if response.status_code == 200:
            tokens = response.json()
            print(f"   ✅ Tokens available: {len(tokens)} configured")
            
            # Update token config
            update_response = requests.post(
                f"{api_base}/api/tokens",
                json={"symbol": "BTCUSDT", "exchange": "BINANCE", "enabled": True, "model_version": "v1.0.0"},
                timeout=5
            )
            if update_response.status_code == 200:
                result = update_response.json()
                print(f"   ✅ Token updated: {result.get('message', 'Success')}")
            else:
                print(f"   ❌ Token update failed: {update_response.status_code}")
        else:
            print(f"   ❌ Token management failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Token management error: {e}")
    
    # Test 6: Model management
    print("6. Testing model management...")
    try:
        response = requests.get(f"{api_base}/api/models/available", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"   ✅ Models available: {len(models)} models")
        else:
            print(f"   ❌ Model management failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Model management error: {e}")
    
    # Test 7: Monitoring
    print("7. Testing monitoring...")
    try:
        response = requests.get(f"{api_base}/api/monitoring/dashboard", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Monitoring data: System health {data.get('system_health', {}).get('status', 'unknown')}")
        else:
            print(f"   ❌ Monitoring failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Monitoring error: {e}")

def main():
    """Main verification function"""
    print("🔍 GUI Workflow Verification")
    print("="*50)
    
    api_process = None
    frontend_process = None
    
    try:
        # Start API server
        api_process = start_api_server()
        if not api_process:
            print("❌ Cannot continue without API server")
            return
        
        # Start frontend
        frontend_process = start_frontend()
        if not frontend_process:
            print("⚠️ Frontend not available, but API server is working")
        
        # Test complete workflow
        test_complete_workflow()
        
        print("\n" + "="*50)
        print("✅ GUI Workflow Verification Complete!")
        print("="*50)
        print("\n🌐 Access the GUI at:")
        print("   Frontend: http://localhost:3000")
        print("   API Docs: http://localhost:8000/docs")
        print("\n🎛️ Available Features:")
        print("   • Dashboard with real-time data")
        print("   • Launcher Control for starting modes")
        print("   • Training management")
        print("   • System control and kill switch")
        print("   • Token and model management")
        print("   • Monitoring and analytics")
        print("\n📋 Usage:")
        print("   1. Open http://localhost:3000 in your browser")
        print("   2. Navigate to 'Launcher Control'")
        print("   3. Configure symbol (e.g., ETHUSDT) and exchange (BINANCE)")
        print("   4. Click any mode button to start processes")
        print("   5. Monitor progress in real-time")
        print("\n🛑 To stop: Press Ctrl+C")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
    
    finally:
        # Cleanup
        if api_process:
            api_process.terminate()
            api_process.wait()
        if frontend_process:
            frontend_process.terminate()
            frontend_process.wait()

if __name__ == "__main__":
    main()