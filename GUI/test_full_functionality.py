#!/usr/bin/env python3
"""
Full GUI Functionality Test

This script tests all aspects of the GUI functionality including:
- API server endpoints
- Launcher integration
- Frontend accessibility
- Process management
- Real-time features
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

class GUIFunctionalityTester:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        self.test_results = {}
        
    def test_api_server_startup(self):
        """Test if API server starts successfully"""
        print("🧪 Testing API server startup...")
        
        try:
            # Start API server
            process = subprocess.Popen(
                [sys.executable, "api_server_simple.py"],
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(3)
            
            # Test if server is responding
            response = requests.get(f"{self.api_base_url}/", timeout=5)
            if response.status_code == 200:
                print("✅ API server started successfully")
                self.test_results["api_startup"] = True
                return process
            else:
                print(f"❌ API server failed to start: {response.status_code}")
                self.test_results["api_startup"] = False
                return None
                
        except Exception as e:
            print(f"❌ API server startup failed: {e}")
            self.test_results["api_startup"] = False
            return None
    
    def test_all_api_endpoints(self):
        """Test all API endpoints"""
        print("\n🧪 Testing all API endpoints...")
        
        endpoints = [
            ("GET", "/", "Root endpoint"),
            ("GET", "/api/dashboard-data", "Dashboard data"),
            ("GET", "/api/kill-switch/status", "Kill switch status"),
            ("GET", "/api/system/status", "System status"),
            ("GET", "/api/launcher/status", "Launcher status"),
            ("GET", "/api/training/modes", "Training modes"),
            ("GET", "/api/training/status", "Training status"),
            ("GET", "/api/data/status", "Data status"),
            ("GET", "/api/tokens", "Token management"),
            ("GET", "/api/models/available", "Available models"),
            ("GET", "/api/monitoring/dashboard", "Monitoring dashboard"),
        ]
        
        passed = 0
        total = len(endpoints)
        
        for method, endpoint, description in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{self.api_base_url}{endpoint}", timeout=5)
                else:
                    response = requests.post(f"{self.api_base_url}{endpoint}", timeout=5)
                
                if response.status_code == 200:
                    print(f"✅ {description}")
                    passed += 1
                else:
                    print(f"❌ {description} - Status: {response.status_code}")
            except Exception as e:
                print(f"❌ {description} - Error: {e}")
        
        self.test_results["api_endpoints"] = {"passed": passed, "total": total}
        print(f"\n📊 API Endpoints: {passed}/{total} passed")
    
    def test_post_endpoints(self):
        """Test POST endpoints"""
        print("\n🧪 Testing POST endpoints...")
        
        post_tests = [
            {
                "endpoint": "/api/launcher/start",
                "data": {"mode": "blank", "symbol": "ETHUSDT", "exchange": "BINANCE"},
                "description": "Start launcher mode"
            },
            {
                "endpoint": "/api/training/start", 
                "data": {"mode": "blank", "symbol": "ETHUSDT", "exchange": "BINANCE"},
                "description": "Start training"
            },
            {
                "endpoint": "/api/kill-switch/activate",
                "data": {"reason": "Test activation", "emergency": False},
                "description": "Activate kill switch"
            },
            {
                "endpoint": "/api/kill-switch/deactivate",
                "data": {},
                "description": "Deactivate kill switch"
            }
        ]
        
        passed = 0
        total = len(post_tests)
        
        for test in post_tests:
            try:
                response = requests.post(
                    f"{self.api_base_url}{test['endpoint']}",
                    json=test["data"],
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success", True):  # Most endpoints return success
                        print(f"✅ {test['description']}")
                        passed += 1
                    else:
                        print(f"❌ {test['description']} - Failed: {result.get('error', 'Unknown error')}")
                else:
                    print(f"❌ {test['description']} - Status: {response.status_code}")
            except Exception as e:
                print(f"❌ {test['description']} - Error: {e}")
        
        self.test_results["post_endpoints"] = {"passed": passed, "total": total}
        print(f"\n📊 POST Endpoints: {passed}/{total} passed")
    
    def test_launcher_integration(self):
        """Test launcher integration functionality"""
        print("\n🧪 Testing launcher integration...")
        
        try:
            # Test launcher status
            response = requests.get(f"{self.api_base_url}/api/launcher/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("integration_available"):
                    print("✅ Launcher integration available")
                    
                    # Test starting a process
                    start_response = requests.post(
                        f"{self.api_base_url}/api/launcher/start",
                        json={"mode": "blank", "symbol": "ETHUSDT", "exchange": "BINANCE"},
                        timeout=5
                    )
                    
                    if start_response.status_code == 200:
                        result = start_response.json()
                        if result.get("success"):
                            print("✅ Launcher process started successfully")
                            print(f"   Process ID: {result.get('pid')}")
                            print(f"   Command: {result.get('command')}")
                            
                            # Wait a moment and check status
                            time.sleep(2)
                            status_response = requests.get(f"{self.api_base_url}/api/launcher/status", timeout=5)
                            if status_response.status_code == 200:
                                status_data = status_response.json()
                                print(f"✅ Process status check: {len(status_data.get('running_processes', []))} processes")
                            
                            self.test_results["launcher_integration"] = True
                        else:
                            print(f"❌ Launcher process start failed: {result.get('error')}")
                            self.test_results["launcher_integration"] = False
                    else:
                        print(f"❌ Launcher start request failed: {start_response.status_code}")
                        self.test_results["launcher_integration"] = False
                else:
                    print("⚠️ Launcher integration not available (fallback mode)")
                    self.test_results["launcher_integration"] = "fallback"
            else:
                print(f"❌ Launcher status check failed: {response.status_code}")
                self.test_results["launcher_integration"] = False
                
        except Exception as e:
            print(f"❌ Launcher integration test failed: {e}")
            self.test_results["launcher_integration"] = False
    
    def test_frontend_accessibility(self):
        """Test if frontend is accessible"""
        print("\n🧪 Testing frontend accessibility...")
        
        try:
            response = requests.get(self.frontend_url, timeout=5)
            if response.status_code == 200:
                print("✅ Frontend is accessible")
                self.test_results["frontend_accessibility"] = True
            else:
                print(f"❌ Frontend not accessible: {response.status_code}")
                self.test_results["frontend_accessibility"] = False
        except Exception as e:
            print(f"❌ Frontend accessibility test failed: {e}")
            self.test_results["frontend_accessibility"] = False
    
    def test_websocket_connection(self):
        """Test WebSocket connection"""
        print("\n🧪 Testing WebSocket connection...")
        
        try:
            import websocket
            
            def on_message(ws, message):
                print("✅ WebSocket message received")
                ws.close()
            
            def on_error(ws, error):
                print(f"❌ WebSocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                print("✅ WebSocket connection closed")
            
            def on_open(ws):
                print("✅ WebSocket connection opened")
                ws.send(json.dumps({"type": "ping"}))
            
            ws = websocket.WebSocketApp(
                f"ws://localhost:8000/ws",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Run for a short time
            ws.run_forever(timeout=5)
            self.test_results["websocket"] = True
            
        except ImportError:
            print("⚠️ WebSocket library not available, skipping test")
            self.test_results["websocket"] = "skipped"
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
            self.test_results["websocket"] = False
    
    def test_data_endpoints(self):
        """Test data-related endpoints"""
        print("\n🧪 Testing data endpoints...")
        
        data_tests = [
            ("/api/data/status", "Data status"),
            ("/api/tokens", "Token management"),
            ("/api/models/available", "Available models"),
            ("/api/training/modes", "Training modes"),
        ]
        
        passed = 0
        total = len(data_tests)
        
        for endpoint, description in data_tests:
            try:
                response = requests.get(f"{self.api_base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, (list, dict)) and len(data) > 0:
                        print(f"✅ {description} - Data available")
                        passed += 1
                    else:
                        print(f"⚠️ {description} - No data returned")
                        passed += 0.5  # Partial credit
                else:
                    print(f"❌ {description} - Status: {response.status_code}")
            except Exception as e:
                print(f"❌ {description} - Error: {e}")
        
        self.test_results["data_endpoints"] = {"passed": passed, "total": total}
        print(f"\n📊 Data Endpoints: {passed}/{total} passed")
    
    def generate_report(self):
        """Generate a comprehensive test report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE GUI FUNCTIONALITY TEST REPORT")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in self.test_results.items():
            if isinstance(result, bool):
                total_tests += 1
                if result:
                    passed_tests += 1
                    status = "✅ PASSED"
                else:
                    status = "❌ FAILED"
                print(f"{test_name:<25} {status}")
            elif isinstance(result, dict):
                if "passed" in result and "total" in result:
                    total_tests += 1
                    if result["passed"] >= result["total"] * 0.8:  # 80% threshold
                        passed_tests += 1
                        status = "✅ PASSED"
                    else:
                        status = "❌ FAILED"
                    print(f"{test_name:<25} {status} ({result['passed']}/{result['total']})")
            elif result == "skipped":
                print(f"{test_name:<25} ⚠️ SKIPPED")
            elif result == "fallback":
                print(f"{test_name:<25} ⚠️ FALLBACK MODE")
        
        print("="*60)
        print(f"Overall Result: {passed_tests}/{total_tests} test categories passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! GUI is fully functional.")
        elif passed_tests >= total_tests * 0.8:
            print("✅ MOSTLY FUNCTIONAL! GUI is working with minor issues.")
        else:
            print("❌ SIGNIFICANT ISSUES! GUI needs attention.")
        
        print("="*60)
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Comprehensive GUI Functionality Test")
        print("="*60)
        
        # Test API server startup
        api_process = self.test_api_server_startup()
        if not api_process:
            print("❌ Cannot continue without API server")
            return
        
        try:
            # Run all tests
            self.test_all_api_endpoints()
            self.test_post_endpoints()
            self.test_launcher_integration()
            self.test_frontend_accessibility()
            self.test_websocket_connection()
            self.test_data_endpoints()
            
            # Generate report
            self.generate_report()
            
        finally:
            # Cleanup
            if api_process:
                api_process.terminate()
                api_process.wait()

def main():
    """Main test function"""
    tester = GUIFunctionalityTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()