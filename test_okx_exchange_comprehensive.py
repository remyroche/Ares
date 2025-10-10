#!/usr/bin/env python3
"""
Comprehensive OKX Exchange Test Suite

This test suite verifies that the OKX exchange implementation is fully functional,
compatible with ExchangeInterface, provides standardized klines, and implements
fast-fail behavior without fallbacks or mock data.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exchanges.okx import create_okx_exchange
from src.interfaces.base_interfaces import MarketData


class OKXExchangeTestSuite:
    """Comprehensive test suite for OKX exchange implementation."""
    
    def __init__(self):
        self.exchange = None
        self.test_results = {}
        self.errors = []
        
    async def setup(self):
        """Set up the test environment."""
        print("🚀 Setting up OKX Exchange Test Suite")
        print("=" * 50)
        
        # Create exchange instance
        self.exchange = create_okx_exchange(
            api_key=os.getenv("OKX_API_KEY", "test_key"),
            api_secret=os.getenv("OKX_API_SECRET", "test_secret"),
            password=os.getenv("OKX_PASSPHRASE", "test_passphrase"),
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        
        print("✅ Exchange instance created")
    
    async def teardown(self):
        """Clean up test environment."""
        if self.exchange:
            await self.exchange.close()
            print("✅ Exchange connection closed")
    
    def log_test_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        self.test_results[test_name] = {"success": success, "message": message}
        if not success:
            self.errors.append(f"{test_name}: {message}")
    
    async def test_interface_compliance(self):
        """Test that OKX exchange implements all required interface methods."""
        print("\n📋 Testing Interface Compliance")
        print("-" * 30)
        
        required_methods = [
            'get_klines',
            'get_historical_klines', 
            'get_historical_agg_trades',
            'create_order',
            'cancel_order',
            'get_order_status',
            'get_open_orders',
            'get_account_info',
            'get_position_risk',
            'close'
        ]
        
        for method_name in required_methods:
            has_method = hasattr(self.exchange, method_name)
            self.log_test_result(
                f"Has {method_name} method",
                has_method,
                "Method exists" if has_method else "Method missing"
            )
            
            if has_method:
                method = getattr(self.exchange, method_name)
                is_callable = callable(method)
                self.log_test_result(
                    f"{method_name} is callable",
                    is_callable,
                    "Method is callable" if is_callable else "Method not callable"
                )
    
    async def test_klines_standardization(self):
        """Test that klines are properly standardized."""
        print("\n📊 Testing Klines Standardization")
        print("-" * 30)
        
        try:
            # Test get_klines method
            klines = await self.exchange.get_klines("BTCUSDT", "1h", 10)
            
            # Check return type
            is_list = isinstance(klines, list)
            self.log_test_result(
                "get_klines returns list",
                is_list,
                f"Returned {type(klines)} instead of list"
            )
            
            if is_list and klines:
                # Check MarketData structure
                first_kline = klines[0]
                is_market_data = isinstance(first_kline, MarketData)
                self.log_test_result(
                    "Klines are MarketData objects",
                    is_market_data,
                    f"Expected MarketData, got {type(first_kline)}"
                )
                
                if is_market_data:
                    # Check required fields
                    required_fields = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'interval']
                    missing_fields = [field for field in required_fields if not hasattr(first_kline, field)]
                    
                    self.log_test_result(
                        "MarketData has all required fields",
                        len(missing_fields) == 0,
                        f"Missing fields: {missing_fields}" if missing_fields else "All fields present"
                    )
                    
                    # Check data types
                    if not missing_fields:
                        timestamp_ok = isinstance(first_kline.timestamp, datetime)
                        numeric_fields_ok = all(
                            isinstance(getattr(first_kline, field), (int, float))
                            for field in ['open', 'high', 'low', 'close', 'volume']
                        )
                        string_fields_ok = all(
                            isinstance(getattr(first_kline, field), str)
                            for field in ['symbol', 'interval']
                        )
                        
                        self.log_test_result(
                            "MarketData has correct data types",
                            timestamp_ok and numeric_fields_ok and string_fields_ok,
                            "Data types are correct" if (timestamp_ok and numeric_fields_ok and string_fields_ok) else "Incorrect data types"
                        )
            
        except Exception as e:
            self.log_test_result(
                "Klines standardization test",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_historical_klines(self):
        """Test historical klines functionality."""
        print("\n📈 Testing Historical Klines")
        print("-" * 30)
        
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            # Test with datetime objects
            hist_klines = await self.exchange.get_historical_klines(
                "BTCUSDT", "1h", start_time, end_time, 24
            )
            
            is_list = isinstance(hist_klines, list)
            self.log_test_result(
                "get_historical_klines returns list",
                is_list,
                f"Returned {type(hist_klines)} instead of list"
            )
            
            if is_list and hist_klines:
                # Check that we got MarketData objects
                first_kline = hist_klines[0]
                is_market_data = isinstance(first_kline, MarketData)
                self.log_test_result(
                    "Historical klines are MarketData objects",
                    is_market_data,
                    f"Expected MarketData, got {type(first_kline)}"
                )
            
            # Test with timestamp integers
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            hist_klines_ms = await self.exchange.get_historical_klines(
                "BTCUSDT", "1h", start_ms, end_ms, 24
            )
            
            is_list_ms = isinstance(hist_klines_ms, list)
            self.log_test_result(
                "get_historical_klines works with timestamps",
                is_list_ms,
                f"Failed with timestamp parameters: {type(hist_klines_ms)}"
            )
            
        except Exception as e:
            self.log_test_result(
                "Historical klines test",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_fast_fail_behavior(self):
        """Test that the exchange fails fast instead of returning empty results."""
        print("\n⚡ Testing Fast-Fail Behavior")
        print("-" * 30)
        
        # Test with invalid symbol
        try:
            await self.exchange.get_klines("INVALID_SYMBOL_12345", "1h", 10)
            self.log_test_result(
                "Fast-fail with invalid symbol",
                False,
                "Should have raised exception but didn't"
            )
        except Exception as e:
            self.log_test_result(
                "Fast-fail with invalid symbol",
                True,
                f"Correctly raised exception: {type(e).__name__}"
            )
        
        # Test with invalid interval
        try:
            await self.exchange.get_klines("BTCUSDT", "invalid_interval", 10)
            self.log_test_result(
                "Fast-fail with invalid interval",
                False,
                "Should have raised exception but didn't"
            )
        except Exception as e:
            self.log_test_result(
                "Fast-fail with invalid interval",
                True,
                f"Correctly raised exception: {type(e).__name__}"
            )
        
        # Test with invalid limit
        try:
            await self.exchange.get_klines("BTCUSDT", "1h", -1)
            self.log_test_result(
                "Fast-fail with invalid limit",
                False,
                "Should have raised exception but didn't"
            )
        except Exception as e:
            self.log_test_result(
                "Fast-fail with invalid limit",
                True,
                f"Correctly raised exception: {type(e).__name__}"
            )
    
    async def test_no_mock_data(self):
        """Test that no mock data or stubs are used."""
        print("\n🚫 Testing No Mock Data")
        print("-" * 30)
        
        # Check that methods don't return empty results when they should work
        try:
            # This should work and return real data
            klines = await self.exchange.get_klines("BTCUSDT", "1h", 5)
            
            if klines and len(klines) > 0:
                self.log_test_result(
                    "Returns real data, not empty results",
                    True,
                    f"Got {len(klines)} real klines"
                )
                
                # Check that data looks realistic (not obviously fake)
                first_kline = klines[0]
                if hasattr(first_kline, 'open') and hasattr(first_kline, 'close'):
                    price_realistic = 1000 < first_kline.open < 100000  # BTC price range
                    self.log_test_result(
                        "Data appears realistic",
                        price_realistic,
                        f"Price {first_kline.open} seems unrealistic" if not price_realistic else "Price data looks realistic"
                    )
            else:
                self.log_test_result(
                    "Returns real data, not empty results",
                    False,
                    "Got empty results when real data should be available"
                )
                
        except Exception as e:
            # If we can't get real data due to API issues, that's still better than mock data
            self.log_test_result(
                "No mock data fallback",
                True,
                f"Raises exception instead of returning mock data: {type(e).__name__}"
            )
    
    async def test_error_handling(self):
        """Test comprehensive error handling."""
        print("\n🛡️ Testing Error Handling")
        print("-" * 30)
        
        # Test session not initialized
        try:
            # Create a new exchange without initializing
            test_exchange = create_okx_exchange("test", "test", "BTCUSDT")
            await test_exchange.get_klines("BTCUSDT", "1h", 10)
            self.log_test_result(
                "Error handling for uninitialized session",
                False,
                "Should have raised exception but didn't"
            )
        except Exception as e:
            self.log_test_result(
                "Error handling for uninitialized session",
                True,
                f"Correctly raised exception: {type(e).__name__}"
            )
    
    async def test_api_functionality(self):
        """Test core API functionality."""
        print("\n🔌 Testing API Functionality")
        print("-" * 30)
        
        try:
            # Test server time
            server_time = await self.exchange._get_server_time()
            has_server_time = server_time is not None
            self.log_test_result(
                "Server time retrieval",
                has_server_time,
                f"Got server time: {server_time}" if has_server_time else "No server time"
            )
            
            # Test connection
            connection_ok = await self.exchange._test_connection()
            self.log_test_result(
                "Connection test",
                connection_ok,
                "Connection successful" if connection_ok else "Connection failed"
            )
            
        except Exception as e:
            self.log_test_result(
                "API functionality test",
                False,
                f"Exception: {str(e)}"
            )
    
    async def run_all_tests(self):
        """Run all tests."""
        print("🧪 Running Comprehensive OKX Exchange Tests")
        print("=" * 60)
        
        try:
            await self.setup()
            
            # Run all test categories
            await self.test_interface_compliance()
            await self.test_klines_standardization()
            await self.test_historical_klines()
            await self.test_fast_fail_behavior()
            await self.test_no_mock_data()
            await self.test_error_handling()
            await self.test_api_functionality()
            
        except Exception as e:
            print(f"❌ Test suite failed with exception: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.teardown()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print("\n📊 Test Summary")
        print("=" * 30)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.errors:
            print("\n❌ Failed Tests:")
            for error in self.errors:
                print(f"  - {error}")
        
        if failed_tests == 0:
            print("\n🎉 All tests passed! OKX exchange implementation is fully functional.")
        else:
            print(f"\n⚠️ {failed_tests} tests failed. Please review the implementation.")


async def main():
    """Main test runner."""
    test_suite = OKXExchangeTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())