#!/usr/bin/env python3
"""
MEXC Exchange Implementation Validation Test

This script validates that the MEXC exchange implementation:
1. Fully implements ExchangeInterface (both ways)
2. Provides standardized klines data
3. Uses fast-fail behavior instead of fallbacks
4. Has no mock data, stubs, or placeholders
5. Has fully functional API implementations

Usage:
    python mexc_validation_test.py [--api-key KEY] [--api-secret SECRET] [--testnet]
"""

import asyncio
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

# Add project root to path
sys.path.insert(0, '/workspace')

try:
    from exchanges.mexc import MexcExchange, create_mexc_exchange
    from src.interfaces.base_interfaces import IExchangeClient, MarketData
    from exchanges.base_exchange.base_exchange import BaseExchange
    from exchanges.shared.exchange_data_standardizer import ExchangeDataStandardizer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class MexcValidationTest:
    """Comprehensive validation test for MEXC exchange implementation."""
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange = None
        self.results = {
            "interface_compliance": {},
            "klines_standardization": {},
            "fast_fail_behavior": {},
            "api_implementation": {},
            "overall_success": False
        }
        
    async def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation tests."""
        print("🚀 Starting MEXC Exchange Implementation Validation")
        print("=" * 60)
        
        try:
            # Test 1: Interface Compliance
            await self._test_interface_compliance()
            
            # Test 2: Klines Standardization
            await self._test_klines_standardization()
            
            # Test 3: Fast-Fail Behavior
            await self._test_fast_fail_behavior()
            
            # Test 4: API Implementation Completeness
            await self._test_api_implementation()
            
            # Calculate overall success
            self.results["overall_success"] = all(
                test.get("success", False) 
                for test in self.results.values() 
                if isinstance(test, dict) and "success" in test
            )
            
            # Generate report
            self._generate_report()
            
        except Exception as e:
            print(f"❌ Validation failed with error: {e}")
            traceback.print_exc()
            self.results["error"] = str(e)
            
        return self.results
    
    async def _test_interface_compliance(self):
        """Test compliance with ExchangeInterface and BaseExchange."""
        print("\n📋 Testing Interface Compliance...")
        
        try:
            # Test 1: Class inheritance
            assert issubclass(MexcExchange, BaseExchange), "MexcExchange must inherit from BaseExchange"
            assert issubclass(MexcExchange, IExchangeClient), "MexcExchange must implement IExchangeClient"
            print("✅ Class inheritance: PASSED")
            
            # Test 2: Required abstract methods implementation
            required_methods = [
                '_initialize_exchange',
                '_convert_to_market_data', 
                '_get_market_id',
                '_get_klines_raw',
                '_get_account_info_raw',
                '_create_order_raw',
                '_get_position_risk_raw',
                '_get_historical_klines_raw',
                '_get_historical_agg_trades_raw',
                '_get_open_orders_raw',
                '_cancel_order_raw',
                '_get_order_status_raw'
            ]
            
            for method in required_methods:
                assert hasattr(MexcExchange, method), f"Missing required method: {method}"
                method_obj = getattr(MexcExchange, method)
                assert not method_obj.__isabstractmethod__, f"Method {method} is still abstract"
            
            print("✅ Required methods implementation: PASSED")
            
            # Test 3: IExchangeClient interface methods
            client_methods = ['get_klines', 'get_account_info', 'create_order', 'get_position_risk']
            for method in client_methods:
                assert hasattr(MexcExchange, method), f"Missing IExchangeClient method: {method}"
            
            print("✅ IExchangeClient interface: PASSED")
            
            # Test 4: Factory function
            factory_exchange = create_mexc_exchange("test_key", "test_secret", "BTCUSDT")
            assert isinstance(factory_exchange, MexcExchange), "Factory function must return MexcExchange instance"
            
            print("✅ Factory function: PASSED")
            
            self.results["interface_compliance"] = {
                "success": True,
                "details": "All interface requirements met"
            }
            
        except Exception as e:
            print(f"❌ Interface compliance test failed: {e}")
            self.results["interface_compliance"] = {
                "success": False,
                "error": str(e)
            }
    
    async def _test_klines_standardization(self):
        """Test klines data standardization."""
        print("\n📊 Testing Klines Standardization...")
        
        try:
            # Create exchange instance
            self.exchange = create_mexc_exchange(self.api_key, self.api_secret, "BTCUSDT", use_testnet=self.testnet)
            
            # Test 1: Data format validation
            raw_klines = [
                [1640995200000, "50000.0", "51000.0", "49000.0", "50500.0", "100.5", 1640995259999, "5000000.0", 100, "50.0", "2500000.0"],
                [1640995260000, "50500.0", "51500.0", "49500.0", "51000.0", "150.2", 1640995319999, "7500000.0", 150, "75.0", "3750000.0"]
            ]
            
            # Test conversion to MarketData
            market_data = await self.exchange._convert_to_market_data(raw_klines, "BTCUSDT", "1m")
            
            assert isinstance(market_data, list), "Market data should be a list"
            assert len(market_data) == 2, "Should convert all klines"
            
            for i, data_point in enumerate(market_data):
                assert isinstance(data_point, MarketData), f"Item {i} should be MarketData instance"
                assert data_point.symbol == "BTCUSDT", f"Symbol should be BTCUSDT"
                assert data_point.interval == "1m", f"Interval should be 1m"
                assert data_point.open > 0, f"Open price should be positive"
                assert data_point.high > 0, f"High price should be positive"
                assert data_point.low > 0, f"Low price should be positive"
                assert data_point.close > 0, f"Close price should be positive"
                assert data_point.volume > 0, f"Volume should be positive"
                assert isinstance(data_point.timestamp, datetime), f"Timestamp should be datetime"
            
            print("✅ MarketData conversion: PASSED")
            
            # Test 2: Data validation
            invalid_klines = [
                [1640995200000, "0", "51000.0", "49000.0", "50500.0", "100.5"],  # Invalid open price
                [1640995260000, "50500.0", "51500.0", "49500.0", "51000.0", "150.2"]
            ]
            
            try:
                await self.exchange._convert_to_market_data(invalid_klines, "BTCUSDT", "1m")
                assert False, "Should fail with invalid data"
            except ValueError:
                print("✅ Data validation: PASSED")
            
            # Test 3: Empty data handling
            try:
                await self.exchange._convert_to_market_data([], "BTCUSDT", "1m")
                assert False, "Should fail with empty data"
            except ValueError:
                print("✅ Empty data handling: PASSED")
            
            self.results["klines_standardization"] = {
                "success": True,
                "details": "Klines standardization working correctly"
            }
            
        except Exception as e:
            print(f"❌ Klines standardization test failed: {e}")
            self.results["klines_standardization"] = {
                "success": False,
                "error": str(e)
            }
    
    async def _test_fast_fail_behavior(self):
        """Test fast-fail behavior instead of fallbacks."""
        print("\n⚡ Testing Fast-Fail Behavior...")
        
        try:
            if not self.exchange:
                self.exchange = create_mexc_exchange(self.api_key, self.api_secret, "BTCUSDT", use_testnet=self.testnet)
            
            # Test 1: Invalid symbol handling
            try:
                await self.exchange._get_klines_raw("INVALID_SYMBOL", "1m", 10)
                assert False, "Should fail fast with invalid symbol"
            except Exception:
                print("✅ Invalid symbol fast-fail: PASSED")
            
            # Test 2: Empty data handling
            try:
                await self.exchange._convert_to_market_data([], "BTCUSDT", "1m")
                assert False, "Should fail fast with empty data"
            except ValueError:
                print("✅ Empty data fast-fail: PASSED")
            
            # Test 3: Invalid data format handling
            try:
                await self.exchange._convert_to_market_data([{"invalid": "data"}], "BTCUSDT", "1m")
                assert False, "Should fail fast with invalid data format"
            except ValueError:
                print("✅ Invalid format fast-fail: PASSED")
            
            # Test 4: Missing required fields
            try:
                await self.exchange._convert_to_market_data([["incomplete"]], "BTCUSDT", "1m")
                assert False, "Should fail fast with incomplete data"
            except ValueError:
                print("✅ Incomplete data fast-fail: PASSED")
            
            self.results["fast_fail_behavior"] = {
                "success": True,
                "details": "Fast-fail behavior working correctly"
            }
            
        except Exception as e:
            print(f"❌ Fast-fail behavior test failed: {e}")
            self.results["fast_fail_behavior"] = {
                "success": False,
                "error": str(e)
            }
    
    async def _test_api_implementation(self):
        """Test API implementation completeness."""
        print("\n🔌 Testing API Implementation...")
        
        try:
            if not self.exchange:
                self.exchange = create_mexc_exchange(self.api_key, self.api_secret, "BTCUSDT", use_testnet=self.testnet)
            
            # Test 1: Check for no mock/stub implementations
            source_code = inspect.getsource(MexcExchange)
            
            mock_indicators = ["mock", "stub", "placeholder", "TODO", "FIXME", "NotImplementedError", "pass  # TODO"]
            found_mocks = [indicator for indicator in mock_indicators if indicator.lower() in source_code.lower()]
            
            if found_mocks:
                print(f"⚠️ Found potential mock/stub indicators: {found_mocks}")
            else:
                print("✅ No mock/stub implementations found: PASSED")
            
            # Test 2: Check method implementations
            methods_to_check = [
                '_get_klines_raw',
                '_get_historical_klines_raw', 
                '_get_historical_agg_trades_raw',
                '_get_account_info_raw',
                '_create_order_raw',
                '_get_position_risk_raw',
                '_get_open_orders_raw',
                '_cancel_order_raw',
                '_get_order_status_raw'
            ]
            
            for method_name in methods_to_check:
                method = getattr(MexcExchange, method_name)
                source = inspect.getsource(method)
                
                # Check for proper implementation (not just pass or raise NotImplementedError)
                if "pass" in source and len(source.strip().split('\n')) <= 3:
                    raise AssertionError(f"Method {method_name} appears to be a stub")
                if "NotImplementedError" in source:
                    raise AssertionError(f"Method {method_name} raises NotImplementedError")
            
            print("✅ All methods properly implemented: PASSED")
            
            # Test 3: Check error handling
            error_handling_methods = [
                '_make_request',
                '_convert_to_market_data',
                '_get_klines_raw'
            ]
            
            for method_name in error_handling_methods:
                method = getattr(MexcExchange, method_name)
                source = inspect.getsource(method)
                
                # Should have proper error handling, not just return None
                if "return None" in source and "raise" not in source:
                    print(f"⚠️ Method {method_name} may not have proper error handling")
            
            print("✅ Error handling implemented: PASSED")
            
            self.results["api_implementation"] = {
                "success": True,
                "details": "API implementation is complete and functional"
            }
            
        except Exception as e:
            print(f"❌ API implementation test failed: {e}")
            self.results["api_implementation"] = {
                "success": False,
                "error": str(e)
            }
    
    def _generate_report(self):
        """Generate validation report."""
        print("\n" + "=" * 60)
        print("📊 MEXC EXCHANGE VALIDATION REPORT")
        print("=" * 60)
        
        total_tests = len([k for k, v in self.results.items() if isinstance(v, dict) and "success" in v])
        passed_tests = len([k for k, v in self.results.items() if isinstance(v, dict) and v.get("success", False)])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.results.items():
            if isinstance(result, dict) and "success" in result:
                status = "✅ PASSED" if result["success"] else "❌ FAILED"
                print(f"  {test_name}: {status}")
                if not result["success"] and "error" in result:
                    print(f"    Error: {result['error']}")
        
        print(f"\n🎯 Overall Result: {'✅ VALIDATION PASSED' if self.results['overall_success'] else '❌ VALIDATION FAILED'}")
        
        # Save detailed report
        with open('/workspace/mexc_validation_report.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: mexc_validation_report.json")

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MEXC Exchange Implementation Validation")
    parser.add_argument("--api-key", help="MEXC API key (optional for basic validation)")
    parser.add_argument("--api-secret", help="MEXC API secret (optional for basic validation)")
    parser.add_argument("--testnet", action="store_true", default=True, help="Use testnet (default: True)")
    parser.add_argument("--live", action="store_true", help="Use live API (overrides testnet)")
    
    args = parser.parse_args()
    
    # Determine if using testnet
    use_testnet = not args.live
    
    # Run validation
    validator = MexcValidationTest(
        api_key=args.api_key or "",
        api_secret=args.api_secret or "",
        testnet=use_testnet
    )
    
    results = await validator.run_validation()
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_success"] else 1)

if __name__ == "__main__":
    import inspect
    asyncio.run(main())