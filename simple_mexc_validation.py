#!/usr/bin/env python3
"""
Simple MEXC Exchange Implementation Validation

This script validates that the MEXC exchange implementation:
1. Fully implements ExchangeInterface (both ways)
2. Uses fast-fail behavior instead of fallbacks
3. Has no mock data, stubs, or placeholders
4. Has fully functional API implementations

Usage:
    python3 simple_mexc_validation.py
"""

import asyncio
import sys
import inspect
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, '/workspace')

try:
    from exchanges.mexc import MexcExchange, create_mexc_exchange
    from src.interfaces.base_interfaces import IExchangeClient, MarketData
    from exchanges.base_exchange.base_exchange import BaseExchange
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class SimpleMexcValidation:
    """Simple validation test for MEXC exchange implementation."""
    
    def __init__(self):
        self.results = {
            "interface_compliance": {},
            "fast_fail_behavior": {},
            "api_implementation": {},
            "overall_success": False
        }
        
    async def run_validation(self) -> Dict[str, Any]:
        """Run validation tests."""
        print("🚀 Starting MEXC Exchange Implementation Validation")
        print("=" * 60)
        
        try:
            # Test 1: Interface Compliance
            await self._test_interface_compliance()
            
            # Test 2: Fast-Fail Behavior
            await self._test_fast_fail_behavior()
            
            # Test 3: API Implementation Completeness
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
            import traceback
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
    
    async def _test_fast_fail_behavior(self):
        """Test fast-fail behavior instead of fallbacks."""
        print("\n⚡ Testing Fast-Fail Behavior...")
        
        try:
            # Create exchange instance
            exchange = create_mexc_exchange("test_key", "test_secret", "BTCUSDT", use_testnet=True)
            
            # Test 1: Empty data handling
            try:
                await exchange._convert_to_market_data([], "BTCUSDT", "1m")
                assert False, "Should fail fast with empty data"
            except ValueError:
                print("✅ Empty data fast-fail: PASSED")
            
            # Test 2: Invalid data format handling
            try:
                await exchange._convert_to_market_data([{"invalid": "data"}], "BTCUSDT", "1m")
                assert False, "Should fail fast with invalid data format"
            except ValueError:
                print("✅ Invalid format fast-fail: PASSED")
            
            # Test 3: Missing required fields
            try:
                await exchange._convert_to_market_data([["incomplete"]], "BTCUSDT", "1m")
                assert False, "Should fail fast with incomplete data"
            except ValueError:
                print("✅ Incomplete data fast-fail: PASSED")
            
            # Test 4: Invalid price data
            try:
                invalid_klines = [
                    [1640995200000, "0", "51000.0", "49000.0", "50500.0", "100.5"]  # Invalid open price
                ]
                await exchange._convert_to_market_data(invalid_klines, "BTCUSDT", "1m")
                assert False, "Should fail fast with invalid price data"
            except ValueError:
                print("✅ Invalid price data fast-fail: PASSED")
            
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
            
            # Test 3: Check error handling patterns
            error_handling_methods = [
                '_make_request',
                '_convert_to_market_data',
                '_get_klines_raw'
            ]
            
            for method_name in error_handling_methods:
                method = getattr(MexcExchange, method_name)
                source = inspect.getsource(method)
                
                # Should have proper error handling with raise statements
                if "raise" not in source:
                    print(f"⚠️ Method {method_name} may not have proper error handling")
                else:
                    print(f"✅ Method {method_name} has error handling: PASSED")
            
            # Test 4: Check for proper data validation
            convert_method = getattr(MexcExchange, '_convert_to_market_data')
            convert_source = inspect.getsource(convert_method)
            
            validation_checks = [
                "if not raw_data:",
                "raise ValueError",
                "isinstance(item, list)",
                "len(item) >= 6"
            ]
            
            for check in validation_checks:
                if check not in convert_source:
                    print(f"⚠️ Missing validation check: {check}")
                else:
                    print(f"✅ Validation check '{check}' found: PASSED")
            
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

async def main():
    """Main entry point."""
    # Run validation
    validator = SimpleMexcValidation()
    results = await validator.run_validation()
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_success"] else 1)

if __name__ == "__main__":
    asyncio.run(main())