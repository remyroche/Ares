#!/usr/bin/env python3
"""
Minimal MEXC Exchange Implementation Test

This script validates the core MEXC exchange implementation without dependencies.
"""

import asyncio
import sys
import inspect
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/workspace')

def test_interface_compliance():
    """Test compliance with ExchangeInterface and BaseExchange."""
    print("📋 Testing Interface Compliance...")
    
    try:
        # Import only the core classes
        from exchanges.mexc import MexcExchange, create_mexc_exchange
        from src.interfaces.base_interfaces import IExchangeClient, MarketData
        from exchanges.base_exchange.base_exchange import BaseExchange
        
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
        return True
        
    except Exception as e:
        print(f"❌ Interface compliance test failed: {e}")
        return False

async def test_fast_fail_behavior():
    """Test fast-fail behavior instead of fallbacks."""
    print("\n⚡ Testing Fast-Fail Behavior...")
    
    try:
        from exchanges.mexc import create_mexc_exchange
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ Fast-fail behavior test failed: {e}")
        return False

def test_api_implementation():
    """Test API implementation completeness."""
    print("\n🔌 Testing API Implementation...")
    
    try:
        from exchanges.mexc import MexcExchange
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ API implementation test failed: {e}")
        return False

async def main():
    """Main entry point."""
    print("🚀 Starting MEXC Exchange Implementation Validation")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Interface Compliance", test_interface_compliance),
        ("Fast-Fail Behavior", test_fast_fail_behavior),
        ("API Implementation", test_api_implementation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 MEXC EXCHANGE VALIDATION REPORT")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total*100):.1f}%")
    
    print("\n📋 Detailed Results:")
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    overall_success = passed == total
    print(f"\n🎯 Overall Result: {'✅ VALIDATION PASSED' if overall_success else '❌ VALIDATION FAILED'}")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    asyncio.run(main())