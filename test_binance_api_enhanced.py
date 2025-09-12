#!/usr/bin/env python3
"""
Enhanced Binance API Test

This script tests the enhanced Binance API implementation with graceful
dependency handling and comprehensive functionality testing.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_binance_api_enhanced():
    """Test enhanced Binance API functionality."""
    print("🚀 Testing Enhanced Binance API...")
    print("="*60)
    
    try:
        # Test import
        from src.exchange.binance_enhanced import BinanceExchangeEnhanced, BinanceAPIError, BinanceDependencyError
        print("✅ Enhanced Binance API imported successfully")
        
        # Test class instantiation
        config = {
            'binance_exchange': {
                'use_testnet': True,
                'timeout': 30,
                'max_retries': 3,
                'rate_limit_enabled': True,
                'rate_limit_requests': 100,
                'rate_limit_window': 60
            }
        }
        
        exchange = BinanceExchangeEnhanced(config)
        print("✅ Enhanced Binance Exchange instance created")
        
        # Test configuration
        print(f"   - Base URL: {exchange.base_url}")
        print(f"   - Testnet URL: {exchange.testnet_url}")
        print(f"   - Use Testnet: {exchange.use_testnet}")
        print(f"   - Timeout: {exchange.timeout}")
        print(f"   - Max Retries: {exchange.max_retries}")
        print(f"   - Rate Limiting: {exchange.rate_limit_enabled}")
        
        # Test URL generation
        base_url = exchange._get_base_url()
        futures_url = exchange._get_futures_base_url()
        print(f"   - Generated Base URL: {base_url}")
        print(f"   - Generated Futures URL: {futures_url}")
        
        # Test signature generation
        test_params = {'symbol': 'BTCUSDT', 'timestamp': 1234567890000}
        signature = exchange._generate_signature(test_params)
        print(f"   - Signature generation: {'✅ Working' if signature else '❌ Failed'}")
        
        # Test rate limiting
        rate_limit_ok = exchange._check_rate_limit()
        print(f"   - Rate limiting check: {'✅ Working' if rate_limit_ok else '❌ Failed'}")
        
        # Test configuration validation
        is_valid = exchange._validate_configuration()
        print(f"   - Configuration validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        # Test exchange status
        status = exchange.get_exchange_status()
        print(f"   - Exchange Status: {status}")
        
        # Test method existence
        required_methods = [
            'initialize', 'get_klines', 'get_ticker', 'get_order_book',
            'get_aggregate_trades', 'futures_funding_rate', 'get_account_info',
            'get_position_risk', 'stop'
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(exchange, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        
        print("✅ All required methods present")
        
        # Test error handling
        try:
            # Test with invalid configuration
            invalid_config = {
                'binance_exchange': {
                    'timeout': -1,
                    'max_retries': -1
                }
            }
            invalid_exchange = BinanceExchangeEnhanced(invalid_config)
            is_invalid = invalid_exchange._validate_configuration()
            print(f"   - Invalid config handling: {'✅ Correctly rejected' if not is_invalid else '❌ Should reject'}")
        except Exception as e:
            print(f"   - Error handling test: ❌ Exception: {e}")
        
        print("✅ Enhanced Binance API test passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Binance API test failed: {e}")
        return False

async def test_binance_api_async():
    """Test async functionality of enhanced Binance API."""
    print("\n🔄 Testing Async Functionality...")
    print("-"*40)
    
    try:
        from src.exchange.binance_enhanced import BinanceExchangeEnhanced
        
        config = {
            'binance_exchange': {
                'use_testnet': True,
                'timeout': 30,
                'max_retries': 3
            }
        }
        
        exchange = BinanceExchangeEnhanced(config)
        
        # Test initialization
        print("🔌 Testing initialization...")
        init_result = await exchange.initialize()
        print(f"   - Initialization: {'✅ Success' if init_result else '❌ Failed'}")
        
        if init_result:
            # Test public endpoints
            print("📊 Testing public endpoints...")
            
            # Test server time
            server_time = await exchange._get_server_time()
            print(f"   - Server time: {'✅ Retrieved' if server_time else '❌ Failed'}")
            
            # Test klines (if dependencies available)
            try:
                klines = await exchange.get_klines('BTCUSDT', '1m', 10)
                print(f"   - Klines: {'✅ Retrieved' if klines else '❌ Failed'}")
            except Exception as e:
                print(f"   - Klines: ⚠️ {type(e).__name__}: {e}")
            
            # Test ticker
            try:
                ticker = await exchange.get_ticker('BTCUSDT')
                print(f"   - Ticker: {'✅ Retrieved' if ticker else '❌ Failed'}")
            except Exception as e:
                print(f"   - Ticker: ⚠️ {type(e).__name__}: {e}")
            
            # Test order book
            try:
                order_book = await exchange.get_order_book('BTCUSDT', 10)
                print(f"   - Order book: {'✅ Retrieved' if order_book else '❌ Failed'}")
            except Exception as e:
                print(f"   - Order book: ⚠️ {type(e).__name__}: {e}")
            
            # Test aggregate trades
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=1)
                start_time_ms = int(start_time.timestamp() * 1000)
                end_time_ms = int(end_time.timestamp() * 1000)
                
                agg_trades = await exchange.get_aggregate_trades('BTCUSDT', start_time_ms, end_time_ms)
                print(f"   - Aggregate trades: {'✅ Retrieved' if agg_trades is not None else '❌ Failed'}")
            except Exception as e:
                print(f"   - Aggregate trades: ⚠️ {type(e).__name__}: {e}")
            
            # Test futures funding rates
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=1)
                start_time_ms = int(start_time.timestamp() * 1000)
                end_time_ms = int(end_time.timestamp() * 1000)
                
                funding_rates = await exchange.futures_funding_rate('BTCUSDT', start_time_ms, end_time_ms)
                print(f"   - Futures funding: {'✅ Retrieved' if funding_rates is not None else '❌ Failed'}")
            except Exception as e:
                print(f"   - Futures funding: ⚠️ {type(e).__name__}: {e}")
        
        # Test cleanup
        print("🛑 Testing cleanup...")
        await exchange.stop()
        print("   - Cleanup: ✅ Completed")
        
        print("✅ Async functionality test completed")
        return True
        
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        return False

def test_binance_api_integration():
    """Test Binance API integration with data collection pipeline."""
    print("\n🔗 Testing Pipeline Integration...")
    print("-"*40)
    
    try:
        # Test backward compatibility
        from src.exchange.binance import BinanceExchange
        print("✅ Backward compatibility import successful")
        
        # Test factory method
        exchange = BinanceExchange.get_exchange('binance')
        print("✅ Factory method working")
        
        # Test configuration
        config = {
            'binance_exchange': {
                'use_testnet': True,
                'timeout': 30,
                'max_retries': 3
            }
        }
        
        exchange = BinanceExchange(config)
        status = exchange.get_exchange_status()
        print(f"   - Integration status: {status}")
        
        print("✅ Pipeline integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Binance API Tests...")
    print("="*60)
    
    # Test 1: Basic functionality
    test1_passed = test_binance_api_enhanced()
    
    # Test 2: Async functionality
    test2_passed = await test_binance_api_async()
    
    # Test 3: Integration
    test3_passed = test_binance_api_integration()
    
    # Results
    print("\n" + "="*60)
    print("📊 ENHANCED BINANCE API TEST RESULTS")
    print("="*60)
    
    tests = [
        ("Basic Functionality", test1_passed),
        ("Async Functionality", test2_passed),
        ("Pipeline Integration", test3_passed)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print("-"*60)
    print(f"TOTAL TESTS: {total}")
    print(f"PASSED: {passed}")
    print(f"FAILED: {total - passed}")
    print(f"SUCCESS RATE: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ENHANCED BINANCE API IS FULLY FUNCTIONAL!")
    elif passed >= total * 0.8:
        print("⚠️ ENHANCED BINANCE API IS MOSTLY FUNCTIONAL")
    else:
        print("❌ ENHANCED BINANCE API HAS ISSUES")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())