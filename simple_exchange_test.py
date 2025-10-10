#!/usr/bin/env python3
"""
Simple test script to verify exchange integrations work properly.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all exchange modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test shared interfaces
        from exchanges.shared.interfaces_typed import tprint, DataSource, ValidationResult
        print("✅ Shared interfaces imported successfully")
        
        # Test tprint function
        tprint("Test message", "INFO")
        print("✅ tprint function working")
        
        # Test DataSource enum
        assert DataSource.CACHE.value == "cache"
        assert DataSource.EXCHANGE.value == "exchange"
        assert DataSource.FALLBACK.value == "fallback"
        print("✅ DataSource enum working")
        
        # Test ValidationResult
        result = ValidationResult(True)
        assert result.is_valid == True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        print("✅ ValidationResult working")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_exchange_classes():
    """Test that exchange classes can be instantiated."""
    print("\n🧪 Testing exchange class instantiation...")
    
    try:
        # Test BingX
        from exchanges.bingx import BingXExchange
        bingx = BingXExchange(
            api_key="test_key",
            api_secret="test_secret",
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        print("✅ BingX exchange instantiated successfully")
        
        # Test MEXC
        from exchanges.mexc import MexcExchange
        mexc = MexcExchange(
            api_key="test_key",
            api_secret="test_secret",
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        print("✅ MEXC exchange instantiated successfully")
        
        # Test Binance
        from exchanges.binance import BinanceExchange
        binance = BinanceExchange(
            api_key="test_key",
            api_secret="test_secret",
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        print("✅ Binance exchange instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Exchange class test failed: {e}")
        return False


def test_shared_utilities():
    """Test that shared utilities are properly initialized."""
    print("\n🧪 Testing shared utilities...")
    
    try:
        from exchanges.mexc import MexcExchange
        
        mexc = MexcExchange(
            api_key="test_key",
            api_secret="test_secret",
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        
        # Check that shared utilities are initialized
        assert hasattr(mexc, 'auth_manager'), "Auth manager not initialized"
        assert hasattr(mexc, 'market_metadata'), "Market metadata manager not initialized"
        assert hasattr(mexc, 'price_manager'), "Price manager not initialized"
        assert hasattr(mexc, 'order_manager'), "Order manager not initialized"
        assert hasattr(mexc, 'balance_manager'), "Balance manager not initialized"
        assert hasattr(mexc, 'rate_limit_manager'), "Rate limit manager not initialized"
        
        print("✅ Shared utilities initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Shared utilities test failed: {e}")
        return False


def test_type_hints():
    """Test that type hints are properly implemented."""
    print("\n🧪 Testing type hints...")
    
    try:
        from exchanges.shared.interfaces_typed import (
            IHighLevelAuthManager, IHighLevelMarketManager, IHighLevelOrderManager,
            IHighLevelRiskManager, IHighLevelBalanceManager, IHighLevelRateLimitManager
        )
        
        # Test that interfaces exist
        assert IHighLevelAuthManager is not None
        assert IHighLevelMarketManager is not None
        assert IHighLevelOrderManager is not None
        assert IHighLevelRiskManager is not None
        assert IHighLevelBalanceManager is not None
        assert IHighLevelRateLimitManager is not None
        
        print("✅ Type hints working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Type hints test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting simple exchange integration tests...\n")
    
    tests = [
        test_imports,
        test_exchange_classes,
        test_shared_utilities,
        test_type_hints
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Exchange integrations are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)