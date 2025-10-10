#!/usr/bin/env python3
"""
Test script to verify exchange integrations work properly.

This script tests the integration of shared interfaces across all exchange implementations.
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exchanges.bingx import BingXExchange, create_bingx_exchange
from exchanges.mexc import MexcExchange, create_mexc_exchange
from exchanges.binance import BinanceExchange, create_binance_exchange
from exchanges.okx import OkxExchange, create_okx_exchange
from src.trading.execution.exchange_interface import ExchangeInterface


async def test_exchange_creation():
    """Test that all exchanges can be created without errors."""
    print("🧪 Testing exchange creation...")
    
    try:
        # Test BingX
        bingx = create_bingx_exchange(
            api_key="test_key",
            api_secret="test_secret",
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        print("✅ BingX exchange created successfully")
        
        # Test MEXC
        mexc = create_mexc_exchange(
            api_key="test_key",
            api_secret="test_secret",
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        print("✅ MEXC exchange created successfully")
        
        # Test Binance
        binance = create_binance_exchange(
            api_key="test_key",
            api_secret="test_secret",
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        print("✅ Binance exchange created successfully")
        
        # Test OKX
        okx = create_okx_exchange(
            api_key="test_key",
            api_secret="test_secret",
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        print("✅ OKX exchange created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Exchange creation failed: {e}")
        return False


async def test_shared_utilities():
    """Test that shared utilities are properly initialized."""
    print("\n🧪 Testing shared utilities initialization...")
    
    try:
        # Test with MEXC exchange
        mexc = create_mexc_exchange(
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


async def test_trading_interface():
    """Test the trading exchange interface."""
    print("\n🧪 Testing trading exchange interface...")
    
    try:
        # Test simulated exchange
        config = {
            'exchange_type': 'simulated',
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'testnet': True
        }
        
        interface = ExchangeInterface(config)
        
        # Test connection
        connected = await interface.connect()
        assert connected, "Failed to connect to simulated exchange"
        
        # Test ticker data
        ticker = await interface.get_ticker("BTCUSDT")
        assert ticker is not None, "Failed to get ticker data"
        
        # Test order book
        order_book = await interface.get_order_book("BTCUSDT")
        assert order_book is not None, "Failed to get order book"
        
        # Test klines
        klines = await interface.get_klines("BTCUSDT", "1m", limit=10)
        assert len(klines) > 0, "Failed to get kline data"
        
        # Test account balance
        balance = await interface.get_account_balance()
        assert len(balance) > 0, "Failed to get account balance"
        
        # Test order creation
        order = await interface.create_order(
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=0.001
        )
        assert order is not None, "Failed to create order"
        
        # Test disconnect
        await interface.disconnect()
        
        print("✅ Trading exchange interface working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Trading interface test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling with tprint."""
    print("\n🧪 Testing error handling...")
    
    try:
        # Test with invalid exchange type
        config = {
            'exchange_type': 'invalid_exchange',
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'testnet': True
        }
        
        interface = ExchangeInterface(config)
        
        # This should handle the error gracefully
        connected = await interface.connect()
        assert not connected, "Should fail to connect to invalid exchange"
        
        print("✅ Error handling working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def test_type_hints():
    """Test that type hints are properly implemented."""
    print("\n🧪 Testing type hints...")
    
    try:
        # Test that methods have proper type hints
        from exchanges.shared.interfaces_typed import tprint, DataSource, ValidationResult
        
        # Test tprint function
        tprint("Test message", "INFO")
        
        # Test DataSource enum
        assert DataSource.CACHE.value == "cache"
        assert DataSource.EXCHANGE.value == "exchange"
        assert DataSource.FALLBACK.value == "fallback"
        
        # Test ValidationResult
        result = ValidationResult(True)
        assert result.is_valid == True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        
        print("✅ Type hints working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Type hints test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting exchange integration tests...\n")
    
    tests = [
        test_exchange_creation,
        test_shared_utilities,
        test_trading_interface,
        test_error_handling,
        test_type_hints
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = await test()
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
    success = asyncio.run(main())
    sys.exit(0 if success else 1)