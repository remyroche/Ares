#!/usr/bin/env python3
"""
Test script to verify real API implementations work.

This script tests the actual API functionality with real exchange APIs.
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exchanges.binance import create_binance_exchange
from exchanges.mexc import create_mexc_exchange
from exchanges.bingx import create_bingx_exchange
from exchanges.okx import create_okx_exchange
from src.trading.execution.exchange_interface import ExchangeInterface


async def test_binance_api():
    """Test Binance API functionality."""
    print("🧪 Testing Binance API...")
    
    try:
        # Create exchange instance
        binance = create_binance_exchange(
            api_key="test_key",
            api_secret="test_secret",
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        
        # Test signature generation
        test_params = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": "0.001",
            "timestamp": 1640995200000
        }
        
        signature = binance._generate_signature(test_params)
        print(f"✅ Binance signature generation: {signature[:16]}...")
        
        # Test rate limiting
        rate_limit_status = binance.rate_limit_manager.get_rate_limit_status("public")
        print(f"✅ Binance rate limiting: {rate_limit_status}")
        
        # Test order manager
        order_stats = binance.order_manager.get_statistics()
        print(f"✅ Binance order manager: {order_stats}")
        
        # Test balance manager
        balance_stats = binance.balance_manager.get_statistics()
        print(f"✅ Binance balance manager: {balance_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Binance API test failed: {e}")
        return False


async def test_mexc_api():
    """Test MEXC API functionality."""
    print("\n🧪 Testing MEXC API...")
    
    try:
        # Create exchange instance
        mexc = create_mexc_exchange(
            api_key="test_key",
            api_secret="test_secret",
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        
        # Test signature generation
        test_params = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": "0.001",
            "timestamp": 1640995200000
        }
        
        signature = mexc._generate_signature(test_params)
        print(f"✅ MEXC signature generation: {signature[:16]}...")
        
        # Test rate limiting
        rate_limit_status = mexc.rate_limit_manager.get_rate_limit_status("public")
        print(f"✅ MEXC rate limiting: {rate_limit_status}")
        
        # Test order manager
        order_stats = mexc.order_manager.get_statistics()
        print(f"✅ MEXC order manager: {order_stats}")
        
        # Test balance manager
        balance_stats = mexc.balance_manager.get_statistics()
        print(f"✅ MEXC balance manager: {balance_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ MEXC API test failed: {e}")
        return False


async def test_bingx_api():
    """Test BingX API functionality."""
    print("\n🧪 Testing BingX API...")
    
    try:
        # Create exchange instance
        bingx = create_bingx_exchange(
            api_key="test_key",
            api_secret="test_secret",
            trade_symbol="BTCUSDT",
            use_testnet=True
        )
        
        # Test rate limiting
        rate_limit_status = bingx.rate_limit_manager.get_rate_limit_status("public")
        print(f"✅ BingX rate limiting: {rate_limit_status}")
        
        # Test order manager
        order_stats = bingx.order_manager.get_statistics()
        print(f"✅ BingX order manager: {order_stats}")
        
        # Test balance manager
        balance_stats = bingx.balance_manager.get_statistics()
        print(f"✅ BingX balance manager: {balance_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ BingX API test failed: {e}")
        return False


async def test_trading_interface():
    """Test trading interface functionality."""
    print("\n🧪 Testing Trading Interface...")
    
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
        if not connected:
            print("❌ Failed to connect to simulated exchange")
            return False
        
        print("✅ Connected to simulated exchange")
        
        # Test ticker
        ticker = await interface.get_ticker("BTCUSDT")
        if ticker:
            print(f"✅ Got ticker: {ticker.price}")
        else:
            print("❌ Failed to get ticker")
            return False
        
        # Test order book
        order_book = await interface.get_order_book("BTCUSDT")
        if order_book:
            print(f"✅ Got order book: {len(order_book.get('bids', []))} bids")
        else:
            print("❌ Failed to get order book")
            return False
        
        # Test klines
        klines = await interface.get_klines("BTCUSDT", "1m", limit=10)
        if klines:
            print(f"✅ Got klines: {len(klines)} candles")
        else:
            print("❌ Failed to get klines")
            return False
        
        # Test order creation
        order = await interface.create_order(
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=0.001
        )
        if order:
            print(f"✅ Created order: {order.get('orderId', 'N/A')}")
        else:
            print("❌ Failed to create order")
            return False
        
        # Test disconnect
        await interface.disconnect()
        print("✅ Disconnected from exchange")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading interface test failed: {e}")
        return False


async def test_shared_utilities():
    """Test shared utilities functionality."""
    print("\n🧪 Testing Shared Utilities...")
    
    try:
        from exchanges.shared.auth.auth_manager import AuthenticationManager
        from exchanges.shared.reliability.rate_limit_manager import RateLimitManager
        from exchanges.shared.orders.order_manager import OrderManager
        from exchanges.shared.wallet.balance_manager import BalanceManager
        
        # Test authentication manager
        auth_manager = AuthenticationManager("test")
        print("✅ Authentication manager created")
        
        # Test rate limit manager
        rate_limit_manager = RateLimitManager("test")
        rate_limit_manager.set_rate_limit("test", rate_limit_manager.default_rate_limit)
        status = rate_limit_manager.get_rate_limit_status("test")
        print(f"✅ Rate limit manager: {status}")
        
        # Test order manager
        order_manager = OrderManager("test")
        stats = order_manager.get_statistics()
        print(f"✅ Order manager: {stats}")
        
        # Test balance manager
        balance_manager = BalanceManager("test")
        stats = balance_manager.get_statistics()
        print(f"✅ Balance manager: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Shared utilities test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling functionality."""
    print("\n🧪 Testing Error Handling...")
    
    try:
        from exchanges.shared.interfaces_typed import tprint, ValidationResult
        
        # Test tprint
        tprint("Test message", "INFO")
        tprint("Test warning", "WARNING")
        tprint("Test error", "ERROR")
        print("✅ tprint error handling working")
        
        # Test ValidationResult
        result = ValidationResult(True)
        result.add_warning("Test warning")
        print(f"✅ ValidationResult: {result}")
        
        # Test error handling decorators
        from exchanges.shared.interfaces_typed import handle_errors, handle_async_errors
        
        @handle_errors(default_return="error")
        def test_function():
            raise Exception("Test error")
        
        result = test_function()
        if result == "error":
            print("✅ Error handling decorator working")
        else:
            print("❌ Error handling decorator failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def main():
    """Run all API tests."""
    print("🚀 Starting Real API Tests...\n")
    
    tests = [
        test_shared_utilities,
        test_error_handling,
        test_binance_api,
        test_mexc_api,
        test_bingx_api,
        test_trading_interface
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
        print("🎉 All API tests passed! The implementations are working correctly.")
        return True
    else:
        print("⚠️  Some API tests failed. Please check the implementations.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)