#!/usr/bin/env python3
"""
API Verification Test

This script tests the actual API implementations to verify they work correctly.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exchanges.binance import create_binance_exchange
from exchanges.bingx import create_bingx_exchange
from src.trading.integration.exchange_integration import create_binance_integration


async def test_binance_api():
    """Test Binance API implementation."""
    print("🧪 Testing Binance API Implementation")
    print("=" * 50)
    
    try:
        # Create Binance exchange (testnet mode)
        exchange = create_binance_exchange(
            api_key="",  # Empty for public endpoints
            api_secret="",
            trade_symbol="BTCUSDT"
        )
        
        # Initialize exchange
        await exchange._initialize_exchange()
        
        # Test server time (public endpoint)
        print("📡 Testing server time endpoint...")
        server_time = await exchange.get_server_time()
        if server_time and "serverTime" in server_time:
            print(f"✅ Server time: {server_time['serverTime']}")
        else:
            print(f"❌ Server time failed: {server_time}")
        
        # Test exchange info (public endpoint)
        print("\n📊 Testing exchange info endpoint...")
        exchange_info = await exchange.get_exchange_info()
        if exchange_info and "symbols" in exchange_info:
            print(f"✅ Exchange info: {len(exchange_info['symbols'])} symbols available")
        else:
            print(f"❌ Exchange info failed: {exchange_info}")
        
        # Test ticker (public endpoint)
        print("\n💰 Testing ticker endpoint...")
        ticker = await exchange.get_ticker("BTCUSDT")
        if ticker and "price" in ticker:
            print(f"✅ BTC/USDT Price: ${ticker['price']}")
        else:
            print(f"❌ Ticker failed: {ticker}")
        
        # Test klines (public endpoint)
        print("\n📈 Testing klines endpoint...")
        klines = await exchange.get_klines("BTCUSDT", "1h", limit=5)
        if klines and len(klines) > 0:
            print(f"✅ Klines: {len(klines)} candles retrieved")
            latest = klines[0]
            print(f"   Latest: O:{latest.open_price:.2f} H:{latest.high_price:.2f} L:{latest.low_price:.2f} C:{latest.close_price:.2f}")
        else:
            print(f"❌ Klines failed: {klines}")
        
        # Test order book (public endpoint)
        print("\n📖 Testing order book endpoint...")
        order_book = await exchange.get_order_book("BTCUSDT", limit=10)
        if order_book and "bids" in order_book and "asks" in order_book:
            print(f"✅ Order book: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
        else:
            print(f"❌ Order book failed: {order_book}")
        
        # Close exchange
        await exchange.close()
        print("\n✅ Binance API test completed")
        
    except Exception as e:
        print(f"❌ Binance API test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_bingx_api():
    """Test BingX API implementation."""
    print("\n🧪 Testing BingX API Implementation")
    print("=" * 50)
    
    try:
        # Create BingX exchange (testnet mode)
        exchange = create_bingx_exchange(
            api_key="",  # Empty for public endpoints
            api_secret="",
            trade_symbol="BTCUSDT"
        )
        
        # Initialize exchange
        await exchange._initialize_exchange()
        
        # Test server time (public endpoint)
        print("📡 Testing server time endpoint...")
        server_time = await exchange.get_server_time()
        if server_time and "serverTime" in server_time:
            print(f"✅ Server time: {server_time['serverTime']}")
        else:
            print(f"❌ Server time failed: {server_time}")
        
        # Test exchange info (public endpoint)
        print("\n📊 Testing exchange info endpoint...")
        exchange_info = await exchange.get_exchange_info()
        if exchange_info and "symbols" in exchange_info:
            print(f"✅ Exchange info: {len(exchange_info['symbols'])} symbols available")
        else:
            print(f"❌ Exchange info failed: {exchange_info}")
        
        # Test ticker (public endpoint)
        print("\n💰 Testing ticker endpoint...")
        ticker = await exchange.get_ticker("BTCUSDT")
        if ticker and "price" in ticker:
            print(f"✅ BTC/USDT Price: ${ticker['price']}")
        else:
            print(f"❌ Ticker failed: {ticker}")
        
        # Test klines (public endpoint)
        print("\n📈 Testing klines endpoint...")
        klines = await exchange.get_klines("BTCUSDT", "1h", limit=5)
        if klines and len(klines) > 0:
            print(f"✅ Klines: {len(klines)} candles retrieved")
            latest = klines[0]
            print(f"   Latest: O:{latest.open_price:.2f} H:{latest.high_price:.2f} L:{latest.low_price:.2f} C:{latest.close_price:.2f}")
        else:
            print(f"❌ Klines failed: {klines}")
        
        # Test order book (public endpoint)
        print("\n📖 Testing order book endpoint...")
        order_book = await exchange.get_order_book("BTCUSDT", limit=10)
        if order_book and "bids" in order_book and "asks" in order_book:
            print(f"✅ Order book: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
        else:
            print(f"❌ Order book failed: {order_book}")
        
        # Close exchange
        await exchange.close()
        print("\n✅ BingX API test completed")
        
    except Exception as e:
        print(f"❌ BingX API test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_integration():
    """Test the integration module."""
    print("\n🧪 Testing Integration Module")
    print("=" * 50)
    
    try:
        # Create integration
        integration = create_binance_integration(
            api_key="",
            api_secret="",
            testnet=True
        )
        
        # Test connection
        print("📡 Testing integration connection...")
        connected = await integration.connect()
        if connected:
            print("✅ Integration connected successfully")
        else:
            print("❌ Integration connection failed")
            return
        
        # Test ticker through integration
        print("\n💰 Testing ticker through integration...")
        ticker = await integration.get_ticker("BTCUSDT")
        if ticker:
            print(f"✅ Ticker: ${ticker.price:.2f}")
        else:
            print("❌ Ticker failed")
        
        # Test klines through integration
        print("\n📈 Testing klines through integration...")
        klines = await integration.get_klines("BTCUSDT", "1h", limit=5)
        if klines and len(klines) > 0:
            print(f"✅ Klines: {len(klines)} candles")
        else:
            print("❌ Klines failed")
        
        # Test status
        print("\n📋 Testing integration status...")
        status = integration.get_status()
        print(f"Status: {status}")
        
        # Disconnect
        await integration.disconnect()
        print("\n✅ Integration test completed")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all API tests."""
    print("🚀 Starting API Verification Tests")
    print("=" * 60)
    print("This will test the actual API implementations to verify they work.")
    print("=" * 60)
    
    # Test individual exchanges
    await test_binance_api()
    await test_bingx_api()
    
    # Test integration
    await test_integration()
    
    print("\n🎉 All API tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())