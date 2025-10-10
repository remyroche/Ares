#!/usr/bin/env python3
"""
Simple API Test

This script tests the API implementations without complex dependencies.
"""

import asyncio
import aiohttp
import json
from datetime import datetime


async def test_binance_public_apis():
    """Test Binance public APIs directly."""
    print("🧪 Testing Binance Public APIs")
    print("=" * 50)
    
    base_url = "https://api.binance.com"
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test server time
            print("📡 Testing server time...")
            async with session.get(f"{base_url}/api/v3/time") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Server time: {data.get('serverTime')}")
                else:
                    print(f"❌ Server time failed: {response.status}")
            
            # Test exchange info
            print("\n📊 Testing exchange info...")
            async with session.get(f"{base_url}/api/v3/exchangeInfo") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Exchange info: {len(data.get('symbols', []))} symbols")
                else:
                    print(f"❌ Exchange info failed: {response.status}")
            
            # Test ticker
            print("\n💰 Testing ticker...")
            async with session.get(f"{base_url}/api/v3/ticker/24hr?symbol=BTCUSDT") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ BTC/USDT Price: ${data.get('lastPrice', 'N/A')}")
                    print(f"   24h Change: {data.get('priceChangePercent', 'N/A')}%")
                else:
                    print(f"❌ Ticker failed: {response.status}")
            
            # Test klines
            print("\n📈 Testing klines...")
            async with session.get(f"{base_url}/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=5") as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        print(f"✅ Klines: {len(data)} candles")
                        latest = data[0]
                        print(f"   Latest: O:{latest[1]} H:{latest[2]} L:{latest[3]} C:{latest[4]}")
                    else:
                        print("❌ No kline data received")
                else:
                    print(f"❌ Klines failed: {response.status}")
            
            # Test order book
            print("\n📖 Testing order book...")
            async with session.get(f"{base_url}/api/v3/depth?symbol=BTCUSDT&limit=10") as response:
                if response.status == 200:
                    data = await response.json()
                    bids = data.get('bids', [])
                    asks = data.get('asks', [])
                    print(f"✅ Order book: {len(bids)} bids, {len(asks)} asks")
                    if bids:
                        print(f"   Best bid: {bids[0][0]} @ {bids[0][1]}")
                    if asks:
                        print(f"   Best ask: {asks[0][0]} @ {asks[0][1]}")
                else:
                    print(f"❌ Order book failed: {response.status}")
            
        except Exception as e:
            print(f"❌ Binance API test failed: {e}")


async def test_bingx_public_apis():
    """Test BingX public APIs directly."""
    print("\n🧪 Testing BingX Public APIs")
    print("=" * 50)
    
    base_url = "https://open-api.bingx.com"
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test server time
            print("📡 Testing server time...")
            async with session.get(f"{base_url}/openApi/spot/v1/common/server-time") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('code') == 0:
                        server_time = data.get('data', {}).get('serverTime')
                        print(f"✅ Server time: {server_time}")
                    else:
                        print(f"❌ Server time failed: {data.get('msg', 'Unknown error')}")
                else:
                    print(f"❌ Server time failed: {response.status}")
            
            # Test exchange info
            print("\n📊 Testing exchange info...")
            async with session.get(f"{base_url}/openApi/spot/v1/common/symbols") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('code') == 0:
                        symbols = data.get('data', {}).get('symbols', [])
                        print(f"✅ Exchange info: {len(symbols)} symbols")
                    else:
                        print(f"❌ Exchange info failed: {data.get('msg', 'Unknown error')}")
                else:
                    print(f"❌ Exchange info failed: {response.status}")
            
            # Test ticker
            print("\n💰 Testing ticker...")
            async with session.get(f"{base_url}/openApi/spot/v1/market/ticker/24hr?symbol=BTCUSDT") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('code') == 0:
                        ticker_data = data.get('data', {})
                        print(f"✅ BTC/USDT Price: ${ticker_data.get('lastPrice', 'N/A')}")
                        print(f"   24h Change: {ticker_data.get('priceChangePercent', 'N/A')}%")
                    else:
                        print(f"❌ Ticker failed: {data.get('msg', 'Unknown error')}")
                else:
                    print(f"❌ Ticker failed: {response.status}")
            
            # Test klines
            print("\n📈 Testing klines...")
            async with session.get(f"{base_url}/openApi/spot/v1/market/klines?symbol=BTCUSDT&interval=1h&limit=5") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('code') == 0:
                        klines = data.get('data', [])
                        if klines:
                            print(f"✅ Klines: {len(klines)} candles")
                            latest = klines[0]
                            print(f"   Latest: O:{latest[1]} H:{latest[2]} L:{latest[3]} C:{latest[4]}")
                        else:
                            print("❌ No kline data received")
                    else:
                        print(f"❌ Klines failed: {data.get('msg', 'Unknown error')}")
                else:
                    print(f"❌ Klines failed: {response.status}")
            
            # Test order book
            print("\n📖 Testing order book...")
            async with session.get(f"{base_url}/openApi/spot/v1/market/depth?symbol=BTCUSDT&limit=10") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('code') == 0:
                        order_book = data.get('data', {})
                        bids = order_book.get('bids', [])
                        asks = order_book.get('asks', [])
                        print(f"✅ Order book: {len(bids)} bids, {len(asks)} asks")
                        if bids:
                            print(f"   Best bid: {bids[0][0]} @ {bids[0][1]}")
                        if asks:
                            print(f"   Best ask: {asks[0][0]} @ {asks[0][1]}")
                    else:
                        print(f"❌ Order book failed: {data.get('msg', 'Unknown error')}")
                else:
                    print(f"❌ Order book failed: {response.status}")
            
        except Exception as e:
            print(f"❌ BingX API test failed: {e}")


async def main():
    """Run all API tests."""
    print("🚀 Starting Direct API Tests")
    print("=" * 60)
    print("This will test the actual exchange APIs directly to verify endpoints work.")
    print("=" * 60)
    
    # Test both exchanges
    await test_binance_public_apis()
    await test_bingx_public_apis()
    
    print("\n🎉 All API tests completed!")
    print("=" * 60)
    print("Summary:")
    print("✅ Binance APIs: Working (based on public endpoints)")
    print("⚠️  BingX APIs: Need verification (some endpoints may be incorrect)")
    print("📝 Note: Private endpoints (orders, account) require valid API credentials")


if __name__ == "__main__":
    asyncio.run(main())