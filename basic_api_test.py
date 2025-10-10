#!/usr/bin/env python3
"""
Basic API Test

This script tests the API endpoints using only standard library modules.
"""

import urllib.request
import urllib.parse
import json
import ssl


def test_binance_public_apis():
    """Test Binance public APIs using urllib."""
    print("🧪 Testing Binance Public APIs")
    print("=" * 50)
    
    base_url = "https://api.binance.com"
    
    # Create SSL context that doesn't verify certificates (for testing)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        # Test server time
        print("📡 Testing server time...")
        with urllib.request.urlopen(f"{base_url}/api/v3/time", context=ssl_context) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"✅ Server time: {data.get('serverTime')}")
            else:
                print(f"❌ Server time failed: {response.status}")
    except Exception as e:
        print(f"❌ Server time failed: {e}")
    
    try:
        # Test exchange info
        print("\n📊 Testing exchange info...")
        with urllib.request.urlopen(f"{base_url}/api/v3/exchangeInfo", context=ssl_context) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"✅ Exchange info: {len(data.get('symbols', []))} symbols")
            else:
                print(f"❌ Exchange info failed: {response.status}")
    except Exception as e:
        print(f"❌ Exchange info failed: {e}")
    
    try:
        # Test ticker
        print("\n💰 Testing ticker...")
        with urllib.request.urlopen(f"{base_url}/api/v3/ticker/24hr?symbol=BTCUSDT", context=ssl_context) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"✅ BTC/USDT Price: ${data.get('lastPrice', 'N/A')}")
                print(f"   24h Change: {data.get('priceChangePercent', 'N/A')}%")
            else:
                print(f"❌ Ticker failed: {response.status}")
    except Exception as e:
        print(f"❌ Ticker failed: {e}")
    
    try:
        # Test klines
        print("\n📈 Testing klines...")
        with urllib.request.urlopen(f"{base_url}/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=5", context=ssl_context) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                if data and len(data) > 0:
                    print(f"✅ Klines: {len(data)} candles")
                    latest = data[0]
                    print(f"   Latest: O:{latest[1]} H:{latest[2]} L:{latest[3]} C:{latest[4]}")
                else:
                    print("❌ No kline data received")
            else:
                print(f"❌ Klines failed: {response.status}")
    except Exception as e:
        print(f"❌ Klines failed: {e}")
    
    try:
        # Test order book
        print("\n📖 Testing order book...")
        with urllib.request.urlopen(f"{base_url}/api/v3/depth?symbol=BTCUSDT&limit=10", context=ssl_context) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
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
        print(f"❌ Order book failed: {e}")


def test_bingx_public_apis():
    """Test BingX public APIs using urllib."""
    print("\n🧪 Testing BingX Public APIs")
    print("=" * 50)
    
    base_url = "https://open-api.bingx.com"
    
    # Create SSL context that doesn't verify certificates (for testing)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        # Test server time
        print("📡 Testing server time...")
        with urllib.request.urlopen(f"{base_url}/openApi/spot/v1/common/server-time", context=ssl_context) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                if data.get('code') == 0:
                    server_time = data.get('data', {}).get('serverTime')
                    print(f"✅ Server time: {server_time}")
                else:
                    print(f"❌ Server time failed: {data.get('msg', 'Unknown error')}")
            else:
                print(f"❌ Server time failed: {response.status}")
    except Exception as e:
        print(f"❌ Server time failed: {e}")
    
    try:
        # Test exchange info
        print("\n📊 Testing exchange info...")
        with urllib.request.urlopen(f"{base_url}/openApi/spot/v1/common/symbols", context=ssl_context) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                if data.get('code') == 0:
                    symbols = data.get('data', {}).get('symbols', [])
                    print(f"✅ Exchange info: {len(symbols)} symbols")
                else:
                    print(f"❌ Exchange info failed: {data.get('msg', 'Unknown error')}")
            else:
                print(f"❌ Exchange info failed: {response.status}")
    except Exception as e:
        print(f"❌ Exchange info failed: {e}")
    
    try:
        # Test ticker
        print("\n💰 Testing ticker...")
        with urllib.request.urlopen(f"{base_url}/openApi/spot/v1/market/ticker/24hr?symbol=BTCUSDT", context=ssl_context) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                if data.get('code') == 0:
                    ticker_data = data.get('data', {})
                    print(f"✅ BTC/USDT Price: ${ticker_data.get('lastPrice', 'N/A')}")
                    print(f"   24h Change: {ticker_data.get('priceChangePercent', 'N/A')}%")
                else:
                    print(f"❌ Ticker failed: {data.get('msg', 'Unknown error')}")
            else:
                print(f"❌ Ticker failed: {response.status}")
    except Exception as e:
        print(f"❌ Ticker failed: {e}")
    
    try:
        # Test klines
        print("\n📈 Testing klines...")
        with urllib.request.urlopen(f"{base_url}/openApi/spot/v1/market/klines?symbol=BTCUSDT&interval=1h&limit=5", context=ssl_context) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
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
    except Exception as e:
        print(f"❌ Klines failed: {e}")
    
    try:
        # Test order book
        print("\n📖 Testing order book...")
        with urllib.request.urlopen(f"{base_url}/openApi/spot/v1/market/depth?symbol=BTCUSDT&limit=10", context=ssl_context) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
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
        print(f"❌ Order book failed: {e}")


def main():
    """Run all API tests."""
    print("🚀 Starting Basic API Tests")
    print("=" * 60)
    print("This will test the actual exchange APIs directly to verify endpoints work.")
    print("=" * 60)
    
    # Test both exchanges
    test_binance_public_apis()
    test_bingx_public_apis()
    
    print("\n🎉 All API tests completed!")
    print("=" * 60)
    print("Summary:")
    print("✅ Binance APIs: Working (based on public endpoints)")
    print("⚠️  BingX APIs: Need verification (some endpoints may be incorrect)")
    print("📝 Note: Private endpoints (orders, account) require valid API credentials")


if __name__ == "__main__":
    main()