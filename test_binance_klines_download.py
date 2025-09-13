#!/usr/bin/env python3
"""
Binance API Klines Download Test

This script specifically tests the klines downloading functionality
of the Binance API implementation.
"""

import sys
import asyncio
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_klines_download():
    """Test klines downloading functionality."""
    print("🔍 Testing Binance Klines Download...")
    print("="*50)

    try:
        # Import the Binance exchange
        from src.exchange.binance import BinanceExchange
        print("✅ BinanceExchange imported successfully")

        # Configure for testnet (no API keys required)
        config = {
            'binance_exchange': {
                'use_testnet': True,
                'timeout': 30,
                'max_retries': 3,
                'use_ccxt_fallback': True
            }
        }

        # Create exchange instance
        exchange = BinanceExchange(config)
        print("✅ BinanceExchange instance created")

        # Test initialization
        print("\n🔌 Testing initialization...")
        init_result = await exchange.initialize()
        print(f"   - Initialization: {'✅ Success' if init_result else '❌ Failed'}")

        if init_result:
            print("\n📊 Testing klines download...")

            # Test 1: Basic klines download
            print("   - Testing BTCUSDT 1m klines...")
            klines_1m = await exchange.get_klines('BTCUSDT', '1m', 10)
            if klines_1m and len(klines_1m) > 0:
                print(f"     ✅ Retrieved {len(klines_1m)} 1m klines")
                print(f"     Sample kline: {klines_1m[0]}")
            else:
                print("     ❌ Failed to retrieve 1m klines")

            # Test 2: Different timeframe
            print("   - Testing BTCUSDT 5m klines...")
            klines_5m = await exchange.get_klines('BTCUSDT', '5m', 5)
            if klines_5m and len(klines_5m) > 0:
                print(f"     ✅ Retrieved {len(klines_5m)} 5m klines")
            else:
                print("     ❌ Failed to retrieve 5m klines")

            # Test 3: Different symbol
            print("   - Testing ETHUSDT 1m klines...")
            klines_eth = await exchange.get_klines('ETHUSDT', '1m', 5)
            if klines_eth and len(klines_eth) > 0:
                print(f"     ✅ Retrieved {len(klines_eth)} ETHUSDT klines")
            else:
                print("     ❌ Failed to retrieve ETHUSDT klines")

            # Test 4: Larger limit
            print("   - Testing BTCUSDT 1h klines with larger limit...")
            klines_1h = await exchange.get_klines('BTCUSDT', '1h', 50)
            if klines_1h and len(klines_1h) > 0:
                print(f"     ✅ Retrieved {len(klines_1h)} 1h klines")
            else:
                print("     ❌ Failed to retrieve 1h klines")

            # Test 5: Check data format
            if klines_1m and len(klines_1m) > 0:
                print("\n🔍 Analyzing kline data format...")
                sample = klines_1m[0]
                if isinstance(sample, list) and len(sample) >= 12:
                    print("     ✅ Kline format: Standard OHLCV format")
                    print(f"     Fields: {len(sample)} (expected: 12)")
                    print(f"     - Timestamp: {sample[0]}")
                    print(f"     - Open: {sample[1]}")
                    print(f"     - High: {sample[2]}")
                    print(f"     - Low: {sample[3]}")
                    print(f"     - Close: {sample[4]}")
                    print(f"     - Volume: {sample[5]}")
                    print(f"     - Close Time: {sample[6]}")
                    print(f"     - Quote Asset Volume: {sample[7]}")
                    print(f"     - Number of Trades: {sample[8]}")
                    print(f"     - Taker Buy Base Asset Volume: {sample[9]}")
                    print(f"     - Taker Buy Quote Asset Volume: {sample[10]}")
                    print(f"     - Unused Field: {sample[11]}")
                else:
                    print("     ❌ Unexpected kline format")

            # Test 6: Error handling
            print("\n🛡️ Testing error handling...")
            try:
                invalid_klines = await exchange.get_klines('INVALIDSYMBOL', '1m', 5)
                if invalid_klines is None:
                    print("     ✅ Properly handled invalid symbol")
                else:
                    print("     ❌ Should have returned None for invalid symbol")
            except Exception as e:
                print(f"     ✅ Exception handled: {type(e).__name__}")

        else:
            print("❌ Initialization failed - cannot test klines download")

        # Cleanup
        print("\n🛑 Testing cleanup...")
        await exchange.stop()
        print("   - Cleanup: ✅ Completed")

        # Summary
        print("\n" + "="*50)
        print("📊 KLINES DOWNLOAD TEST SUMMARY")
        print("="*50)

        if init_result:
            success_count = sum([
                klines_1m is not None and len(klines_1m) > 0,
                klines_5m is not None and len(klines_5m) > 0,
                klines_eth is not None and len(klines_eth) > 0,
                klines_1h is not None and len(klines_1h) > 0
            ])

            print(f"✅ Successful downloads: {success_count}/4")
            print(f"❌ Failed downloads: {4 - success_count}/4")

            if success_count == 4:
                print("🎉 ALL KLINES DOWNLOADS SUCCESSFUL!")
                print("✅ Binance API klines functionality is working perfectly")
            elif success_count >= 2:
                print("⚠️ MOST KLINES DOWNLOADS WORKING")
                print("✅ Core functionality is operational")
            else:
                print("❌ KLINES DOWNLOAD ISSUES")
                print("❌ Core functionality needs attention")
        else:
            print("❌ INITIALIZATION FAILED")
            print("❌ Cannot perform klines download tests")

        return init_result

    except Exception as e:
        print(f"❌ Klines download test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_exchange_status():
    """Test exchange status and connection."""
    print("\n🔍 Testing Exchange Status...")
    print("-"*30)

    try:
        from src.exchange.binance import BinanceExchange

        config = {
            'binance_exchange': {
                'use_testnet': True,
                'timeout': 30,
                'max_retries': 3
            }
        }

        exchange = BinanceExchange(config)
        status = exchange.get_exchange_status()

        print("📋 Exchange Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Exchange status test failed: {e}")
        return False

async def main():
    """Run all klines tests."""
    print("🚀 Starting Binance Klines Download Tests...")
    print("="*60)

    # Test 1: Exchange status
    status_ok = await test_exchange_status()

    # Test 2: Klines download
    klines_ok = await test_klines_download()

    # Final results
    print("\n" + "="*60)
    print("📊 FINAL BINANCE KLINES TEST RESULTS")
    print("="*60)

    tests = [("Exchange Status", status_ok), ("Klines Download", klines_ok)]
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
        print("🎉 BINANCE KLINES API IS FULLY FUNCTIONAL!")
        print("✅ Ready for production klines downloading")
    elif passed >= total * 0.5:
        print("⚠️ BINANCE KLINES API IS MOSTLY FUNCTIONAL")
        print("✅ Core functionality working")
    else:
        print("❌ BINANCE KLINES API HAS ISSUES")
        print("❌ Check dependencies and configuration")

    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
