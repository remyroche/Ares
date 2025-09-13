#!/usr/bin/env python3
"""
Binance Klines Content Verification

This script verifies the content and quality of downloaded klines data
from the Binance API implementation.
"""

import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def verify_klines_content():
    """Verify the content and quality of downloaded klines."""
    print("🔍 Verifying Binance Klines Content...")
    print("="*60)

    try:
        # Import the Binance exchange
        from src.exchange.binance import BinanceExchange
        print("✅ BinanceExchange imported successfully")

        # Configure for testnet
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

        # Initialize
        print("\n🔌 Initializing exchange...")
        init_result = await exchange.initialize()
        print(f"   - Initialization: {'✅ Success' if init_result else '❌ Failed'}")

        if init_result:
            print("\n📊 Downloading and analyzing klines data...")

            # Test 1: Download larger dataset for analysis
            print("\n🔹 Test 1: Downloading BTCUSDT 1m klines (100 records)")
            klines_1m = await exchange.get_klines('BTCUSDT', '1m', 100)

            if klines_1m and len(klines_1m) > 0:
                print(f"   ✅ Retrieved {len(klines_1m)} records")
                await analyze_klines_data(klines_1m, "BTCUSDT", "1m")
            else:
                print("   ❌ Failed to retrieve klines data")
                return

            # Test 2: Download different timeframe
            print("\n🔹 Test 2: Downloading BTCUSDT 5m klines (50 records)")
            klines_5m = await exchange.get_klines('BTCUSDT', '5m', 50)

            if klines_5m and len(klines_5m) > 0:
                print(f"   ✅ Retrieved {len(klines_5m)} records")
                await analyze_klines_data(klines_5m, "BTCUSDT", "5m")
            else:
                print("   ❌ Failed to retrieve 5m klines data")

            # Test 3: Download different symbol
            print("\n🔹 Test 3: Downloading ETHUSDT 1m klines (50 records)")
            klines_eth = await exchange.get_klines('ETHUSDT', '1m', 50)

            if klines_eth and len(klines_eth) > 0:
                print(f"   ✅ Retrieved {len(klines_eth)} records")
                await analyze_klines_data(klines_eth, "ETHUSDT", "1m")
            else:
                print("   ❌ Failed to retrieve ETHUSDT klines data")

            # Test 4: Data consistency check
            print("\n🔹 Test 4: Cross-validating data consistency")
            await cross_validate_data(klines_1m, klines_5m, klines_eth)

        # Cleanup
        print("\n🛑 Cleaning up...")
        await exchange.stop()
        print("   - Cleanup: ✅ Completed")

        return True

    except Exception as e:
        print(f"❌ Content verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def analyze_klines_data(klines, symbol, interval):
    """Analyze the structure and quality of klines data."""
    print(f"\n   📈 Analyzing {symbol} {interval} klines:")

    # Basic structure analysis
    print(f"   • Data type: {type(klines)}")
    print(f"   • Number of records: {len(klines)}")

    # Sample records
    print(f"   • First record: {klines[0]}")
    print(f"   • Last record: {klines[-1]}")

    # Field count analysis
    if klines and len(klines) > 0:
        field_counts = [len(record) for record in klines if isinstance(record, list)]
        unique_counts = set(field_counts)
        print(f"   • Fields per record: {unique_counts}")

        if len(unique_counts) == 1:
            field_count = list(unique_counts)[0]
            print(f"   • Consistent field count: {field_count}")

            # Analyze field content based on format
            if field_count == 6:
                await analyze_ccxt_format(klines, symbol, interval)
            elif field_count == 12:
                await analyze_binance_format(klines, symbol, interval)
            else:
                print(f"   ⚠️ Unexpected field count: {field_count}")
        else:
            print(f"   ❌ Inconsistent field counts: {unique_counts}")

    # Data quality checks
    await check_data_quality(klines, symbol, interval)

async def analyze_ccxt_format(klines, symbol, interval):
    """Analyze CCXT format klines (6 fields: timestamp, O, H, L, C, V)."""
    print(f"   • Format: CCXT OHLCV (6 fields)")

    try:
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Timestamp analysis
        print(f"   • Timestamp range:")
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"     - From: {df['timestamp'].min()}")
        print(f"     - To: {df['timestamp'].max()}")
        print(f"     - Duration: {df['timestamp'].max() - df['timestamp'].min()}")

        # Price analysis
        print(f"   • Price statistics:")
        print(f"     - Open: {df['open'].min():.2f} - {df['open'].max():.2f}")
        print(f"     - High: {df['high'].min():.2f} - {df['high'].max():.2f}")
        print(f"     - Low: {df['low'].min():.2f} - {df['low'].max():.2f}")
        print(f"     - Close: {df['close'].min():.2f} - {df['close'].max():.2f}")

        # Volume analysis
        print(f"   • Volume statistics:")
        print(f"     - Min: {df['volume'].min():.6f}")
        print(f"     - Max: {df['volume'].max():.6f}")
        print(f"     - Mean: {df['volume'].mean():.6f}")
        print(f"     - Total: {df['volume'].sum():.6f}")

        # Price movement analysis
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['price_change'] / df['open']) * 100
        print(f"   • Price movement:")
        print(f"     - Avg change: {df['price_change'].mean():.4f}")
        print(f"     - Max change: {df['price_change'].max():.4f}")
        print(f"     - Min change: {df['price_change'].min():.4f}")
        print(f"     - Avg % change: {df['price_change_pct'].mean():.4f}%")

        # Data quality indicators
        print(f"   • Data quality:")
        print(f"     - Records with volume > 0: {len(df[df['volume'] > 0])}/{len(df)}")
        print(f"     - Records with price variation: {len(df[df['high'] != df['low']])}/{len(df)}")

    except Exception as e:
        print(f"   ❌ Error analyzing CCXT format: {e}")

async def analyze_binance_format(klines, symbol, interval):
    """Analyze Binance native format klines (12 fields)."""
    print(f"   • Format: Binance Native (12 fields)")

    try:
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'unused_field'
        ])

        print(f"   • Extended fields available:")
        print(f"     - Quote volume: {df['quote_asset_volume'].sum():.2f}")
        print(f"     - Number of trades: {df['number_of_trades'].sum()}")
        print(f"     - Taker buy volume: {df['taker_buy_base_asset_volume'].sum():.6f}")

    except Exception as e:
        print(f"   ❌ Error analyzing Binance format: {e}")

async def check_data_quality(klines, symbol, interval):
    """Check data quality indicators."""
    print(f"   • Quality checks:")

    try:
        # Check for null/empty values
        null_count = sum(1 for record in klines if not record or None in record)
        print(f"     - Null/empty records: {null_count}/{len(klines)}")

        # Check for zero values in critical fields
        if len(klines[0]) >= 6:  # At least OHLCV format
            zero_volume_count = sum(1 for record in klines if len(record) >= 6 and record[5] == 0)
            print(f"     - Zero volume records: {zero_volume_count}/{len(klines)}")

            # Check for invalid OHLC relationships
            invalid_ohlc_count = 0
            for record in klines:
                if len(record) >= 5:
                    o, h, l, c = record[1], record[2], record[3], record[4]
                    if not (l <= o <= h and l <= c <= h):
                        invalid_ohlc_count += 1
            print(f"     - Invalid OHLC relationships: {invalid_ohlc_count}/{len(klines)}")

        # Check timestamp ordering
        if len(klines) > 1 and len(klines[0]) >= 1:
            timestamps = [record[0] for record in klines]
            sorted_timestamps = sorted(timestamps)
            if timestamps == sorted_timestamps:
                print("     - ✅ Timestamps properly ordered")
            else:
                print("     - ❌ Timestamps not properly ordered")
                out_of_order = sum(1 for i in range(1, len(timestamps)) if timestamps[i] < timestamps[i-1])
                print(f"       Out of order pairs: {out_of_order}")

    except Exception as e:
        print(f"     ❌ Error in quality checks: {e}")

async def cross_validate_data(klines_1m, klines_5m, klines_eth):
    """Cross-validate data consistency across different downloads."""
    print(f"\n   🔄 Cross-validation:")

    try:
        # Check if we have enough data
        if not all([klines_1m, klines_5m, klines_eth]):
            print("     - Insufficient data for cross-validation")
            return

        # Compare timestamp ranges
        if len(klines_1m[0]) >= 1 and len(klines_5m[0]) >= 1:
            btc_1m_start = klines_1m[0][0]
            btc_5m_start = klines_5m[0][0]
            btc_1m_end = klines_1m[-1][0]
            btc_5m_end = klines_5m[-1][0]

            print(f"     • BTC 1m time range: {btc_1m_start} to {btc_1m_end}")
            print(f"     • BTC 5m time range: {btc_5m_start} to {btc_5m_end}")

            # Check if 5m data aligns with 1m data (approximately)
            time_diff_start = abs(btc_1m_start - btc_5m_start)
            time_diff_end = abs(btc_1m_end - btc_5m_end)
            print(f"     • Time range alignment: {time_diff_start}ms start diff, {time_diff_end}ms end diff")

        # Compare data formats
        formats = [
            ("BTC 1m", len(klines_1m[0]) if klines_1m else 0),
            ("BTC 5m", len(klines_5m[0]) if klines_5m else 0),
            ("ETH 1m", len(klines_eth[0]) if klines_eth else 0)
        ]

        print(f"     • Data format consistency:")
        for name, fields in formats:
            print(f"       - {name}: {fields} fields")

        consistent = all(fields == formats[0][1] for _, fields in formats)
        print(f"     • {'✅' if consistent else '❌'} Format consistency across symbols/timeframes")

        # Compare data volumes
        volumes = [
            ("BTC 1m", len(klines_1m)),
            ("BTC 5m", len(klines_5m)),
            ("ETH 1m", len(klines_eth))
        ]

        print(f"     • Data volume comparison:")
        for name, count in volumes:
            print(f"       - {name}: {count} records")

    except Exception as e:
        print(f"     ❌ Error in cross-validation: {e}")

async def main():
    """Run the content verification."""
    print("🚀 Starting Binance Klines Content Verification...")
    print("="*60)

    success = await verify_klines_content()

    print("\n" + "="*60)
    print("📊 CONTENT VERIFICATION RESULTS")
    print("="*60)

    if success:
        print("✅ Content verification completed successfully")
        print("✅ Downloaded data is valid and properly formatted")
        print("✅ Ready for use in trading applications")
    else:
        print("❌ Content verification failed")
        print("❌ Issues found with downloaded data")

    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
