#!/usr/bin/env python3
"""
Test script for Step 1.5 Unified Data Converter - Granularity Preservation

This script tests that the converter correctly preserves the original granularity
of each data type (klines at 1-minute, aggtrades at per-second/trade, futures at sparse intervals).
"""

import asyncio
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.steps.step1_5_data_converter import run_step


def create_test_data():
    """Create test data with different granularities."""
    print("📊 Creating test data with different granularities...")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    data_cache_dir = os.path.join(temp_dir, "data_cache")
    os.makedirs(data_cache_dir, exist_ok=True)

    print(f"   📁 Created data cache directory: {data_cache_dir}")

    # Base timestamp (2024-01-01 00:00:00 UTC)
    base_ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

    # 1. Klines data (1-minute intervals)
    print("   📈 Creating klines data (1-minute intervals)...")
    klines_data = []
    for i in range(1440):  # 24 hours of minutes
        ts = base_ts + timedelta(minutes=i)
        klines_data.append(
            {
                "timestamp": ts,
                "open": 50000 + i * 0.1,
                "high": 50000 + i * 0.1 + 10,
                "low": 50000 + i * 0.1 - 10,
                "close": 50000 + i * 0.1 + 5,
                "volume": 100 + i * 0.01,
            }
        )

    klines_df = pd.DataFrame(klines_data)
    klines_path = os.path.join(
        data_cache_dir, "klines_binance_BTCUSDT_1m_consolidated.parquet"
    )
    klines_df.to_parquet(klines_path, index=False)
    print(f"   ✅ Created {len(klines_df)} klines records at: {klines_path}")
    print(f"   📋 File exists: {os.path.exists(klines_path)}")

    # 2. Aggtrades data (per-second granularity)
    print("   💹 Creating aggtrades data (per-second granularity)...")
    aggtrades_data = []
    for i in range(86400):  # 24 hours of seconds
        ts = base_ts + timedelta(seconds=i)
        aggtrades_data.append(
            {
                "timestamp": ts,
                "price": 50000 + i * 0.01,
                "quantity": 0.1 + (i % 10) * 0.01,
                "is_buyer_maker": bool(i % 2),
                "agg_trade_id": i,
            }
        )

    aggtrades_df = pd.DataFrame(aggtrades_data)
    aggtrades_path = os.path.join(
        data_cache_dir, "aggtrades_binance_BTCUSDT_consolidated.parquet"
    )
    aggtrades_df.to_parquet(aggtrades_path, index=False)
    print(f"   ✅ Created {len(aggtrades_df)} aggtrades records at: {aggtrades_path}")
    print(f"   📋 File exists: {os.path.exists(aggtrades_path)}")

    # 3. Futures data (sparse intervals - every 8 hours)
    print("   📊 Creating futures data (sparse intervals - every 8 hours)...")
    futures_data = []
    for i in range(3):  # 3 funding periods in 24 hours
        ts = base_ts + timedelta(hours=i * 8)
        futures_data.append(
            {
                "timestamp": ts,
                "fundingRate": 0.0001 + i * 0.0001,
                "fundingTime": ts.timestamp() * 1000,
            }
        )

    futures_df = pd.DataFrame(futures_data)
    futures_path = os.path.join(
        data_cache_dir, "futures_binance_BTCUSDT_consolidated.parquet"
    )
    futures_df.to_parquet(futures_path, index=False)
    print(f"   ✅ Created {len(futures_df)} futures records at: {futures_path}")
    print(f"   📋 File exists: {os.path.exists(futures_path)}")

    # List all files in the directory
    print(f"   📋 All files in {data_cache_dir}:")
    for file in os.listdir(data_cache_dir):
        print(f"      - {file}")

    return temp_dir, data_cache_dir


async def test_granularity_preservation():
    """Test that granularity is preserved correctly."""
    print("\n🧪 Testing granularity preservation...")

    # Create test data
    temp_dir, data_cache_dir = create_test_data()

    try:
        # Run the converter
        print("\n🔄 Running Step 1.5 converter...")
        print(f"   📁 Data directory: {data_cache_dir}")
        print(f"   📋 Files in data directory:")
        for file in os.listdir(data_cache_dir):
            print(f"      - {file}")

        try:
            success = await run_step(
                symbol="BTCUSDT",
                exchange="binance",
                timeframe="1m",
                data_dir=data_cache_dir,
                force_rerun=True,
            )
            print(f"   ✅ run_step returned: {success}")
        except Exception as e:
            print(f"   ❌ Exception in run_step: {e}")
            import traceback

            traceback.print_exc()
            return False

        if not success:
            print("❌ Converter failed")
            return False

        # Check the results
        unified_dir = os.path.join(data_cache_dir, "unified")

        # 1. Check klines dataset (should be 1-minute intervals)
        print("\n📈 Checking klines granularity...")
        klines_path = os.path.join(unified_dir, "klines", "binance_BTCUSDT_1m")
        if os.path.exists(klines_path):
            # Read a sample to check granularity
            import pyarrow.parquet as pq

            table = pq.read_table(klines_path)
            df = table.to_pandas()

            if len(df) > 1:
                # Check that timestamps are 1 minute apart
                timestamps = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                time_diffs = timestamps.diff().dropna()
                expected_diff = pd.Timedelta(minutes=1)

                # Allow small tolerance for rounding
                tolerance = pd.Timedelta(seconds=1)
                is_correct_granularity = all(
                    abs(diff - expected_diff) <= tolerance for diff in time_diffs
                )

                if is_correct_granularity:
                    print(
                        f"   ✅ Klines granularity preserved: {len(df)} records at 1-minute intervals"
                    )
                else:
                    print(
                        f"   ❌ Klines granularity incorrect: {time_diffs.iloc[0]} intervals"
                    )
                    return False
            else:
                print("   ❌ Not enough klines data to check granularity")
                return False
        else:
            print("   ❌ Klines dataset not found")
            return False

        # 2. Check aggtrades dataset (should be per-second granularity)
        print("\n💹 Checking aggtrades granularity...")
        aggtrades_path = os.path.join(unified_dir, "aggtrades", "binance_BTCUSDT")
        if os.path.exists(aggtrades_path):
            table = pq.read_table(aggtrades_path)
            df = table.to_pandas()

            if len(df) > 1:
                # Check that timestamps are at second granularity (not forced to minutes)
                timestamps = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                time_diffs = timestamps.diff().dropna()

                # Should have many different time differences (not just 1 minute)
                unique_diffs = time_diffs.value_counts()

                if len(unique_diffs) > 1:  # Multiple different intervals
                    print(
                        f"   ✅ Aggtrades granularity preserved: {len(df)} records with original timestamps"
                    )
                    print(f"      Time intervals: {unique_diffs.head().to_dict()}")
                else:
                    print(
                        f"   ❌ Aggtrades granularity incorrect: all intervals are {time_diffs.iloc[0]}"
                    )
                    return False
            else:
                print("   ❌ Not enough aggtrades data to check granularity")
                return False
        else:
            print("   ❌ Aggtrades dataset not found")
            return False

        # 3. Check futures dataset (should be sparse intervals)
        print("\n📊 Checking futures granularity...")
        futures_path = os.path.join(unified_dir, "futures", "binance_BTCUSDT")
        if os.path.exists(futures_path):
            table = pq.read_table(futures_path)
            df = table.to_pandas()

            if len(df) > 1:
                # Check that timestamps are sparse (8 hours apart)
                timestamps = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                time_diffs = timestamps.diff().dropna()
                expected_diff = pd.Timedelta(hours=8)

                # Allow small tolerance
                tolerance = pd.Timedelta(minutes=1)
                is_correct_granularity = all(
                    abs(diff - expected_diff) <= tolerance for diff in time_diffs
                )

                if is_correct_granularity:
                    print(
                        f"   ✅ Futures granularity preserved: {len(df)} records at 8-hour intervals"
                    )
                else:
                    print(
                        f"   ❌ Futures granularity incorrect: {time_diffs.iloc[0]} intervals"
                    )
                    return False
            else:
                print("   ❌ Not enough futures data to check granularity")
                return False
        else:
            print("   ❌ Futures dataset not found")
            return False

        # 4. Check metadata
        print("\n📋 Checking metadata...")
        metadata_path = os.path.join(unified_dir, "metadata.json")
        if os.path.exists(metadata_path):
            import json

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            print("   ✅ Metadata created successfully")
            print(f"      Klines: {metadata['data_types']['klines']['rows']} rows")
            print(
                f"      Aggtrades: {metadata['data_types']['aggtrades']['rows']} rows"
            )
            print(f"      Futures: {metadata['data_types']['futures']['rows']} rows")
        else:
            print("   ❌ Metadata not found")
            return False

        print("\n🎉 All granularity tests passed!")
        return True

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")


async def test_missing_data_handling():
    """Test handling of missing data sources."""
    print("\n🧪 Testing missing data handling...")

    # Create test data with only klines
    temp_dir = tempfile.mkdtemp()
    data_cache_dir = os.path.join(temp_dir, "data_cache")
    os.makedirs(data_cache_dir, exist_ok=True)

    # Only create klines data
    base_ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    klines_data = []
    for i in range(100):  # 100 minutes
        ts = base_ts + timedelta(minutes=i)
        klines_data.append(
            {
                "timestamp": ts,
                "open": 50000 + i * 0.1,
                "high": 50000 + i * 0.1 + 10,
                "low": 50000 + i * 0.1 - 10,
                "close": 50000 + i * 0.1 + 5,
                "volume": 100 + i * 0.01,
            }
        )

    klines_df = pd.DataFrame(klines_data)
    klines_path = os.path.join(
        data_cache_dir, "klines_binance_BTCUSDT_1m_consolidated.parquet"
    )
    klines_df.to_parquet(klines_path, index=False)

    try:
        # Run converter with missing aggtrades and futures
        print("   🔄 Running converter with missing aggtrades and futures...")
        success = await run_step(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1m",
            data_dir=data_cache_dir,
            force_rerun=True,
        )

        if success:
            print("   ✅ Converter handled missing data gracefully")

            # Check that only klines dataset was created
            unified_dir = os.path.join(data_cache_dir, "unified")
            klines_exists = os.path.exists(
                os.path.join(unified_dir, "klines", "binance_BTCUSDT_1m")
            )
            aggtrades_exists = os.path.exists(
                os.path.join(unified_dir, "aggtrades", "binance_BTCUSDT")
            )
            futures_exists = os.path.exists(
                os.path.join(unified_dir, "futures", "binance_BTCUSDT")
            )

            if klines_exists and not aggtrades_exists and not futures_exists:
                print("   ✅ Only klines dataset created (as expected)")
                return True
            else:
                print("   ❌ Unexpected datasets created")
                return False
        else:
            print("   ❌ Converter failed with missing data")
            return False

    finally:
        shutil.rmtree(temp_dir)


async def main():
    """Run all tests."""
    print("🚀 Step 1.5 Granularity Preservation Tests")
    print("=" * 50)

    # Test 1: Granularity preservation
    test1_passed = await test_granularity_preservation()

    # Test 2: Missing data handling
    test2_passed = await test_missing_data_handling()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(
        f"   Granularity Preservation: {'✅ PASSED' if test1_passed else '❌ FAILED'}"
    )
    print(f"   Missing Data Handling: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print(
            "\n🎉 All tests passed! Step 1.5 converter correctly preserves granularity."
        )
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
