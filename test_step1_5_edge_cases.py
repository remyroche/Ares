#!/usr/bin/env python3
"""
Test script for step1.5_data_converter.py edge cases

This script tests the unified data converter with various data availability scenarios.
"""

import asyncio
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def create_test_data():
    """Create test data files to demonstrate different scenarios."""

    # Create data_cache directory
    os.makedirs("data_cache", exist_ok=True)

    # Scenario 1: Only klines data (no aggtrades, no futures)
    print("📊 Creating test data for different scenarios...")

    # Create klines data (1-minute intervals)
    klines_start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    klines_data = []

    for i in range(60):  # 60 minutes of data
        timestamp = klines_start + timedelta(minutes=i)
        klines_data.append(
            {
                "timestamp": timestamp,
                "open": 100.0 + i * 0.1,
                "high": 100.0 + i * 0.1 + 0.5,
                "low": 100.0 + i * 0.1 - 0.3,
                "close": 100.0 + i * 0.1 + 0.2,
                "volume": 1000.0 + i * 10,
            }
        )

    klines_df = pd.DataFrame(klines_data)
    klines_df.to_parquet(
        "data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet", index=False
    )
    print(f"   ✅ Created klines data: {len(klines_df)} rows")

    # Create aggtrades data (per-second granularity)
    aggtrades_data = []
    for i in range(60):  # 60 minutes
        for j in range(60):  # 60 seconds per minute
            timestamp = klines_start + timedelta(minutes=i, seconds=j)
            # Add some random trades within each second
            for k in range(np.random.randint(1, 5)):  # 1-4 trades per second
                aggtrades_data.append(
                    {
                        "timestamp": timestamp + timedelta(milliseconds=k * 200),
                        "price": 100.0 + i * 0.1 + j * 0.001 + k * 0.0001,
                        "quantity": np.random.uniform(0.1, 2.0),
                        "is_buyer_maker": np.random.choice([True, False]),
                        "agg_trade_id": i * 3600 + j * 60 + k,
                    }
                )

    aggtrades_df = pd.DataFrame(aggtrades_data)
    aggtrades_df.to_parquet(
        "data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet", index=False
    )
    print(f"   ✅ Created aggtrades data: {len(aggtrades_df)} rows")

    # Create futures data (every 8 hours - funding rate intervals)
    futures_data = []
    for i in range(8):  # 8 funding rate periods
        timestamp = klines_start + timedelta(hours=i * 8)
        futures_data.append(
            {
                "timestamp": timestamp,
                "fundingRate": np.random.uniform(-0.01, 0.01),  # -1% to +1%
            }
        )

    futures_df = pd.DataFrame(futures_data)
    futures_df.to_parquet(
        "data_cache/futures_BINANCE_ETHUSDT_consolidated.parquet", index=False
    )
    print(f"   ✅ Created futures data: {len(futures_df)} rows")

    return {"klines": klines_df, "aggtrades": aggtrades_df, "futures": futures_df}


async def test_scenario_1_no_futures():
    """Test scenario: No futures data available."""
    print("\n🧪 SCENARIO 1: No futures data")
    print("=" * 50)

    # Remove futures file to simulate missing data
    futures_file = "data_cache/futures_BINANCE_ETHUSDT_consolidated.parquet"
    if os.path.exists(futures_file):
        os.remove(futures_file)
        print("   🗑️ Removed futures file to simulate missing data")

    # Run converter
    from src.training.steps.step1_5_data_converter import run_step

    success = await run_step(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        force_rerun=True,
    )

    if success:
        print("   ✅ Converter completed successfully without futures data")

        # Check output
        unified_path = "data_cache/unified/binance/ETHUSDT/1m"
        if os.path.exists(unified_path):
            print(f"   📁 Unified dataset created at: {unified_path}")

            # Check if funding_rate column exists and is filled with zeros
            try:
                from src.training.enhanced_training_manager_optimized import (
                    ParquetDatasetManager,
                )

                pdm = ParquetDatasetManager()
                sample_data = pdm.scan_dataset(
                    unified_path, columns=["timestamp", "funding_rate"]
                )
                if "funding_rate" in sample_data.columns:
                    zero_count = (sample_data["funding_rate"] == 0).sum()
                    total_count = len(sample_data)
                    print(
                        f"   📊 Funding rate column: {zero_count}/{total_count} rows are zero (expected)"
                    )
            except Exception as e:
                print(f"   ⚠️ Could not verify funding rate column: {e}")
    else:
        print("   ❌ Converter failed without futures data")

    return success


async def test_scenario_2_no_aggtrades():
    """Test scenario: No aggtrades data available."""
    print("\n🧪 SCENARIO 2: No aggtrades data")
    print("=" * 50)

    # Remove aggtrades file to simulate missing data
    aggtrades_file = "data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet"
    if os.path.exists(aggtrades_file):
        os.remove(aggtrades_file)
        print("   🗑️ Removed aggtrades file to simulate missing data")

    # Run converter
    from src.training.steps.step1_5_data_converter import run_step

    success = await run_step(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        force_rerun=True,
    )

    if success:
        print("   ✅ Converter completed successfully without aggtrades data")

        # Check output
        unified_path = "data_cache/unified/binance/ETHUSDT/1m"
        if os.path.exists(unified_path):
            print(f"   📁 Unified dataset created at: {unified_path}")

            # Check if trade-related columns exist and are filled with zeros
            try:
                from src.training.enhanced_training_manager_optimized import (
                    ParquetDatasetManager,
                )

                pdm = ParquetDatasetManager()
                sample_data = pdm.scan_dataset(
                    unified_path, columns=["timestamp", "trade_count", "trade_volume"]
                )
                if "trade_count" in sample_data.columns:
                    zero_count = (sample_data["trade_count"] == 0).sum()
                    total_count = len(sample_data)
                    print(
                        f"   📊 Trade count column: {zero_count}/{total_count} rows are zero (expected)"
                    )
            except Exception as e:
                print(f"   ⚠️ Could not verify trade columns: {e}")
    else:
        print("   ❌ Converter failed without aggtrades data")

    return success


async def test_scenario_3_timestamp_granularity():
    """Test scenario: Verify aggtrades aggregation to kline boundaries."""
    print("\n🧪 SCENARIO 3: Timestamp granularity handling")
    print("=" * 50)

    # Recreate all test data
    test_data = create_test_data()

    # Run converter
    from src.training.steps.step1_5_data_converter import run_step

    success = await run_step(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        force_rerun=True,
    )

    if success:
        print("   ✅ Converter completed successfully with all data")

        # Analyze the results
        unified_path = "data_cache/unified/binance/ETHUSDT/1m"
        if os.path.exists(unified_path):
            try:
                from src.training.enhanced_training_manager_optimized import (
                    ParquetDatasetManager,
                )

                pdm = ParquetDatasetManager()

                # Get sample data
                sample_data = pdm.scan_dataset(
                    unified_path,
                    columns=[
                        "timestamp",
                        "trade_count",
                        "trade_volume",
                        "funding_rate",
                    ],
                )

                print(f"   📊 Unified dataset analysis:")
                print(f"      Total rows: {len(sample_data)}")
                print(
                    f"      Rows with trade data: {(sample_data['trade_count'] > 0).sum()}"
                )
                print(
                    f"      Rows with funding rates: {(sample_data['funding_rate'] != 0).sum()}"
                )

                # Check timestamp alignment
                timestamps = pd.to_datetime(
                    sample_data["timestamp"], unit="ms", utc=True
                )
                minute_boundaries = timestamps.dt.floor("1min")
                aligned_count = (timestamps == minute_boundaries).sum()
                print(
                    f"      Timestamps aligned to minute boundaries: {aligned_count}/{len(timestamps)}"
                )

                if aligned_count == len(timestamps):
                    print(
                        "      ✅ All timestamps properly aligned to minute boundaries"
                    )
                else:
                    print("      ⚠️ Some timestamps not aligned to minute boundaries")

            except Exception as e:
                print(f"   ⚠️ Could not analyze unified dataset: {e}")
    else:
        print("   ❌ Converter failed with all data")

    return success


async def main():
    """Run all test scenarios."""
    print("🧪 Testing Step 1.5 Data Converter Edge Cases")
    print("=" * 60)

    # Create test data
    test_data = create_test_data()

    # Test scenarios
    scenarios = [
        ("No Futures Data", test_scenario_1_no_futures),
        ("No Aggtrades Data", test_scenario_2_no_aggtrades),
        ("Timestamp Granularity", test_scenario_3_timestamp_granularity),
    ]

    results = {}
    for scenario_name, test_func in scenarios:
        try:
            result = await test_func()
            results[scenario_name] = result
        except Exception as e:
            print(f"   💥 {scenario_name} failed with error: {e}")
            results[scenario_name] = False

    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 30)
    for scenario_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {scenario_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All edge case tests passed!")
    else:
        print("\n⚠️ Some edge case tests failed!")

    return all_passed


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
