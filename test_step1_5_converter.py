#!/usr/bin/env python3
"""
Test script for step1.5_data_converter.py

This script tests the unified data converter functionality.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_step1_5_converter():
    """Test the step1.5 data converter."""
    try:
        print("🧪 Testing Step 1.5: Unified Data Converter")
        print("=" * 60)

        # Import the converter
        from src.training.steps.step1_5_data_converter import run_step

        # Test parameters
        symbol = "ETHUSDT"
        exchange = "BINANCE"
        timeframe = "1m"
        data_dir = "data_cache"
        force_rerun = True  # Force re-run for testing

        print(f"📊 Test Parameters:")
        print(f"   Symbol: {symbol}")
        print(f"   Exchange: {exchange}")
        print(f"   Timeframe: {timeframe}")
        print(f"   Data Directory: {data_dir}")
        print(f"   Force Re-run: {force_rerun}")
        print()

        # Check if source data exists
        source_files = [
            f"data_cache/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet",
            f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet",
            f"data_cache/futures_{exchange}_{symbol}_consolidated.parquet",
        ]

        print("📁 Checking source data files:")
        for file_path in source_files:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   ✅ {file_path} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {file_path} (not found)")
        print()

        # Run the converter
        print("🔄 Running Step 1.5 converter...")
        success = await run_step(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force_rerun,
        )

        if success:
            print("✅ Step 1.5 converter completed successfully!")

            # Check output
            unified_path = f"data_cache/unified/{exchange.lower()}/{symbol}/{timeframe}"
            if os.path.exists(unified_path):
                print(f"📁 Unified dataset created at: {unified_path}")

                # Count parquet files
                parquet_files = []
                for root, dirs, files in os.walk(unified_path):
                    for file in files:
                        if file.endswith(".parquet"):
                            parquet_files.append(os.path.join(root, file))

                print(f"📊 Found {len(parquet_files)} parquet files in unified dataset")

                # Check metadata
                metadata_path = os.path.join(unified_path, "metadata.json")
                if os.path.exists(metadata_path):
                    print(f"📋 Metadata file created: {metadata_path}")

                # Check config
                config_path = f"data_cache/unified/{exchange.lower()}_{symbol}_{timeframe}_config.json"
                if os.path.exists(config_path):
                    print(f"⚙️ Configuration file created: {config_path}")

            else:
                print(f"❌ Unified dataset not found at: {unified_path}")

        else:
            print("❌ Step 1.5 converter failed!")
            return False

        print()
        print("🎉 Test completed!")
        return True

    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_step1_5_converter())

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
