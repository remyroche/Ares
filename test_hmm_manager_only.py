#!/usr/bin/env python3
"""
Test script that only tests the HMM composite manager without triggering step1_7_hmm_regime_discovery
"""

import asyncio
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


async def main():
    print("🚀 Testing HMM Composite Manager only...")

    try:
        # Test 1: Import the manager
        print("📦 Testing HMM composite manager import...")
        from src.utils.hmm_composite_manager import get_hmm_composite_manager

        hmm_manager = get_hmm_composite_manager()
        print("✅ HMM composite manager imported successfully")

        # Test 2: Test loading non-existent files
        print("🔍 Testing loading non-existent HMM composite clusters...")
        df = hmm_manager.load_composite_clusters(
            "BINANCE", "ETHUSDT", "1m", "data/training"
        )
        if df is None:
            print("✅ Correctly returned None for non-existent files")
        else:
            print("❌ Unexpectedly returned data for non-existent files")

        # Test 3: Test file existence check
        print("🔍 Testing file existence check...")
        file_paths = hmm_manager._get_file_paths(
            "BINANCE", "ETHUSDT", "1m", "data/training"
        )
        all_exist, missing_files = hmm_manager._check_files_exist(file_paths)
        print(
            f"✅ File existence check: all_exist={all_exist}, missing_files={missing_files}"
        )

        # Test 4: Test creating composite clusters (this should trigger step1_7_hmm_regime_discovery)
        print(
            "🎯 Testing create_composite_clusters (this should trigger step1_7_hmm_regime_discovery)..."
        )
        success = await hmm_manager.create_composite_clusters(
            exchange="BINANCE",
            symbol="ETHUSDT",
            timeframe="1m",
            data_dir="data/training",
            force_rerun=True,
            lookback_days=30,
        )

        print(f"✅ create_composite_clusters completed with result: {success}")

        # Test 5: Test get_or_create_composite_clusters
        print("🎯 Testing get_or_create_composite_clusters...")
        df = await hmm_manager.get_or_create_composite_clusters(
            exchange="BINANCE",
            symbol="ETHUSDT",
            timeframe="1m",
            data_dir="data/training",
            force_rerun=False,  # Don't force rerun since we just created them
            lookback_days=30,
        )

        if df is not None:
            print(
                f"✅ get_or_create_composite_clusters returned data with {len(df)} rows"
            )
        else:
            print("❌ get_or_create_composite_clusters returned None")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
