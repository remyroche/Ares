#!/usr/bin/env python3
"""
Very simple test to see if the system is working
"""

import asyncio
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


async def main():
    print("🚀 Starting very simple test...")

    try:
        # Test 1: Just import the manager
        print("📦 Testing import...")
        from src.utils.hmm_composite_manager import get_hmm_composite_manager

        print("✅ Import successful")

        # Test 2: Get the manager
        print("🔍 Getting manager...")
        hmm_manager = get_hmm_composite_manager()
        print("✅ Manager created")

        # Test 3: Test file paths
        print("🔍 Testing file paths...")
        file_paths = hmm_manager._get_file_paths(
            "BINANCE", "ETHUSDT", "1m", "data/training"
        )
        print(f"✅ File paths: {file_paths}")

        # Test 4: Test file existence check
        print("🔍 Testing file existence...")
        all_exist, missing_files = hmm_manager._check_files_exist(file_paths)
        print(f"✅ All exist: {all_exist}, Missing: {missing_files}")

        print("✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
