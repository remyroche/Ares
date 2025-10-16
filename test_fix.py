#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.steps.data_collection.unified_data_loader import UnifiedDataLoader

async def test_unified_data_loader():
    """Test the fixed UnifiedDataLoader."""
    try:
        loader = UnifiedDataLoader()
        print("Testing data loading...")

        # Test loading data
        data = await loader.load_unified_data('ETHUSDT', 'BINANCE', '15m')

        if data is not None:
            print(f"✅ Successfully loaded data with shape: {data.shape}")
            print(f"✅ Data is not empty: {not data.empty}")
            return True
        else:
            print("❌ Failed to load data")
            return False

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_unified_data_loader())
    print(f"Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
