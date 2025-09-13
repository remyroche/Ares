#!/usr/bin/env python3
"""
Migration Demo: parquet_utils → klines_parquet

This script demonstrates how to migrate from utils/parquet_utils.py
to steps/data_collection/klines_data/klines_parquet.py

Run this script to see the migration in action.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def demo_old_approach():
    """Demonstrate the old approach using parquet_utils."""
    print("🔄 OLD APPROACH (parquet_utils)")
    print("=" * 50)

    try:
        from src.utils.parquet_utils import get_parquet_utils, safe_read_parquet

        print("✅ Successfully imported from utils.parquet_utils")

        # Get utils instance
        utils = get_parquet_utils()
        print(f"✅ Got utils instance: {type(utils)}")

        # Try to read a klines file (this would work but without klines-specific features)
        file_path = "historical_data/binance/ethusdt/raw/ethusdt_1m_2024_09.parquet"
        if Path(file_path).exists():
            print(f"📁 File exists: {file_path}")
            df = safe_read_parquet(file_path)
            if df is not None:
                print(f"✅ Successfully read {len(df)} rows, {len(df.columns)} columns")
            else:
                print("❌ Failed to read file")
        else:
            print(f"⚠️ File not found: {file_path}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print()

def demo_new_approach():
    """Demonstrate the new approach using klines_parquet."""
    print("🚀 NEW APPROACH (klines_parquet)")
    print("=" * 50)

    try:
        from src.steps.data_collection.klines_data import (
            get_parquet_utils,
            safe_read_parquet,
            get_klines_manager
        )

        print("✅ Successfully imported from steps.data_collection.klines_data")

        # Method 1: Use backward compatibility (same API as parquet_utils)
        print("\n📋 Method 1: Backward Compatibility")
        utils = get_parquet_utils()
        print(f"✅ Got utils instance: {type(utils)}")

        # Try to read a klines file (same API, enhanced functionality)
        file_path = "historical_data/binance/ethusdt/raw/ethusdt_1m_2024_09.parquet"
        if Path(file_path).exists():
            print(f"📁 File exists: {file_path}")
            df = safe_read_parquet(file_path)
            if df is not None:
                print(f"✅ Successfully read {len(df)} rows, {len(df.columns)} columns")
                print(f"📊 Columns: {list(df.columns)}")
            else:
                print("❌ Failed to read file")
        else:
            print(f"⚠️ File not found: {file_path}")

        # Method 2: Use enhanced klines manager (new functionality)
        print("\n📋 Method 2: Enhanced Klines Manager")
        manager = get_klines_manager()
        print(f"✅ Got klines manager: {type(manager)}")

        # Get data info (new feature)
        info = manager.get_data_info("ETHUSDT", "1m", "raw")
        print("📊 Data info:")
        print(f"   Available: {info['available']}")
        print(f"   Files: {info['files_count']}")
        print(f"   Records: {info['total_records']:,}")
        if info['date_range']:
            start, end = info['date_range']
            print(f"   Date range: {start} to {end}")
        print(f"   Size: {info['file_size_mb']:.1f} MB")
        # Read data using enhanced method (new feature)
        df = manager.read_data("ETHUSDT", "1m")
        if df is not None:
            print(f"✅ Successfully read {len(df)} rows using enhanced method")
        else:
            print("❌ Failed to read using enhanced method")

        # List available data (new feature)
        available = manager.list_available_data()
        print(f"📋 Available data: {available}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print()

def demo_migration_comparison():
    """Compare old vs new approach side by side."""
    print("⚖️ MIGRATION COMPARISON")
    print("=" * 50)

    print("OLD WAY:")
    print("```python")
    print("from src.utils.parquet_utils import get_parquet_utils, safe_read_parquet")
    print("")
    print("utils = get_parquet_utils()")
    print("df = utils.safe_read_parquet('historical_data/binance/ethusdt/raw/ethusdt_1m_2024_09.parquet')")
    print("```")
    print()

    print("NEW WAY (Backward Compatible):")
    print("```python")
    print("from src.steps.data_collection.klines_data import get_parquet_utils, safe_read_parquet")
    print("")
    print("utils = get_parquet_utils()")
    print("df = utils.safe_read_parquet('historical_data/binance/ethusdt/raw/ethusdt_1m_2024_09.parquet')")
    print("```")
    print()

    print("NEW WAY (Enhanced Features):")
    print("```python")
    print("from src.steps.data_collection.klines_data import get_klines_manager")
    print("")
    print("manager = get_klines_manager()")
    print("df = manager.read_data('ETHUSDT', '1m')  # Much simpler!")
    print("info = manager.get_data_info('ETHUSDT', '1m')")
    print("stats = manager.get_data_statistics('ETHUSDT', '1m')")
    print("```")
    print()

if __name__ == "__main__":
    print("🚀 Klines Parquet Migration Demo")
    print("=" * 60)
    print()

    demo_old_approach()
    demo_new_approach()
    demo_migration_comparison()

    print("🎉 Migration demo completed!")
    print()
    print("📖 For more details, see:")
    print("   - MIGRATION_GUIDE.md")
    print("   - README.md")
    print("   - test_klines_structure.py")
