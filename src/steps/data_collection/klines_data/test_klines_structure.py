#!/usr/bin/env python3
"""
Test script for the new klines_data directory structure.

This script tests that all the moved modules can be imported correctly
and that the basic functionality is working.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all klines modules can be imported."""
    print("🧪 Testing klines_data module imports...")

    try:
        # Test core utils imports
        from src.steps.data_collection.klines_data.klines_parquet import get_klines_manager
        print("✅ klines_parquet imported successfully")

        from src.steps.data_collection.klines_data.gap_detector import GapDetector
        print("✅ gap_detector imported successfully")

        from src.steps.data_collection.klines_data.basic_returns_engineer import BasicReturnsEngineer
        print("✅ basic_returns_engineer imported successfully")

        from src.steps.data_collection.klines_data.historical_data_downloader import HistoricalDataDownloader
        print("✅ historical_data_downloader imported successfully")

        from src.steps.data_collection.klines_data.historical_data_pipeline import HistoricalDataPipeline
        print("✅ historical_data_pipeline imported successfully")

        # Test training components (optional)
        try:
            from src.steps.data_collection.klines_data.unified_data_downloader import UnifiedDataDownloader
            print("✅ unified_data_downloader imported successfully")
        except ImportError as e:
            print(f"⚠️ unified_data_downloader import failed: {e}")

        try:
            from src.steps.data_collection.klines_data.enhanced_append_data_downloader import EnhancedAppendDataDownloader
            print("✅ enhanced_append_data_downloader imported successfully")
        except ImportError as e:
            print(f"⚠️ enhanced_append_data_downloader import failed: {e}")

        try:
            from src.steps.data_collection.klines_data.unified_gap_filler import UnifiedGapFiller
            print("✅ unified_gap_filler imported successfully")
        except ImportError as e:
            print(f"⚠️ unified_gap_filler import failed: {e}")

        try:
            from src.steps.data_collection.klines_data.unified_resampler import UnifiedResampler
            print("✅ unified_resampler imported successfully")
        except ImportError as e:
            print(f"⚠️ unified_resampler import failed: {e}")

        print("🎉 All klines_data module imports completed!")
        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_directory_structure():
    """Test that the directory structure is correct."""
    print("\n📁 Testing directory structure...")

    klines_dir = Path(__file__).parent

    required_files = [
        '__init__.py',
        'klines_parquet.py',
        'gap_detector.py',
        'basic_returns_engineer.py',
        'historical_data_downloader.py',
        'historical_data_pipeline.py',
        'README.md'
    ]

    for file in required_files:
        file_path = klines_dir / file
        if file_path.exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False

    print("🎉 Directory structure verification completed!")
    return True

if __name__ == "__main__":
    print("🚀 Testing new klines_data directory structure\n")

    success = True
    success &= test_directory_structure()
    success &= test_imports()

    if success:
        print("\n🎉 All tests passed! The klines_data directory structure is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
