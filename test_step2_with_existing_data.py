#!/usr/bin/env python3
"""
Test script for the new step2 command functionality.

This script demonstrates how to use ares_launcher to start the enhanced_training_pipeline 
from step2 with existing data (collected and processed in step1 and step1_5), 
without triggering new downloads.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.data_completeness_validator import (
    DataCompletenessValidator,
    validate_data_for_step2,
    print_data_validation_report
)


def test_data_validation():
    """Test the data completeness validation functionality."""
    print("🧪 Testing Data Completeness Validation")
    print("=" * 60)
    
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    
    # Test with empty data_cache (should show missing data)
    print(f"\n📊 Testing validation for {symbol} on {exchange}")
    print("   (with empty data_cache directory)")
    
    validator = DataCompletenessValidator()
    validation_result = validator.validate_step1_data_completeness(symbol, exchange)
    
    print("\n📋 Validation Results:")
    print(f"   Step1 Complete: {validation_result['step1_complete']}")
    print(f"   Step1_5 Complete: {validation_result['step1_5_complete']}")
    print(f"   Warnings: {len(validation_result['warnings'])}")
    print(f"   Gaps: {len(validation_result['gaps'])}")
    
    if validation_result['warnings']:
        print("\n⚠️  Warnings:")
        for warning in validation_result['warnings']:
            print(f"   • {warning}")
    
    if validation_result['gaps']:
        print("\n🕳️  Data Gaps:")
        for gap in validation_result['gaps']:
            print(f"   • {gap}")
    
    # Test the convenience functions
    print(f"\n🔍 Testing convenience functions...")
    can_start, result = validate_data_for_step2(symbol, exchange)
    print(f"   Can start from step2: {can_start}")
    
    print(f"\n📊 Printing validation report...")
    print_data_validation_report(symbol, exchange)


def test_step2_command():
    """Test the new step2 command functionality."""
    print("\n🚀 Testing Step2 Command Functionality")
    print("=" * 60)
    
    print("""
The new 'step2' command allows you to start the enhanced_training_pipeline 
from step2 with existing data without triggering new downloads.

Usage examples:
1. python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE
2. python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE --step step2_feature_engineering
3. python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE --force-rerun

Key features:
✅ Validates existing step1 and step1_5 data
✅ Provides warnings for incomplete data
✅ Detects data gaps
✅ Does NOT trigger new downloads
✅ Proceeds with existing data if validation passes
❌ Fails gracefully if required data is missing
    """)


def create_mock_data_files():
    """Create mock data files for testing."""
    print("\n🔧 Creating mock data files for testing...")
    
    # Create data_cache directory if it doesn't exist
    data_cache_dir = Path("data_cache")
    data_cache_dir.mkdir(exist_ok=True)
    
    # Create mock step1 files
    step1_files = [
        "klines_BINANCE_ETHUSDT_1m_consolidated.parquet",
        "klines_BINANCE_ETHUSDT_5m_consolidated.parquet",
        "aggtrades_BINANCE_ETHUSDT_consolidated.parquet"
    ]
    
    for filename in step1_files:
        file_path = data_cache_dir / filename
        if not file_path.exists():
            # Create empty file
            file_path.touch()
            print(f"   ✅ Created mock file: {filename}")
    
    # Create mock step1_5 files
    step1_5_files = [
        "processed_BINANCE_ETHUSDT_train.parquet",
        "processed_BINANCE_ETHUSDT_validation.parquet",
        "processed_BINANCE_ETHUSDT_test.parquet"
    ]
    
    for filename in step1_5_files:
        file_path = data_cache_dir / filename
        if not file_path.exists():
            # Create empty file
            file_path.touch()
            print(f"   ✅ Created mock file: {filename}")
    
    print("   📁 Mock data files created successfully")


def test_with_mock_data():
    """Test validation with mock data files."""
    print("\n🧪 Testing with mock data files...")
    
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    
    # Create mock data
    create_mock_data_files()
    
    # Test validation
    can_start, validation_result = validate_data_for_step2(symbol, exchange)
    
    print(f"\n📊 Validation Results with Mock Data:")
    print(f"   Can start from step2: {can_start}")
    print(f"   Step1 Complete: {validation_result['step1_complete']}")
    print(f"   Step1_5 Complete: {validation_result['step1_5_complete']}")
    
    print(f"\n📋 Data Files Found:")
    for step, files in validation_result['data_files'].items():
        print(f"   {step}: {len(files)} files")
        for filename in files.keys():
            print(f"     • {filename}")
    
    print(f"\n📊 Full Validation Report:")
    print_data_validation_report(symbol, exchange)


def main():
    """Main test function."""
    print("🧪 STEP2 WITH EXISTING DATA TEST")
    print("=" * 80)
    print("This script tests the new step2 command functionality that allows")
    print("starting the enhanced_training_pipeline from step2 with existing data.")
    print("=" * 80)
    
    # Test 1: Data validation with empty cache
    test_data_validation()
    
    # Test 2: Step2 command explanation
    test_step2_command()
    
    # Test 3: Data validation with mock data
    test_with_mock_data()
    
    print("\n" + "=" * 80)
    print("✅ STEP2 WITH EXISTING DATA TEST COMPLETED")
    print("=" * 80)
    print("\nTo use the new functionality:")
    print("1. Ensure you have step1 and step1_5 data in data_cache/")
    print("2. Run: python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE")
    print("3. The launcher will validate existing data and proceed with step2")
    print("\nThe system will:")
    print("✅ Check for existing step1 and step1_5 data")
    print("✅ Provide warnings for incomplete data")
    print("✅ Detect data gaps")
    print("✅ NOT trigger new downloads")
    print("✅ Proceed with existing data if validation passes")


if __name__ == "__main__":
    main()