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

try:
    # Try comprehensive validator first
    from src.utils.comprehensive_data_quality_validator import (
        validate_step1_quality as validate_step1_files,
        validate_step1_5_quality as validate_step1_5_files
    )
    print("🔬 Using comprehensive data quality validator")
except ImportError:
    # Fallback to simple file existence validator
    from src.utils.simple_data_validator import (
        validate_step1_files,
        validate_step1_5_files
    )
    print("📁 Using simple file existence validator")


def print_step2_validation_report(step1_result: dict, step1_5_result: dict, symbol: str, exchange: str):
    """Print a formatted validation report for step2 readiness."""
    print("\n" + "="*80)
    print(f"📊 DATA VALIDATION REPORT FOR STEP2")
    print(f"🎯 Symbol: {symbol}")
    print(f"🏢 Exchange: {exchange}")
    print("="*80)
    
    # Step1 status
    step1_status = "✅ PASSED" if step1_result.get("validation_passed", False) else "❌ FAILED"
    step1_issues = len(step1_result.get("issues", []))
    print(f"📁 Step1 Data Collection: {step1_status}")
    if step1_issues > 0:
        print(f"   ⚠️  Found {step1_issues} issues")
    
    # Show step1 file checks
    file_checks = step1_result.get("file_checks", {})
    if file_checks:
        print(f"   📄 File Status:")
        for filename, check in file_checks.items():
            status = "✅" if check.get("exists", False) else "❌"
            print(f"     {status} {filename}")
    
    # Step1_5 status
    step1_5_status = "✅ PASSED" if step1_5_result.get("validation_passed", False) else "❌ FAILED"
    step1_5_issues = len(step1_5_result.get("issues", []))
    print(f"🔄 Step1_5 Data Converter: {step1_5_status}")
    if step1_5_issues > 0:
        print(f"   ⚠️  Found {step1_5_issues} issues")
    
    # Show step1_5 file checks
    file_checks_1_5 = step1_5_result.get("file_checks", {})
    if file_checks_1_5:
        print(f"   📄 File Status:")
        for filename, check in file_checks_1_5.items():
            status = "✅" if check.get("exists", False) else "❌"
            print(f"     {status} {filename}")
    
    # Issues summary
    total_issues = step1_issues + step1_5_issues
    if total_issues > 0:
        print(f"\n⚠️  ISSUES SUMMARY ({total_issues} total):")
        for issue in step1_result.get("issues", []):
            print(f"   • Step1: {issue}")
        for issue in step1_5_result.get("issues", []):
            print(f"   • Step1_5: {issue}")
    
    # Overall assessment
    can_start = step1_result.get("validation_passed", False) and step1_5_result.get("validation_passed", False)
    if can_start:
        print(f"\n✅ READY TO START FROM STEP2")
        print(f"   Proceeding with existing data...")
    else:
        print(f"\n❌ NOT READY FOR STEP2")
        print(f"   Data validation failed - missing or invalid data")
    
    print("="*80 + "\n")


def test_data_validation():
    """Test the data validation functionality using existing validator."""
    print("🧪 Testing Data Quality Validation")
    print("=" * 60)
    
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    
    # Test with empty data_cache (should show missing data)
    print(f"\n📊 Testing validation for {symbol} on {exchange}")
    print("   (with empty data_cache directory)")
    
    # Use existing validators
    step1_result = validate_step1_files(symbol, exchange)
    step1_5_result = validate_step1_5_files(symbol, exchange)
    
    print("\n📋 Validation Results:")
    print(f"   Step1 Passed: {step1_result.get('validation_passed', False)}")
    print(f"   Step1_5 Passed: {step1_5_result.get('validation_passed', False)}")
    print(f"   Step1 Issues: {len(step1_result.get('issues', []))}")
    print(f"   Step1_5 Issues: {len(step1_5_result.get('issues', []))}")
    
    if step1_result.get('issues'):
        print("\n⚠️  Step1 Issues:")
        for issue in step1_result['issues']:
            print(f"   • {issue}")
    
    if step1_5_result.get('issues'):
        print("\n⚠️  Step1_5 Issues:")
        for issue in step1_5_result['issues']:
            print(f"   • {issue}")
    
    # Test the overall readiness
    print(f"\n🔍 Testing step2 readiness...")
    can_start = step1_result.get('validation_passed', False) and step1_5_result.get('validation_passed', False)
    print(f"   Can start from step2: {can_start}")
    
    print(f"\n📊 Printing validation report...")
    print_step2_validation_report(step1_result, step1_5_result, symbol, exchange)


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
    step1_result = validate_step1_files(symbol, exchange)
    step1_5_result = validate_step1_5_files(symbol, exchange)
    can_start = step1_result.get('validation_passed', False) and step1_5_result.get('validation_passed', False)
    
    print(f"\n📊 Validation Results with Mock Data:")
    print(f"   Can start from step2: {can_start}")
    print(f"   Step1 Passed: {step1_result.get('validation_passed', False)}")
    print(f"   Step1_5 Passed: {step1_5_result.get('validation_passed', False)}")
    
    print(f"\n📋 File Checks Found:")
    step1_files = step1_result.get('file_checks', {})
    step1_5_files = step1_5_result.get('file_checks', {})
    
    if step1_files:
        print(f"   Step1 files: {len(step1_files)}")
        for filename, check in step1_files.items():
            status = "✅" if check.get('exists', False) else "❌"
            print(f"     {status} {filename}")
    
    if step1_5_files:
        print(f"   Step1_5 files: {len(step1_5_files)}")
        for filename, check in step1_5_files.items():
            status = "✅" if check.get('exists', False) else "❌"
            print(f"     {status} {filename}")
    
    print(f"\n📊 Full Validation Report:")
    print_step2_validation_report(step1_result, step1_5_result, symbol, exchange)


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