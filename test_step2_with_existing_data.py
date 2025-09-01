#!/usr/bin/env python3
"""
Test script for the new step2 command functionality.

This script demonstrates how to use ares_launcher to start the enhanced_training_pipeline
from step2 with existing data (collected and processed in step1 and step1_5),
without triggering new downloads.
"""

import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Use existing validator orchestrator
from src.utils.validator_orchestrator import ValidatorOrchestrator
print("🔧 Using existing validator orchestrator")


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


async def test_data_validation():
    """Test the data validation functionality using existing validator orchestrator."""
    print("🧪 Testing Data Quality Validation")
    print("=" * 60)

    symbol = "ETHUSDT"
    exchange = "BINANCE"

    # Test with empty data_cache (should show missing data)
    print(f"\n📊 Testing validation for {symbol} on {exchange}")
    print("   (with empty data_cache directory)")

    # Use existing validator orchestrator
    validator_orchestrator = ValidatorOrchestrator()

    # Prepare training input for validation
    training_input = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": "1m",
        "data_dir": "data_cache"
    }

    # Empty pipeline state since we're checking existing data
    pipeline_state = {}

    # Import CONFIG
    from src.config import CONFIG

    # Validate step1 and step1_5 using existing validators
    print("🔍 Validating step1_data_collection using existing validator")
    step1_result = await validator_orchestrator.run_step_validator(
        "step1_data_collection", training_input, pipeline_state, CONFIG
    )

    print("🔍 Validating step1_5_data_converter using existing validator")
    step1_5_result = await validator_orchestrator.run_step_validator(
        "step1_5_data_converter", training_input, pipeline_state, CONFIG
    )

    print("\n📋 Validation Results:")
    print(f"   Step1 Passed: {step1_result.get('validation_passed', False)}")
    print(f"   Step1_5 Passed: {step1_5_result.get('validation_passed', False)}")
    print(f"   Step1 Warnings: {len(step1_result.get('warnings', []))}")
    print(f"   Step1_5 Warnings: {len(step1_5_result.get('warnings', []))}")

    if step1_result.get('warnings'):
        print("\n⚠️  Step1 Warnings:")
        for warning in step1_result['warnings']:
            print(f"   • {warning}")

    if step1_5_result.get('warnings'):
        print("\n⚠️  Step1_5 Warnings:")
        for warning in step1_5_result['warnings']:
            print(f"   • {warning}")

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


async def test_with_mock_data():
    """Test validation with mock data files."""
    print("\n🧪 Testing with mock data files...")

    symbol = "ETHUSDT"
    exchange = "BINANCE"

    # Create mock data
    create_mock_data_files()

    # Use existing validator orchestrator
    validator_orchestrator = ValidatorOrchestrator()

    # Prepare training input for validation
    training_input = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": "1m",
        "data_dir": "data_cache"
    }

    # Empty pipeline state since we're checking existing data
    pipeline_state = {}

    # Import CONFIG
    from src.config import CONFIG

    # Test validation
    print("🔍 Validating step1_data_collection using existing validator")
    step1_result = await validator_orchestrator.run_step_validator(
        "step1_data_collection", training_input, pipeline_state, CONFIG
    )

    print("🔍 Validating step1_5_data_converter using existing validator")
    step1_5_result = await validator_orchestrator.run_step_validator(
        "step1_5_data_converter", training_input, pipeline_state, CONFIG
    )

    can_start = step1_result.get('validation_passed', False) and step1_5_result.get('validation_passed', False)

    print(f"\n📊 Validation Results with Mock Data:")
    print(f"   Can start from step2: {can_start}")
    print(f"   Step1 Passed: {step1_result.get('validation_passed', False)}")
    print(f"   Step1_5 Passed: {step1_5_result.get('validation_passed', False)}")

    print(f"\n📊 Full Validation Report:")
    print_step2_validation_report(step1_result, step1_5_result, symbol, exchange)


async def main():
    """Main test function."""
    print("🧪 STEP2 WITH EXISTING DATA TEST")
    print("=" * 80)
    print("This script tests the new step2 command functionality that allows")
    print("starting the enhanced_training_pipeline from step2 with existing data.")
    print("=" * 80)

    # Test 1: Data validation with empty cache
    await test_data_validation()

    # Test 2: Step2 command explanation
    test_step2_command()

    # Test 3: Data validation with mock data
    await test_with_mock_data()

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
    import asyncio
    asyncio.run(main())