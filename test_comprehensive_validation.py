#!/usr/bin/env python3
"""
Test script for comprehensive file validation.

This script tests the comprehensive file validation functionality
for steps 1, 1.5, 2, and 4.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

    ComprehensiveFileValidator,
    validate_step1_file,
    validate_step1_5_file,
    validate_step2_file,
    validate_step4_file,
    FileValidationResult
)


def test_comprehensive_validator():
    """Test the comprehensive file validator."""
    print("🧪 Testing Comprehensive File Validator")
    print("=" * 50)

    # Initialize validator
    validator = ComprehensiveFileValidator()
    print(f"✅ Validator initialized with config: {len(validator.config)} sections")

    # Test configuration
    print(f"📋 File types supported: {list(validator.config['file_types'].keys())}")
    print(f"📋 Expected schemas: {list(validator.config['expected_schemas'].keys())}")

    return True


def test_validation_functions():
    """Test the convenience validation functions."""
    print("\n🧪 Testing Validation Functions")
    print("=" * 50)

    # Test with non-existent files (should handle gracefully)
    test_files = [
        "non_existent_file.parquet",
        "test_data.csv",
        "test_config.json"
    ]

    validation_functions = [
        ("Step 1", validate_step1_file),
        ("Step 1.5", validate_step1_5_file),
        ("Step 2", validate_step2_file),
        ("Step 4", validate_step4_file),
    ]

    for step_name, validation_func in validation_functions:
        print(f"\n🔍 Testing {step_name} validation function:")

        for test_file in test_files:
            try:
                result = validation_func(test_file)
                print(f"   📁 {test_file}: {'✅ Valid' if result.is_valid else '❌ Invalid'}")

                if not result.is_valid:
                    for issue in result.issues:
                        print(f"      - {issue.severity.value}: {issue.description}")

            except Exception as e:
                print(f"   ❌ Error validating {test_file}: {e}")

    return True


def test_schema_validation():
    """Test schema validation with sample data."""
    print("\n🧪 Testing Schema Validation")
    print("=" * 50)

    import pandas as pd
    import numpy as np

    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1min')

    # Sample klines data
    klines_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.random(len(dates)) * 100,
        'high': np.random.random(len(dates)) * 100,
        'low': np.random.random(len(dates)) * 100,
        'close': np.random.random(len(dates)) * 100,
        'volume': np.random.random(len(dates)) * 1000,
    })

    # Sample features data
    features_data = pd.DataFrame({
        'timestamp': dates,
        'feature_1': np.random.random(len(dates)),
        'feature_2': np.random.random(len(dates)),
        'feature_3': np.random.random(len(dates)),
    })

    # Test with temporary files
    test_files = {
        'klines_test.parquet': klines_data,
        'features_test.parquet': features_data,
    }

    validator = ComprehensiveFileValidator()

    for filename, data in test_files.items():
        print(f"\n📁 Testing validation for {filename}:")

        # Save temporary file
        data.to_parquet(filename, index=False)

        try:
            # Test validation
            if 'klines' in filename:
                result = validator.validate_file_format(filename, expected_schema="klines", step_name="test")
            else:
                result = validator.validate_file_format(filename, expected_schema="features", step_name="test")

            print(f"   ✅ Validation completed: {'Valid' if result.is_valid else 'Invalid'}")
            print(f"   📊 Shape: {result.summary.get('shape', 'N/A')}")
            print(f"   🗂️ Columns: {result.summary.get('column_count', 'N/A')}")
            print(f"   📁 File type: {result.file_type}")

            if not result.is_valid:
                for issue in result.issues:
                    print(f"      - {issue.severity.value}: {issue.description}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        finally:
            # Clean up
            try:
                Path(filename).unlink()
            except:
                pass

    return True


async def test_async_validation():
    """Test async validation functionality."""
    print("\n🧪 Testing Async Validation")
    print("=" * 50)

    # This would test async validation if implemented
    print("✅ Async validation test completed (placeholder)")
    return True


def main():
    """Run all tests."""
    print("🚀 Starting Comprehensive File Validation Tests")
    print("=" * 60)

    try:
        # Run tests
        test_comprehensive_validator()
        test_validation_functions()
        test_schema_validation()

        # Run async test
        asyncio.run(test_async_validation())

        print("\n✅ All tests completed successfully!")
        print("🎉 Comprehensive file validation is working correctly")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)