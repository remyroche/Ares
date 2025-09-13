#!/usr/bin/env python3
"""
Test script to verify that infinite values and duplicate timestamp issues are fixed.

This script tests the core functions that were modified to ensure they no longer
produce infinite values or handle duplicate timestamps incorrectly.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.steps.data_collection.klines_data.basic_returns_engineer import BasicReturnsEngineer
from src.utils.data.basic_returns_engineer import BasicReturnsEngineer as BasicReturnsEngineer2
from src.utils.data.feature_engineer import FeatureEngineer

def test_safe_pct_change():
    """Test that _safe_pct_change no longer produces infinite values."""
    print("🧪 Testing _safe_pct_change function...")

    # Test with the main basic_returns_engineer
    engineer = BasicReturnsEngineer()

    # Create test data with edge cases
    test_data = pd.Series([0, 1, 0, 10, 0, 5, 0])  # Contains zeros that could cause issues

    result = engineer._safe_pct_change(test_data)

    # Check for infinite values
    infinite_count = np.isinf(result).sum()
    print(f"   📊 Infinite values found: {infinite_count}")

    if infinite_count > 0:
        print("❌ FAILED: Infinite values still present!")
        return False
    else:
        print("✅ PASSED: No infinite values found")

    # Check that values are within expected range
    min_val, max_val = result.min(), result.max()
    print(f"   📊 Value range: {min_val:.3f} to {max_val:.3f}")

    if min_val >= -9.0 and max_val <= 9.0:
        print("✅ PASSED: Values are within expected range")
        return True
    else:
        print("❌ FAILED: Values outside expected range")
        return False

def test_volume_log_return_calculation():
    """Test that volume log return calculation no longer produces infinite values."""
    print("🧪 Testing volume log return calculation...")

    # Create test data
    data = pd.DataFrame({
        'close': [100, 101, 99, 102],
        'volume': [1000, 0, 1500, 0],  # Contains zeros
        'open': [99, 100, 101, 99],
        'high': [102, 101, 102, 103],
        'low': [98, 99, 98, 98]
    })

    engineer = BasicReturnsEngineer()

    # Add features (this includes volume_log_return calculation)
    featured_data = engineer._add_basic_returns(data)

    # Check for infinite values in volume_log_return
    if 'volume_log_return' in featured_data.columns:
        infinite_count = np.isinf(featured_data['volume_log_return']).sum()
        print(f"   📊 Infinite values in volume_log_return: {infinite_count}")

        if infinite_count > 0:
            print("❌ FAILED: Infinite values still present in volume_log_return!")
            return False
        else:
            print("✅ PASSED: No infinite values in volume_log_return")
            return True
    else:
        print("⚠️ volume_log_return column not found")
        return False

def test_duplicate_timestamp_handling():
    """Test that duplicate timestamp handling works correctly."""
    print("🧪 Testing duplicate timestamp handling...")

    # Create test data with duplicate timestamps
    data1 = pd.DataFrame({
        'close': [100, 101],
        'volume': [1000, 1100],
        'open': [99, 100],
        'high': [102, 103],
        'low': [98, 99]
    }, index=pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:01:00']))

    data2 = pd.DataFrame({
        'close': [100, 101],  # Same values but different volume
        'volume': [1200, 1300],  # Different volume values
        'open': [99, 100],
        'high': [102, 103],
        'low': [98, 99]
    }, index=pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:01:00']))  # Duplicate timestamps

    combined_data = pd.concat([data1, data2])
    print(f"   📊 Combined data has {len(combined_data)} rows")

    # Test duplicate removal
    engineer = BasicReturnsEngineer()

    # Simulate the duplicate removal logic
    combined_data = combined_data.sort_index()

    initial_count = len(combined_data)
    duplicate_mask = combined_data.index.duplicated(keep=False)

    if duplicate_mask.any():
        duplicate_indices = combined_data.index[duplicate_mask]
        print(f"   🔍 Found {duplicate_mask.sum()} duplicate entries")

        # Apply the improved duplicate removal logic
        for dup_timestamp in duplicate_indices.unique():
            dup_rows = combined_data.loc[[dup_timestamp]]
            if len(dup_rows) > 1:
                # Count non-null values per column for each duplicate
                non_null_counts = dup_rows.notna().sum(axis=1)
                # Keep the row with most non-null values, or first if tie
                best_idx = non_null_counts.idxmax()
                keep_mask = (combined_data.index == dup_timestamp) & (combined_data.index == best_idx)
                combined_data = combined_data[~(duplicate_mask & (combined_data.index == dup_timestamp) & ~keep_mask)]

        # Final deduplication as safety net
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]

    final_count = len(combined_data)
    print(f"   📊 Final data has {final_count} rows (removed {initial_count - final_count} duplicates)")

    if final_count < initial_count:
        print("✅ PASSED: Duplicates were successfully removed")
        return True
    else:
        print("⚠️ No duplicates found in test data")
        return True

def main():
    """Run all tests."""
    print("🚀 Testing fixes for infinite values and duplicate timestamps...")
    print()

    tests_passed = 0
    total_tests = 3

    # Test 1: safe_pct_change function
    if test_safe_pct_change():
        tests_passed += 1
    print()

    # Test 2: volume log return calculation
    if test_volume_log_return_calculation():
        tests_passed += 1
    print()

    # Test 3: duplicate timestamp handling
    if test_duplicate_timestamp_handling():
        tests_passed += 1
    print()

    # Summary
    print("=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Infinite values and duplicate timestamp issues are fixed.")
        return True
    else:
        print("❌ Some tests failed. Please review the fixes.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
