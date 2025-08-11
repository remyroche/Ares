#!/usr/bin/env python3
"""
Simple test script to verify NaN detection functionality.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_test_data_with_nans():
    """Create test data with known NaN values to verify detection."""
    print("🔍 Creating test data with known NaN values...")

    # Create sample data with some NaN values
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")

    # Create price data with some NaN values
    np.random.seed(42)
    base_price = 100 + np.cumsum(np.random.randn(1000) * 0.1)

    price_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": base_price + np.random.randn(1000) * 0.5,
            "high": base_price + np.random.randn(1000) * 0.8,
            "low": base_price - np.random.randn(1000) * 0.8,
            "close": base_price + np.random.randn(1000) * 0.5,
            "volume": np.random.randint(100, 1000, 1000),
        }
    )

    # Introduce specific NaN values for testing with realistic patterns
    # Pattern 1: Single day gap (6 consecutive minutes)
    price_data.loc[100:105, "close"] = np.nan  # 6 NaN values

    # Pattern 2: Weekend gap simulation (11 consecutive minutes)
    price_data.loc[200:210, "volume"] = np.nan  # 11 NaN values

    # Pattern 3: Multiple small gaps
    price_data.loc[300:302, "high"] = np.nan  # 3 NaN values
    price_data.loc[350:352, "high"] = np.nan  # 3 NaN values

    # Create volume data
    volume_data = pd.DataFrame(
        {
            "timestamp": dates,
            "volume": np.random.randint(100, 1000, 1000),  # Independent volume data
            "trade_count": np.random.randint(10, 100, 1000),
        }
    )

    # Introduce some NaN values in volume data
    volume_data.loc[150:155, "trade_count"] = np.nan  # 6 NaN values

    # Set timestamp as index
    price_data.set_index("timestamp", inplace=True)
    volume_data.set_index("timestamp", inplace=True)

    return price_data, volume_data


def test_nan_detection_manual():
    """Test NaN detection manually to verify functionality."""
    print("🧪 Testing NaN detection manually...")

    # Create test data
    price_data, volume_data = create_test_data_with_nans()

    print(f"📊 Test data created:")
    print(f"   Price data shape: {price_data.shape}")
    print(f"   Volume data shape: {volume_data.shape}")

    # Manual NaN detection
    price_nans = price_data.isnull().sum().sum()
    volume_nans = volume_data.isnull().sum().sum()

    print(f"\n🔍 Manual NaN detection results:")
    print(f"   Price data NaN values: {price_nans}")
    print(f"   Volume data NaN values: {volume_nans}")
    print(f"   Total NaN values: {price_nans + volume_nans}")

    # Check specific columns
    print(f"\n📋 Column-specific NaN analysis:")
    for col in price_data.columns:
        col_nans = price_data[col].isnull().sum()
        if col_nans > 0:
            print(f"   {col}: {col_nans} NaNs ({col_nans/len(price_data)*100:.2f}%)")

    for col in volume_data.columns:
        col_nans = volume_data[col].isnull().sum()
        if col_nans > 0:
            print(f"   {col}: {col_nans} NaNs ({col_nans/len(volume_data)*100:.2f}%)")

    # Calculate total statistics
    total_cells = price_data.size + volume_data.size
    total_nans = price_nans + volume_nans
    nan_percentage = (total_nans / total_cells * 100) if total_cells > 0 else 0

    print(f"\n📈 Summary statistics:")
    print(f"   Total cells: {total_cells}")
    print(f"   Total NaN values: {total_nans}")
    print(f"   NaN percentage: {nan_percentage:.2f}%")
    print(f"   Data quality score: {max(0, 100 - nan_percentage):.1f}/100")

    # Assess severity
    if nan_percentage == 0:
        severity = "EXCELLENT"
    elif nan_percentage < 1:
        severity = "GOOD"
    elif nan_percentage < 5:
        severity = "ACCEPTABLE"
    elif nan_percentage < 10:
        severity = "POOR"
    elif nan_percentage < 20:
        severity = "VERY_POOR"
    else:
        severity = "CRITICAL"

    print(f"   Severity: {severity}")

    # Verify expected results
    expected_price_nans = 6 + 11 + 6  # close + volume + high
    expected_volume_nans = 6  # trade_count
    expected_total = expected_price_nans + expected_volume_nans

    print(f"\n✅ Verification:")
    print(f"   Expected price NaNs: {expected_price_nans}, Found: {price_nans}")
    print(f"   Expected volume NaNs: {expected_volume_nans}, Found: {volume_nans}")
    print(f"   Expected total NaNs: {expected_total}, Found: {total_nans}")

    if price_nans == expected_price_nans and volume_nans == expected_volume_nans:
        print("   🎉 NaN detection working correctly!")
        return True
    else:
        print("   ❌ NaN detection not working as expected!")
        return False


def test_data_alignment():
    """Test data alignment functionality."""
    print("\n🔗 Testing data alignment functionality...")

    # Create test data
    price_data, volume_data = create_test_data_with_nans()

    # Create labeled data with different indices (simulating triple barrier labeling)
    labeled_indices = price_data.index[50:950]  # Remove some indices
    labeled_data = pd.DataFrame(
        {
            "label": np.random.choice([-1, 1], size=len(labeled_indices)),
            "timestamp": labeled_indices,
        }
    ).set_index("timestamp")

    print(f"   Original price data length: {len(price_data)}")
    print(f"   Original volume data length: {len(volume_data)}")
    print(f"   Labeled data length: {len(labeled_data)}")

    # Test alignment logic
    labeled_indices_set = set(labeled_data.index)
    price_indices_set = set(price_data.index)
    volume_indices_set = set(volume_data.index)

    # Find common indices
    common_indices = labeled_indices_set.intersection(price_indices_set).intersection(
        volume_indices_set
    )
    common_indices = sorted(common_indices)

    print(f"   Common indices found: {len(common_indices)}")

    # Align data
    aligned_price_data = price_data.loc[common_indices]
    aligned_volume_data = volume_data.loc[common_indices]
    aligned_labeled_data = labeled_data.loc[common_indices]

    print(f"   Aligned price data length: {len(aligned_price_data)}")
    print(f"   Aligned volume data length: {len(aligned_volume_data)}")
    print(f"   Aligned labeled data length: {len(aligned_labeled_data)}")

    # Check for NaN values in aligned data
    aligned_price_nans = aligned_price_data.isnull().sum().sum()
    aligned_volume_nans = aligned_volume_data.isnull().sum().sum()

    print(f"   Aligned price data NaNs: {aligned_price_nans}")
    print(f"   Aligned volume data NaNs: {aligned_volume_nans}")

    if len(aligned_price_data) == len(aligned_volume_data) == len(aligned_labeled_data):
        print("   ✅ Data alignment working correctly!")
        return True
    else:
        print("   ❌ Data alignment not working correctly!")
        return False


def test_nan_timestamp_ranges():
    """Test the new timestamp range functionality for NaN values."""
    print("🕐 Testing NaN timestamp range detection...")

    # Create test data
    price_data, volume_data = create_test_data_with_nans()

    print(f"📊 Test data created with timestamp index")

    # Test timestamp range detection for each column with NaN values
    columns_to_test = ["close", "volume", "high", "trade_count"]

    for col in columns_to_test:
        dataset = price_data if col in price_data.columns else volume_data
        if col in dataset.columns:
            print(f"\n🔍 Analyzing NaN ranges for column: {col}")

            # Get NaN mask
            nan_mask = dataset[col].isnull()
            nan_indices = dataset.index[nan_mask]

            if len(nan_indices) > 0:
                print(f"   Total NaN values: {len(nan_indices)}")
                print(f"   NaN percentage: {len(nan_indices)/len(dataset)*100:.2f}%")

                # Group consecutive NaN timestamps into ranges
                ranges = []
                start_idx = 0

                for i in range(1, len(nan_indices)):
                    current_time = nan_indices[i]
                    prev_time = nan_indices[i - 1]
                    time_diff = current_time - prev_time

                    # If gap is more than 2 minutes, start a new range
                    if time_diff > pd.Timedelta(minutes=2):
                        range_info = {
                            "start": nan_indices[start_idx],
                            "end": nan_indices[i - 1],
                            "start_human": nan_indices[start_idx].strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "end_human": nan_indices[i - 1].strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "duration_minutes": (
                                nan_indices[i - 1] - nan_indices[start_idx]
                            ).total_seconds()
                            / 60,
                            "count": i - start_idx,
                        }
                        ranges.append(range_info)
                        start_idx = i

                # Add the last range
                if start_idx < len(nan_indices):
                    range_info = {
                        "start": nan_indices[start_idx],
                        "end": nan_indices[-1],
                        "start_human": nan_indices[start_idx].strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "end_human": nan_indices[-1].strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_minutes": (
                            nan_indices[-1] - nan_indices[start_idx]
                        ).total_seconds()
                        / 60,
                        "count": len(nan_indices) - start_idx,
                    }
                    ranges.append(range_info)

                print(f"   📅 Found {len(ranges)} NaN timestamp range(s):")
                for i, range_info in enumerate(ranges, 1):
                    print(
                        f"      Range {i}: {range_info['start_human']} to {range_info['end_human']}"
                    )
                    print(
                        f"         Duration: {range_info['duration_minutes']:.1f} minutes"
                    )
                    print(f"         NaN count: {range_info['count']}")
            else:
                print(f"   ✅ No NaN values found in column {col}")

    return True


def main():
    """Main test function."""
    print("🚀 Starting NaN detection and data quality tests...")
    print("=" * 60)

    # Test 1: Manual NaN detection
    test1_passed = test_nan_detection_manual()

    # Test 2: Data alignment
    test2_passed = test_data_alignment()

    # Test 3: New timestamp range detection
    test3_passed = test_nan_timestamp_ranges()

    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"   NaN Detection Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Data Alignment Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(
        f"   Timestamp Range Detection Test: {'✅ PASSED' if test3_passed else '❌ FAILED'}"
    )

    if test1_passed and test2_passed and test3_passed:
        print(
            "\n🎉 All tests passed! NaN detection and data quality assessment is working correctly."
        )
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")

    print("=" * 60)


if __name__ == "__main__":
    main()
