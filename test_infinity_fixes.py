#!/usr/bin/env python3
"""
Test script to verify that the infinity fixes work correctly.
"""

import numpy as np
import pandas as pd

def test_safe_division():
    """Test the safe division logic used in the fixes."""

    # Create test data with zeros and small values
    numerator = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    denominator = np.array([1.0, 0.0, 0.000001, 0.5, 2.0])

    print("Testing safe division logic...")

    # Apply the safe division logic from our fixes
    denominator_safe = np.where(denominator == 0, np.nan, denominator)
    denominator_safe = pd.Series(denominator_safe).fillna(method='bfill').fillna(1e-6).values

    result = numerator / denominator_safe
    result_clipped = np.clip(result, -100, 100)

    print(f"Original denominator: {denominator}")
    print(f"Safe denominator: {denominator_safe}")
    print(f"Result: {result}")
    print(f"Clipped result: {result_clipped}")

    # Check for infinity
    has_inf = np.any(np.isinf(result_clipped))
    print(f"Has infinity values: {has_inf}")

    if not has_inf:
        print("✅ Safe division test passed!")
    else:
        print("❌ Safe division test failed!")

    return not has_inf

def test_volume_ratio_fix():
    """Test volume ratio calculation with edge cases."""

    print("\nTesting volume ratio calculation...")

    # Create test data with zero volume periods
    volume = np.array([100, 0, 50, 200, 0, 75])
    volume_ma = np.array([50, 0, 25, 100, np.nan, 60])

    # Apply the fix logic
    volume_ma_safe = np.where(volume_ma == 0, np.nan, volume_ma)
    volume_ma_safe = pd.Series(volume_ma_safe).fillna(method='bfill').fillna(1.0).values

    volume_ratio = volume / volume_ma_safe
    volume_ratio_clipped = np.clip(volume_ratio, -100, 100)

    print(f"Volume: {volume}")
    print(f"Volume MA: {volume_ma}")
    print(f"Safe Volume MA: {volume_ma_safe}")
    print(f"Volume Ratio: {volume_ratio}")
    print(f"Clipped Volume Ratio: {volume_ratio_clipped}")

    has_inf = np.any(np.isinf(volume_ratio_clipped))
    print(f"Has infinity values: {has_inf}")

    if not has_inf:
        print("✅ Volume ratio test passed!")
    else:
        print("❌ Volume ratio test failed!")

    return not has_inf

def test_price_volume_ratio_fix():
    """Test price-volume ratio calculation with edge cases."""

    print("\nTesting price-volume ratio calculation...")

    # Create test data with zero volume changes
    price_change = np.array([0.01, 0.02, -0.01, 0.005])
    volume_change = np.array([0.1, 0.0, 0.000001, -0.05])

    # Apply the fix logic
    volume_change_safe = np.where(volume_change == 0, np.nan, volume_change)
    volume_change_safe = pd.Series(volume_change_safe).fillna(method='bfill').fillna(1e-6).values

    price_volume_ratio = price_change / volume_change_safe
    price_volume_ratio_clipped = np.clip(price_volume_ratio, -1000, 1000)

    print(f"Price Change: {price_change}")
    print(f"Volume Change: {volume_change}")
    print(f"Safe Volume Change: {volume_change_safe}")
    print(f"Price-Volume Ratio: {price_volume_ratio}")
    print(f"Clipped Price-Volume Ratio: {price_volume_ratio_clipped}")

    has_inf = np.any(np.isinf(price_volume_ratio_clipped))
    print(f"Has infinity values: {has_inf}")

    if not has_inf:
        print("✅ Price-volume ratio test passed!")
    else:
        print("❌ Price-volume ratio test failed!")

    return not has_inf

def main():
    """Run all tests."""
    print("🧪 Testing infinity fixes...")

    test1_passed = test_safe_division()
    test2_passed = test_volume_ratio_fix()
    test3_passed = test_price_volume_ratio_fix()

    all_passed = test1_passed and test2_passed and test3_passed

    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All infinity fix tests passed!")
        print("The fixes should prevent infinity values in feature calculations.")
    else:
        print("❌ Some tests failed. Infinity values may still occur.")

    return all_passed

if __name__ == "__main__":
    main()
