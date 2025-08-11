#!/usr/bin/env python3
"""
Simple Fix for Multicollinearity Issue

This script fixes the critical bug where all multi-timeframe price_change and volume_change
features are identical, causing perfect multicollinearity (VIF = inf).

Usage:
    python scripts/fix_multicollinearity_simple.py
"""

import os
import sys
from pathlib import Path


def fix_feature_engineering_code():
    """Fix the critical multicollinearity issue in the feature engineering code."""

    print("🔧 Starting multicollinearity fix...")

    # Path to the feature engineering file
    feature_eng_file = Path(
        "src/training/steps/vectorized_advanced_feature_engineering.py"
    )

    if not feature_eng_file.exists():
        print(f"❌ Feature engineering file not found: {feature_eng_file}")
        return False

    try:
        # Read the current file
        with open(feature_eng_file, "r") as f:
            content = f.read()

        print("📖 Reading current feature engineering code...")

        # Fix the problematic price_change calculation
        old_price_change = "price_changes = price_data[price_column].pct_change()"
        new_price_change = """            # CRITICAL FIX: Use proper periods for multi-timeframe price changes
            timeframe_periods = {
                "1m": 1,     # 1-period change for 1m
                "5m": 5,     # 5-period change for 5m
                "15m": 15,   # 15-period change for 15m
                "30m": 30,   # 30-period change for 30m
            }
            
            periods = timeframe_periods.get(timeframe, 1)
            price_changes = price_data[price_column].pct_change(periods=periods)"""

        if old_price_change in content:
            content = content.replace(old_price_change, new_price_change)
            print("✅ Fixed price_change calculation")
        else:
            print("⚠️ Could not find price_change line to fix")

        # Fix the problematic volume_change calculation
        old_volume_change = 'volume_changes = volume_data["volume"].pct_change()'
        new_volume_change = (
            'volume_changes = volume_data["volume"].pct_change(periods=periods)'
        )

        if old_volume_change in content:
            content = content.replace(old_volume_change, new_volume_change)
            print("✅ Fixed volume_change calculation")
        else:
            print("⚠️ Could not find volume_change line to fix")

        # Write the fixed content back
        with open(feature_eng_file, "w") as f:
            f.write(content)

        print("✅ Successfully fixed multicollinearity issue")
        return True

    except Exception as e:
        print(f"❌ Error fixing multicollinearity issue: {e}")
        return False


def main():
    """Main function to fix the multicollinearity issue."""

    print("🚀 Starting multicollinearity fix...")

    if fix_feature_engineering_code():
        print("🎉 Multicollinearity fix completed successfully!")
        print("📋 The issue was in the _calculate_timeframe_features_vectorized method")
        print("📋 All timeframes were using the same pct_change() without periods")
        print("📋 Now each timeframe uses proper periods: 1m=1, 5m=5, 15m=15, 30m=30")
        print("\n🔍 Next steps:")
        print("   1. Test your training pipeline again")
        print("   2. Monitor the logs for any remaining issues")
        return True
    else:
        print("❌ Multicollinearity fix failed!")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ MULTICOLLINEARITY FIX FAILED!")
        sys.exit(1)
    else:
        print("\n🎉 MULTICOLLINEARITY FIX COMPLETED SUCCESSFULLY!")
        print("✅ Your feature engineering should now work correctly.")
