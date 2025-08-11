#!/usr/bin/env python3
"""
Simple Validation of Multicollinearity Fix

This script validates that the multicollinearity issue has been fixed by directly
checking the modified code file.

Usage:
    python scripts/validate_fix_simple.py
"""

import sys
from pathlib import Path


def validate_fix():
    """Validate that the multicollinearity fix was applied correctly."""

    print("🔍 Validating multicollinearity fix...")

    # Path to the feature engineering file
    feature_eng_file = Path(
        "src/training/steps/vectorized_advanced_feature_engineering.py"
    )

    if not feature_eng_file.exists():
        print(f"❌ Feature engineering file not found: {feature_eng_file}")
        return False

    try:
        # Read the file
        with open(feature_eng_file, "r") as f:
            content = f.read()

        print("📖 Reading feature engineering code...")

        # Check for the fix
        fix_indicators = [
            "CRITICAL FIX: Use proper periods for multi-timeframe price changes",
            "timeframe_periods = {",
            '"1m": 1,     # 1-period change for 1m',
            '"5m": 5,     # 5-period change for 5m',
            '"15m": 15,   # 15-period change for 15m',
            '"30m": 30,   # 30-period change for 30m',
            "periods = timeframe_periods.get(timeframe, 1)",
            "price_changes = price_data[price_column].pct_change(periods=periods)",
            'volume_changes = volume_data["volume"].pct_change(periods=periods)',
        ]

        missing_indicators = []
        for indicator in fix_indicators:
            if indicator not in content:
                missing_indicators.append(indicator)

        if missing_indicators:
            print("❌ Fix validation failed! Missing indicators:")
            for indicator in missing_indicators:
                print(f"   - {indicator}")
            return False

        # Check that the old problematic code is gone
        problematic_code = [
            "price_changes = price_data[price_column].pct_change()",
            'volume_changes = volume_data["volume"].pct_change()',
        ]

        remaining_problems = []
        for problem in problematic_code:
            if problem in content:
                remaining_problems.append(problem)

        if remaining_problems:
            print("❌ Fix validation failed! Problematic code still present:")
            for problem in remaining_problems:
                print(f"   - {problem}")
            return False

        print("✅ All fix indicators found!")
        print("✅ Problematic code removed!")
        print("✅ Multicollinearity fix validation passed!")
        return True

    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False


def check_feature_selection_config():
    """Check that the feature selection config was updated."""

    print("🔍 Checking feature selection configuration...")

    config_file = Path("src/config/feature_selection_config.yaml")

    if not config_file.exists():
        print(f"❌ Feature selection config not found: {config_file}")
        return False

    try:
        with open(config_file, "r") as f:
            content = f.read()

        # Check for updated settings
        expected_settings = [
            "max_removal_percentage: 0.7",
            "emergency_override_perfect_correlation: true",
            "emergency_override_infinite_vif: true",
            "emergency_override_zero_importance: true",
        ]

        missing_settings = []
        for setting in expected_settings:
            if setting not in content:
                missing_settings.append(setting)

        if missing_settings:
            print("❌ Feature selection config validation failed! Missing settings:")
            for setting in missing_settings:
                print(f"   - {setting}")
            return False

        print("✅ Feature selection config validation passed!")
        return True

    except Exception as e:
        print(f"❌ Error checking feature selection config: {e}")
        return False


def main():
    """Main function to run the validation."""

    print("🚀 Starting multicollinearity fix validation...")

    # Validate the feature engineering fix
    feature_eng_ok = validate_fix()

    # Validate the feature selection config
    config_ok = check_feature_selection_config()

    if feature_eng_ok and config_ok:
        print("\n🎉 MULTICOLLINEARITY FIX VALIDATION PASSED!")
        print("✅ Your feature engineering fix has been applied correctly.")
        print("✅ Your feature selection config has been updated.")
        print("\n📋 Summary of fixes applied:")
        print(
            "   1. ✅ Fixed multi-timeframe price_change calculations with proper periods"
        )
        print(
            "   2. ✅ Fixed multi-timeframe volume_change calculations with proper periods"
        )
        print("   3. ✅ Increased max_removal_percentage from 0.3 to 0.7")
        print("   4. ✅ Added emergency override settings for perfect correlations")
        print("\n🔍 Next steps:")
        print("   1. Test your training pipeline again")
        print("   2. Monitor the logs for any remaining issues")
        print("   3. The VIF should no longer be infinite")
        return True
    else:
        print("\n❌ MULTICOLLINEARITY FIX VALIDATION FAILED!")
        print("❌ Some fixes were not applied correctly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
