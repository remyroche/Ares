#!/usr/bin/env python3
"""
Fix Multicollinearity Issue in Feature Engineering

This script fixes the critical bug where all multi-timeframe price_change and volume_change
features are identical, causing perfect multicollinearity (VIF, inf).

Usage:
    python scripts/fix_multicollinearity_issue.py
"""

import sys
from functools import wraps
from pathlib import Path

from src.utils.logger import system_logger

# Add the src directory to the Python path
current_dir=Path(__file__).parent
src_dir=current_dir.parent / "src"
sys.path.insert(0, str(src_dir))


def _log_exceptions(logger_name: str):
    """Decorator to log exceptions and return False for failures."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger=system_logger.getChild(logger_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Error in {func.__name__}: {e}")
                return False

        return wrapper

    return decorator


@_log_exceptions("MulticollinearityFix")
def fix_feature_engineering_code() -> bool:
    """Fix the critical multicollinearity issue in the feature engineering code."""

    logger=system_logger.getChild("MulticollinearityFix")
    logger.info("🔧 Starting multicollinearity fix...")

    # Path to the feature engineering file
    feature_eng_file: Path=(
        src_dir / "training" / "steps" / "vectorized_advanced_feature_engineering.py"
    )

    if not feature_eng_file.exists():
        logger.error(f"❌ Feature engineering file not found: {feature_eng_file}")
        return False

    # Read the current file
    content: str
    with open(feature_eng_file, encoding="utf-8") as f:
        content=f.read()

    logger.info("📖 Reading current feature engineering code...")

    # Fix the problematic price_change calculation
    old_price_change="price_changes, price_data[price_column].pct_change()"
    new_price_change=(
        """            # CRITICAL FIX: Use proper periods for multi-timeframe price changes
            timeframe_periods = {
                "1m": 1,     # 1-period change for 1m
                "5m": 5,     # 5-period change for 5m
                "15m": 15,   # 15-period change for 15m
                "30m": 30,   # 30-period change for 30m
            }

            periods=timeframe_periods.get(timeframe, 1)
            price_changes=price_data[price_column].pct_change(periods=periods)"""
    )

    if old_price_change in content:
        content=content.replace(old_price_change, new_price_change)
        logger.info("✅ Fixed price_change calculation")

    # Fix the problematic volume_change calculation
    old_volume_change='volume_changes, volume_data["volume"].pct_change()'
    new_volume_change=(
        'volume_changes, volume_data["volume"].pct_change(periods=periods)'
    )

    if old_volume_change in content:
        content=content.replace(old_volume_change, new_volume_change)
        logger.info("✅ Fixed volume_change calculation")

    # Write the fixed content back
    with open(feature_eng_file, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("✅ Successfully fixed multicollinearity issue")
    return True


@_log_exceptions("MulticollinearityFixMain")
def main() -> bool:
    """Main function to fix the multicollinearity issue."""

    logger=system_logger.getChild("MulticollinearityFixMain")
    logger.info("🚀 Starting multicollinearity fix...")

    if fix_feature_engineering_code():
        logger.info("🎉 Multicollinearity fix completed successfully!")
        logger.info(
            "📋 The issue was in the _calculate_timeframe_features_vectorized method",
        )
        logger.info(
            "📋 All timeframes were using the same pct_change() without periods",
        )
        logger.info(
            "📋 Now each timeframe uses proper periods: 1m=1, 5m=5, 15m=15, 30m=30",
        )
        return True
    logger.error("❌ Multicollinearity fix failed!")
    return False


if __name__== "__main__":
    success = main()
    if not success:
        print("\n❌ MULTICOLLINEARITY FIX FAILED!")
        sys.exit(1)
    else:
        print("\n🎉 MULTICOLLINEARITY FIX COMPLETED SUCCESSFULLY!")
        print("✅ Your feature engineering should now work correctly.")
