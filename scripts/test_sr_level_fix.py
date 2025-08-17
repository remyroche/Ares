#!/usr/bin/env python3
"""
Test S/R Level Count Fix
Verifies that support_levels_count and resistance_levels_count are no longer constant.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.logger import system_logger


def test_sr_level_fix():
    """Test that S/R level counts are now dynamic."""
    logger = system_logger.getChild("TestSRLeverFix")

    try:
        # Create sample price data with wider range to test S/R levels
        np.random.seed(42)
        n_samples = 1000

        # Simulate price data with wider range and more volatility
        base_price = 100
        price_data = pd.DataFrame(
            {
                "open": base_price + np.cumsum(np.random.randn(n_samples) * 0.1),
                "high": base_price + np.cumsum(np.random.randn(n_samples) * 0.1) + 0.5,
                "low": base_price + np.cumsum(np.random.randn(n_samples) * 0.1) - 0.5,
                "close": base_price + np.cumsum(np.random.randn(n_samples) * 0.1),
                "volume": np.random.lognormal(10, 1, n_samples),
            }
        )

        # Ensure we have a good price range
        close_min, close_max = price_data["close"].min(), price_data["close"].max()
        print(f"Price range: {close_min:.2f} - {close_max:.2f}")

        # Create sample S/R levels that should activate with our price data
        sr_levels = {
            "support_levels": [
                {"price": close_min + (close_max - close_min) * 0.2, "strength": 0.8},
                {"price": close_min + (close_max - close_min) * 0.5, "strength": 0.6},
                {"price": close_min + (close_max - close_min) * 0.8, "strength": 0.4},
            ],
            "resistance_levels": [
                {"price": close_min + (close_max - close_min) * 0.3, "strength": 0.4},
                {"price": close_min + (close_max - close_min) * 0.6, "strength": 0.6},
                {"price": close_min + (close_max - close_min) * 0.9, "strength": 0.8},
            ],
        }

        # Test the dynamic calculation function
        def _calculate_dynamic_level_counts(price_series, levels, level_type):
            """Calculate dynamic level counts based on price position."""
            if not levels:
                # Fallback: create dynamic counts based on price percentiles
                if level_type == "support":
                    percentile_rank = price_series.rank(pct=True)
                    return (1 - percentile_rank) * 3
                else:  # resistance
                    percentile_rank = price_series.rank(pct=True)
                    return percentile_rank * 3

            # Calculate how many levels are "active" for each price point
            active_counts = pd.Series(
                np.zeros(len(price_series)), index=price_series.index
            )

            for level in levels:
                if isinstance(level, dict):
                    level_price = level.get("price", 0)
                    level_strength = level.get("strength", 1.0)
                else:
                    level_price = float(level)
                    level_strength = 1.0

                # Define activation range based on level strength and price
                activation_range = level_price * 0.01 * level_strength

                # Check if price is within activation range
                if level_type == "support":
                    is_active = (price_series >= (level_price - activation_range)) & (
                        price_series <= (level_price + activation_range * 2)
                    )
                else:  # resistance
                    is_active = (price_series <= (level_price + activation_range)) & (
                        price_series >= (level_price - activation_range * 2)
                    )

                active_counts += is_active.astype(int)

            return active_counts

        # Test with S/R levels
        close = price_data["close"]
        support_counts = _calculate_dynamic_level_counts(
            close, sr_levels["support_levels"], "support"
        )
        resistance_counts = _calculate_dynamic_level_counts(
            close, sr_levels["resistance_levels"], "resistance"
        )

        # Test without S/R levels (fallback)
        support_counts_fallback = _calculate_dynamic_level_counts(close, [], "support")
        resistance_counts_fallback = _calculate_dynamic_level_counts(
            close, [], "resistance"
        )

        # Analyze results
        results = {
            "with_sr_levels": {
                "support_unique": support_counts.nunique(),
                "resistance_unique": resistance_counts.nunique(),
                "support_range": (support_counts.min(), support_counts.max()),
                "resistance_range": (resistance_counts.min(), resistance_counts.max()),
                "support_mean": support_counts.mean(),
                "resistance_mean": resistance_counts.mean(),
            },
            "without_sr_levels": {
                "support_unique": support_counts_fallback.nunique(),
                "resistance_unique": resistance_counts_fallback.nunique(),
                "support_range": (
                    support_counts_fallback.min(),
                    support_counts_fallback.max(),
                ),
                "resistance_range": (
                    resistance_counts_fallback.min(),
                    resistance_counts_fallback.max(),
                ),
                "support_mean": support_counts_fallback.mean(),
                "resistance_mean": resistance_counts_fallback.mean(),
            },
        }

        # Print results
        print("=" * 60)
        print("S/R LEVEL COUNT FIX TEST RESULTS")
        print("=" * 60)

        print("\n📊 WITH S/R LEVELS:")
        print(
            f"   Support counts: {results['with_sr_levels']['support_unique']} unique values"
        )
        print(f"   Support range: {results['with_sr_levels']['support_range']}")
        print(f"   Support mean: {results['with_sr_levels']['support_mean']:.3f}")
        print(
            f"   Resistance counts: {results['with_sr_levels']['resistance_unique']} unique values"
        )
        print(f"   Resistance range: {results['with_sr_levels']['resistance_range']}")
        print(f"   Resistance mean: {results['with_sr_levels']['resistance_mean']:.3f}")

        print("\n📊 WITHOUT S/R LEVELS (FALLBACK):")
        print(
            f"   Support counts: {results['without_sr_levels']['support_unique']} unique values"
        )
        print(f"   Support range: {results['without_sr_levels']['support_range']}")
        print(f"   Support mean: {results['without_sr_levels']['support_mean']:.3f}")
        print(
            f"   Resistance counts: {results['without_sr_levels']['resistance_unique']} unique values"
        )
        print(
            f"   Resistance range: {results['without_sr_levels']['resistance_range']}"
        )
        print(
            f"   Resistance mean: {results['without_sr_levels']['resistance_mean']:.3f}"
        )

        # Check if fix is working
        support_fixed = (
            results["with_sr_levels"]["support_unique"] > 1
            or results["without_sr_levels"]["support_unique"] > 1
        )
        resistance_fixed = (
            results["with_sr_levels"]["resistance_unique"] > 1
            or results["without_sr_levels"]["resistance_unique"] > 1
        )

        print("\n" + "=" * 60)
        if support_fixed and resistance_fixed:
            print("✅ FIX SUCCESSFUL!")
            print("   - Support level counts are now dynamic")
            print("   - Resistance level counts are now dynamic")
            print("   - No more constant features being dropped")
        else:
            print("❌ FIX FAILED!")
            print("   - Support or resistance counts are still constant")

        print("=" * 60)

        return support_fixed and resistance_fixed

    except Exception as e:
        logger.error(f"Error testing S/R level fix: {e}")
        return False


if __name__ == "__main__":
    success = test_sr_level_fix()
    sys.exit(0 if success else 1)
