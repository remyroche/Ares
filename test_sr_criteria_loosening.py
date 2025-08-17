#!/usr/bin/env python3
"""
Test script to verify S/R criteria loosening changes.
This script tests the VectorizedSRDistanceCalculator with the new loosened criteria.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from training.steps.vectorized_advanced_feature_engineering import (
    VectorizedSRDistanceCalculator,
)


def create_test_data():
    """Create test price data with some volatility."""
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=1000, freq="1min")

    # Create price data with some trends and volatility
    base_price = 1000
    price_changes = np.random.normal(0, 0.001, 1000)  # 0.1% volatility
    prices = [base_price]

    for i in range(1, 1000):
        # Add some trend and mean reversion
        trend = 0.0001 * np.sin(i / 100)  # Small cyclical trend
        price = prices[-1] * (1 + price_changes[i] + trend)
        prices.append(price)

    # Create OHLC data
    data = pd.DataFrame(
        {
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
            "close": prices,
            "volume": np.random.randint(1000, 10000, 1000),
        },
        index=dates,
    )

    return data


def create_test_sr_levels():
    """Create test S/R levels."""
    # Create more granular support and resistance levels that match the test data range
    support_levels = [
        {"price": 995, "strength": 0.8},
        {"price": 998, "strength": 0.6},
        {"price": 1000, "strength": 0.9},
        {"price": 1002, "strength": 0.7},
        {"price": 1005, "strength": 0.5},
        {"price": 1008, "strength": 0.6},
        {"price": 1010, "strength": 0.7},
    ]

    resistance_levels = [
        {"price": 1000, "strength": 0.8},
        {"price": 1002, "strength": 0.7},
        {"price": 1005, "strength": 0.9},
        {"price": 1008, "strength": 0.6},
        {"price": 1010, "strength": 0.5},
        {"price": 1012, "strength": 0.7},
        {"price": 1015, "strength": 0.6},
    ]

    return {"support_levels": support_levels, "resistance_levels": resistance_levels}


async def test_sr_calculator():
    """Test the S/R calculator with new loosened criteria."""
    print("🧪 Testing S/R criteria loosening...")

    # Create test data
    price_data = create_test_data()
    sr_levels = create_test_sr_levels()

    # Configuration with loosened criteria
    config = {
        "vectorized_labelling_orchestrator": {
            "sr_distance_scale": 0.02,  # 2% scale
            "sr_proximity_threshold": 0.05,  # 5% threshold
            "sr_activation_range_multiplier": 0.03,  # 3% activation range
            "sr_fallback_range": 8,  # 0-8 range
            "sr_volatility_factor": 0.1,  # Volatility factor (reduced)
        }
    }

    # Initialize calculator
    calculator = VectorizedSRDistanceCalculator(config)
    await calculator.initialize()

    # Test with S/R levels
    print("\n🔍 Testing with S/R levels...")
    results = await calculator.calculate_sr_distances(price_data, sr_levels)

    # Analyze results
    support_counts = pd.Series(results["support_levels_count"])
    resistance_counts = pd.Series(results["resistance_levels_count"])

    print(f"\n📊 Results Analysis (with S/R levels):")
    print(f"Support counts unique values: {support_counts.nunique()}")
    print(f"Resistance counts unique values: {resistance_counts.nunique()}")
    print(
        f"Support counts range: {support_counts.min():.2f} - {support_counts.max():.2f}"
    )
    print(
        f"Resistance counts range: {resistance_counts.min():.2f} - {resistance_counts.max():.2f}"
    )
    print(f"Support counts mean: {support_counts.mean():.2f}")
    print(f"Resistance counts mean: {resistance_counts.mean():.2f}")

    # Test fallback case (no S/R levels)
    print("\n🔍 Testing fallback case (no S/R levels)...")
    fallback_results = await calculator.calculate_sr_distances(price_data, {})

    fallback_support_counts = pd.Series(fallback_results["support_levels_count"])
    fallback_resistance_counts = pd.Series(fallback_results["resistance_levels_count"])

    print(f"\n📊 Results Analysis (fallback case):")
    print(f"Support counts unique values: {fallback_support_counts.nunique()}")
    print(f"Resistance counts unique values: {fallback_resistance_counts.nunique()}")
    print(
        f"Support counts range: {fallback_support_counts.min():.2f} - {fallback_support_counts.max():.2f}"
    )
    print(
        f"Resistance counts range: {fallback_resistance_counts.min():.2f} - {fallback_resistance_counts.max():.2f}"
    )
    print(f"Support counts mean: {fallback_support_counts.mean():.2f}")
    print(f"Resistance counts mean: {fallback_resistance_counts.mean():.2f}")

    # Check if we have more granularity than before
    total_success = True
    if support_counts.nunique() > 3 and resistance_counts.nunique() > 4:
        print("✅ SUCCESS: S/R criteria loosening worked with S/R levels!")
        print(f"   Previous: support=3, resistance=4 unique values")
        print(
            f"   Current:  support={support_counts.nunique()}, resistance={resistance_counts.nunique()} unique values"
        )
    else:
        print("⚠️  WARNING: S/R criteria may still be too restrictive with S/R levels.")
        total_success = False

    if (
        fallback_support_counts.nunique() > 3
        and fallback_resistance_counts.nunique() > 4
    ):
        print("✅ SUCCESS: S/R criteria loosening worked in fallback case!")
        print(f"   Previous: support=3, resistance=4 unique values")
        print(
            f"   Current:  support={fallback_support_counts.nunique()}, resistance={fallback_resistance_counts.nunique()} unique values"
        )
    else:
        print("⚠️  WARNING: S/R criteria may still be too restrictive in fallback case.")
        total_success = False

    if total_success:
        print(
            "\n🎉 OVERALL SUCCESS: S/R criteria loosening is working in both scenarios!"
        )

    # Show some sample values
    print(f"\n📈 Sample Support Counts (with S/R levels, first 10):")
    print(support_counts.head(10).values)
    print(f"\n📉 Sample Resistance Counts (with S/R levels, first 10):")
    print(resistance_counts.head(10).values)

    return results, fallback_results


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_sr_calculator())
