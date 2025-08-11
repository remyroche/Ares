#!/usr/bin/env python3
"""
Enhanced NaN Analysis Example

This script demonstrates the enhanced NaN detection functionality that includes
timestamp ranges for missing data, making it easier to identify patterns and
time periods with data gaps.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_realistic_trading_data():
    """Create realistic trading data with various NaN patterns."""
    print("📊 Creating realistic trading data with NaN patterns...")

    # Create 7 days of minute-level data (24 hours * 60 minutes * 7 days)
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    dates = pd.date_range(start_date, periods=7 * 24 * 60, freq="1min")

    # Create realistic price data
    np.random.seed(42)
    base_price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.001)

    price_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": base_price + np.random.randn(len(dates)) * 0.1,
            "high": base_price + np.random.randn(len(dates)) * 0.2,
            "low": base_price - np.random.randn(len(dates)) * 0.2,
            "close": base_price + np.random.randn(len(dates)) * 0.1,
            "volume": np.random.randint(1000, 10000, len(dates)),
        }
    )

    # Introduce realistic NaN patterns

    # Pattern 1: Weekend gaps (no trading on weekends)
    weekend_mask = dates.weekday >= 5  # Saturday = 5, Sunday = 6
    price_data.loc[weekend_mask, ["open", "high", "low", "close"]] = np.nan
    price_data.loc[weekend_mask, "volume"] = np.nan

    # Pattern 2: Market hours gaps (simulate market closed hours)
    # Assume market is closed from 22:00 to 06:00 UTC
    market_closed_mask = (dates.hour >= 22) | (dates.hour < 6)
    price_data.loc[market_closed_mask, ["open", "high", "low", "close"]] = np.nan
    price_data.loc[market_closed_mask, "volume"] = np.nan

    # Pattern 3: Random data gaps (simulate API issues, network problems)
    # Create some random gaps during market hours
    market_hours_mask = ~market_closed_mask & ~weekend_mask
    market_hours_indices = price_data[market_hours_mask].index

    # Random gaps
    gap_indices = np.random.choice(market_hours_indices, size=50, replace=False)
    price_data.loc[gap_indices, "volume"] = np.nan

    # Longer gaps (simulate maintenance)
    maintenance_start = datetime(2024, 1, 3, 14, 0, 0)  # 2 PM on Jan 3rd
    maintenance_end = datetime(2024, 1, 3, 16, 0, 0)  # 4 PM on Jan 3rd
    maintenance_mask = (dates >= maintenance_start) & (dates <= maintenance_end)
    price_data.loc[maintenance_mask, ["open", "high", "low", "close", "volume"]] = (
        np.nan
    )

    # Set timestamp as index
    price_data.set_index("timestamp", inplace=True)

    return price_data


def analyze_nan_patterns(data):
    """Analyze NaN patterns in the data."""
    print("\n🔍 Analyzing NaN patterns in trading data...")

    total_rows = len(data)
    total_cells = data.size

    print(f"📊 Dataset Overview:")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Total cells: {total_cells:,}")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")

    # Analyze each column
    for col in data.columns:
        nan_mask = data[col].isnull()
        nan_count = nan_mask.sum()
        nan_percentage = (nan_count / total_rows) * 100

        if nan_count > 0:
            print(f"\n📋 Column: {col}")
            print(f"   NaN count: {nan_count:,} ({nan_percentage:.2f}%)")

            # Get timestamp ranges for NaN values
            nan_indices = data.index[nan_mask]
            ranges = group_consecutive_timestamps(nan_indices)

            print(f"   📅 Found {len(ranges)} NaN timestamp range(s):")
            for i, range_info in enumerate(ranges, 1):
                start_time = range_info["start"].strftime("%Y-%m-%d %H:%M:%S")
                end_time = range_info["end"].strftime("%Y-%m-%d %H:%M:%S")
                duration_hours = range_info["duration_minutes"] / 60
                count = range_info["count"]

                print(f"      Range {i}: {start_time} to {end_time}")
                print(
                    f"         Duration: {duration_hours:.1f} hours ({range_info['duration_minutes']:.0f} minutes)"
                )
                print(f"         NaN count: {count:,}")

                # Identify pattern type
                pattern_type = identify_pattern_type(range_info)
                print(f"         Pattern: {pattern_type}")
        else:
            print(f"\n✅ Column: {col} - No NaN values")

    # Overall statistics
    total_nans = data.isnull().sum().sum()
    total_nan_percentage = (total_nans / total_cells) * 100

    print(f"\n📈 Overall Statistics:")
    print(f"   Total NaN values: {total_nans:,}")
    print(f"   Total NaN percentage: {total_nan_percentage:.2f}%")
    print(f"   Data quality score: {max(0, 100 - total_nan_percentage):.1f}/100")


def group_consecutive_timestamps(timestamps):
    """Group consecutive timestamps into ranges."""
    if len(timestamps) == 0:
        return []

    ranges = []
    start_idx = 0

    for i in range(1, len(timestamps)):
        current_time = timestamps[i]
        prev_time = timestamps[i - 1]
        time_diff = current_time - prev_time

        # If gap is more than 2 minutes, start a new range
        if time_diff > pd.Timedelta(minutes=2):
            range_info = {
                "start": timestamps[start_idx],
                "end": timestamps[i - 1],
                "duration_minutes": (
                    timestamps[i - 1] - timestamps[start_idx]
                ).total_seconds()
                / 60,
                "count": i - start_idx,
            }
            ranges.append(range_info)
            start_idx = i

    # Add the last range
    if start_idx < len(timestamps):
        range_info = {
            "start": timestamps[start_idx],
            "end": timestamps[-1],
            "duration_minutes": (timestamps[-1] - timestamps[start_idx]).total_seconds()
            / 60,
            "count": len(timestamps) - start_idx,
        }
        ranges.append(range_info)

    return ranges


def identify_pattern_type(range_info):
    """Identify the type of NaN pattern."""
    start_time = range_info["start"]
    end_time = range_info["end"]
    duration_hours = range_info["duration_minutes"] / 60

    # Weekend pattern
    if start_time.weekday() >= 5 or end_time.weekday() >= 5:
        return "Weekend (No Trading)"

    # Market closed hours
    if (start_time.hour >= 22 or start_time.hour < 6) or (
        end_time.hour >= 22 or end_time.hour < 6
    ):
        return "Market Closed Hours"

    # Long maintenance period
    if duration_hours >= 2:
        return "Maintenance Period"

    # Short gaps
    if duration_hours < 0.5:
        return "Brief Data Gap"

    return "Unknown Pattern"


def main():
    """Main function to demonstrate enhanced NaN analysis."""
    print("🚀 Enhanced NaN Analysis with Timestamp Ranges")
    print("=" * 60)

    # Create realistic trading data
    trading_data = create_realistic_trading_data()

    # Analyze NaN patterns
    analyze_nan_patterns(trading_data)

    print("\n" + "=" * 60)
    print("💡 Key Benefits of Enhanced NaN Analysis:")
    print("   ✅ Identifies specific time periods with missing data")
    print("   ✅ Groups consecutive NaN values into meaningful ranges")
    print("   ✅ Provides duration and count for each gap")
    print("   ✅ Helps identify patterns (weekends, maintenance, etc.)")
    print("   ✅ Enables targeted data collection strategies")
    print("=" * 60)


if __name__ == "__main__":
    main()
