#!/usr/bin/env python3
"""
Debug script to investigate data quality issues:
1. Infinite values in volume_return and volume_log_return
2. Duplicate timestamps
3. Data type detection issues
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def debug_data_issues():
    """Debug the data quality issues."""

    # Path to the features file
    features_path = "/Users/remyroche/Documents/Ares/historical_data/binance/ethusdt/processed/ethusdt_1m/features_ethusdt_1m_consolidated_cleaned.parquet"

    print("🔍 Loading data file...")
    try:
        df = pd.read_parquet(features_path)
        print(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Check for infinite values in volume columns
    print("\n📊 Checking for infinite values...")

    if 'volume_return' in df.columns:
        inf_count = np.isinf(df['volume_return']).sum()
        print(f"   - volume_return: {inf_count} infinite values")

        if inf_count > 0:
            print("   - Sample infinite values:")
            inf_mask = np.isinf(df['volume_return'])
            print(df.loc[inf_mask, ['timestamp', 'volume', 'volume_return']].head(5))

    if 'volume_log_return' in df.columns:
        inf_count = np.isinf(df['volume_log_return']).sum()
        print(f"   - volume_log_return: {inf_count} infinite values")

        if inf_count > 0:
            print("   - Sample infinite values:")
            inf_mask = np.isinf(df['volume_log_return'])
            print(df.loc[inf_mask, ['timestamp', 'volume', 'volume_log_return']].head(5))

    # Check for duplicate timestamps
    print("\n📅 Checking for duplicate timestamps...")

    if 'timestamp' in df.columns:
        duplicate_count = df['timestamp'].duplicated().sum()
        total_count = len(df)
        duplicate_pct = (duplicate_count / total_count) * 100

        print(f"   - Total timestamps: {total_count}")
        print(f"   - Duplicate timestamps: {duplicate_count}")
        print(f"   - Duplicate percentage: {duplicate_pct:.2f}%")

        if duplicate_count > 0:
            print("   - Sample duplicates:")
            duplicates = df[df['timestamp'].duplicated(keep=False)]
            print(duplicates[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head(10))

            # Check if duplicates have different values
            duplicate_timestamps = df['timestamp'].value_counts()
            problematic_timestamps = duplicate_timestamps[duplicate_timestamps > 1]
            print(f"\n   - Timestamps with multiple entries: {len(problematic_timestamps)}")
            print("   - Top problematic timestamps:")
            print(problematic_timestamps.head(5))

    # Check data structure
    print("\n📋 Data structure analysis...")

    # Check column types
    print("   - Column types:")
    for col in df.columns[:10]:  # Show first 10 columns
        print(f"     {col}: {df[col].dtype}")

    # Check for OHLCV columns
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    has_ohlcv = all(col in df.columns for col in ohlcv_cols)
    print(f"\n   - Has OHLCV columns: {has_ohlcv}")

    if has_ohlcv:
        print("   - OHLCV sample:")
        print(df[ohlcv_cols].head(3))

    # Check for trade-related columns (aggtrades indicators)
    trade_cols = ['trade_count', 'buy_volume', 'sell_volume', 'aggression_ratio']
    has_trade_cols = any(col in df.columns for col in trade_cols)
    print(f"   - Has trade-related columns: {has_trade_cols}")

    if has_trade_cols:
        found_trade_cols = [col for col in trade_cols if col in df.columns]
        print(f"   - Found trade columns: {found_trade_cols}")

    # Check timestamp intervals
    if 'timestamp' in df.columns and len(df) > 1:
        print("\n⏱️  Timestamp analysis...")
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()

        # Convert to seconds for analysis
        time_diffs_seconds = time_diffs.dt.total_seconds()

        print(f"   - Time differences (seconds):")
        print(f"     Min: {time_diffs_seconds.min()}")
        print(f"     Max: {time_diffs_seconds.max()}")
        print(f"     Mean: {time_diffs_seconds.mean():.2f}")
        print(f"     Median: {time_diffs_seconds.median():.2f}")

        # Most common intervals
        common_intervals = time_diffs_seconds.value_counts().head(5)
        print(f"   - Most common intervals:")
        for interval, count in common_intervals.items():
            print(f"     {interval}s: {count} occurrences")

if __name__ == "__main__":
    debug_data_issues()
