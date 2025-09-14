#!/usr/bin/env python3
"""
Debug script to investigate the specific rows with infinity values:
- Rows: 8751, 10985, 3270, 2540, 11418, 6219, 2456
- Features: 65, 66, 67, 8, 68, 69, 72
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def debug_infinity_rows():
    """Debug the specific rows with infinity values."""

    # Path to the features file
    features_path = "/Users/remyroche/Documents/Ares/historical_data/binance/ethusdt/processed/ethusdt_1m/features_ethusdt_1m_consolidated.parquet"

    print("🔍 Loading data file...")
    try:
        df = pd.read_parquet(features_path)
        print(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Problematic row indices from the terminal output
    problematic_rows = [8751, 10985, 3270, 2540, 11418, 6219, 2456]

    # Problematic feature indices from the terminal output
    problematic_features = [65, 66, 67, 8, 68, 69, 72]

    print("\n🎯 Examining problematic rows and features...")
    print(f"📊 Total rows in dataset: {len(df)}")
    print(f"📊 Total features in dataset: {len(df.columns)}")

    # Check if row indices are valid
    valid_rows = []
    for row_idx in problematic_rows:
        if 0 <= row_idx < len(df):
            valid_rows.append(row_idx)
        else:
            print(f"⚠️ Row index {row_idx} is out of bounds (0-{len(df)-1})")

    # Check if feature indices are valid
    valid_features = []
    for feat_idx in problematic_features:
        if 0 <= feat_idx < len(df.columns):
            valid_features.append(feat_idx)
        else:
            print(f"⚠️ Feature index {feat_idx} is out of bounds (0-{len(df.columns)-1})")

    print(f"✅ Valid rows to examine: {valid_rows}")
    print(f"✅ Valid features to examine: {valid_features}")

    # Get feature names for the problematic indices
    print("\n📋 Feature names for problematic indices:")
    for feat_idx in valid_features:
        if feat_idx < len(df.columns):
            feature_name = df.columns[feat_idx]
            print(f"  Feature {feat_idx}: {feature_name}")

    # Examine each problematic row
    for row_idx in valid_rows:
        print(f"\n🔍 Examining row {row_idx}:")

        # Get the row data
        row_data = df.iloc[row_idx]

        # Check timestamp if available
        if 'timestamp' in df.columns:
            timestamp = row_data['timestamp']
            print(f"  📅 Timestamp: {timestamp}")

        # Check for infinity values in this row
        row_data = df.iloc[row_idx]
        numeric_row = pd.to_numeric(row_data, errors='coerce')
        inf_mask = np.isinf(numeric_row.values)

        if np.any(inf_mask):
            inf_feature_indices = np.where(inf_mask)[0]
            print(f"  ⚠️ Infinity values found in {len(inf_feature_indices)} features:")
            for feat_idx in inf_feature_indices:
                if feat_idx < len(df.columns):
                    feature_name = df.columns[feat_idx]
                    value = numeric_row.iloc[feat_idx]
                    print(f"    - {feature_name} (idx {feat_idx}): {value}")

        # Check the specific problematic features for this row
        print(f"  📊 Values of problematic features in this row:")
        for feat_idx in valid_features:
            if feat_idx < len(df.columns):
                feature_name = df.columns[feat_idx]
                value = numeric_row.iloc[feat_idx]
                is_inf = np.isinf(value) if np.isfinite(value) else False
                status = "❌ INFINITY" if is_inf else "✅ finite"
                print(f"    - {feature_name} (idx {feat_idx}): {value} [{status}]")

        # Check related volume and price data that might cause infinity
        volume_related_cols = ['volume', 'volume_ma_5', 'volume_ma_10', 'volume_ma_20', 'volume_ma_50']
        price_related_cols = ['close', 'price_change', 'volume_change']

        print(f"  📈 Related data that might cause infinity:")
        for col in volume_related_cols + price_related_cols:
            if col in df.columns:
                value = row_data[col]
                print(f"    - {col}: {value}")

    # Check for patterns across all problematic rows
    print("\n📊 Analyzing patterns across problematic rows...")
    if len(valid_rows) > 0:
        problematic_subset = df.iloc[valid_rows]

        # Check volume patterns
        if 'volume' in df.columns:
            volumes = problematic_subset['volume'].values
            print(f"  📊 Volume values in problematic rows: {volumes}")
            print(f"  📊 Volume statistics: min={np.min(volumes)}, max={np.max(volumes)}, mean={np.mean(volumes):.2f}")

        # Check for zero or very small values that could cause division by zero
        print("\n🔍 Checking for zero/small values that could cause infinity:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols[:20]:  # Check first 20 numeric columns
            values = problematic_subset[col].values
            zero_count = np.sum(values == 0)
            very_small_count = np.sum(np.abs(values) < 1e-10)

            if zero_count > 0 or very_small_count > 0:
                print(f"  ⚠️ {col}: {zero_count} zeros, {very_small_count} very small values")

    # Overall infinity analysis
    print("\n📈 Overall infinity analysis...")
    numeric_data = df.select_dtypes(include=[np.number])
    inf_matrix = np.isinf(numeric_data.values)
    total_inf = np.sum(inf_matrix)
    inf_per_feature = np.sum(inf_matrix, axis=0)
    inf_per_row = np.sum(inf_matrix, axis=1)

    print(f"  📊 Total infinity values in dataset: {total_inf}")
    print(f"  📊 Features with infinity: {np.sum(inf_per_feature > 0)}")
    print(f"  📊 Rows with infinity: {np.sum(inf_per_row > 0)}")

    if total_inf > 0:
        print("  📊 Top features with most infinity values:")
        feature_inf_counts = list(zip(df.columns, inf_per_feature))
        feature_inf_counts.sort(key=lambda x: x[1], reverse=True)

        for feature_name, count in feature_inf_counts[:10]:
            if count > 0:
                print(f"    - {feature_name}: {count} infinity values")

if __name__ == "__main__":
    debug_infinity_rows()
