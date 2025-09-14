#!/usr/bin/env python3
"""
Script to investigate what is causing infinity values around row 4581
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def investigate_infinity_values():
    """Investigate what features have infinity values around row 4581"""

    print("🔍 Investigating infinity values around row 4581...")
    print("📋 Based on the terminal selection, infinity values were found in features:")
    print("   • Feature indices: 65, 66, 67, 68, 69, 8, 72")
    print("   • Affected rows: 4581-4600 range")
    print("   • Types: Both positive and negative infinity")

    # First, let's understand what these feature indices might correspond to
    print("\n🔬 Analyzing potential feature mappings:")
    print("   Based on feature engineering patterns, these indices likely correspond to:")

    # Common feature engineering patterns
    potential_features = {
        8: "Early feature (price_range_pct, volatility_pct, or basic ratio)",
        65: "Complex feature (rolling calculation, cross-timeframe interaction)",
        66: "Complex feature (momentum interaction, polynomial feature)",
        67: "Complex feature (pattern recognition, regime-dependent feature)",
        68: "Complex feature (volatility interaction, advanced indicator)",
        69: "Complex feature (momentum-volatility interaction)",
        72: "Complex feature (advanced mathematical operation)"
    }

    for idx, description in potential_features.items():
        print(f"   • Feature {idx}: {description}")

    print("\n⚠️ Common causes of infinity in feature engineering:")
    print("   1. Division by zero: price_range_pct = price_range / close (when close = 0)")
    print("   2. Percentage changes: pct_change() on zero or very small values")
    print("   3. Rolling operations: std() on constant data segments")
    print("   4. Complex ratios: volume_ratio = volume / avg_volume (when avg_volume = 0)")
    print("   5. Log transformations: log(negative_value) or log(0)")
    print("   6. Power operations: extreme exponents on small values")

    # Look for data files that might contain the features
    data_paths = [
        "/Users/remyroche/Documents/Ares/data_cache",
        "/Users/remyroche/Documents/Ares/historical_data",
        "/Users/remyroche/Documents/Ares/artifacts/dataframes"
    ]

    feature_files = []

    # Find all parquet and CSV files
    for data_path in data_paths:
        if os.path.exists(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith(('.parquet', '.csv')) and ('feature' in file.lower() or 'engineer' in file.lower()):
                        feature_files.append(os.path.join(root, file))

    print(f"\n📁 Found {len(feature_files)} potential feature files:")
    for f in feature_files[:5]:  # Show first 5
        print(f"   {f}")

    if not feature_files:
        print("❌ No feature files found. Looking for any data files...")
        # Look for any data files
        for data_path in data_paths:
            if os.path.exists(data_path):
                for root, dirs, files in os.walk(data_path):
                    for file in files:
                        if file.endswith('.parquet'):
                            feature_files.append(os.path.join(root, file))

        print(f"📁 Found {len(feature_files)} parquet files:")
        for f in feature_files[:3]:
            print(f"   {f}")

    # Try to examine the largest file that might contain features
    for file_path in feature_files:
        try:
            print(f"\n📊 Examining: {file_path}")

            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)

            print(f"   📏 Shape: {df.shape}")
            print(f"   📊 Columns: {len(df.columns)}")

            if len(df) > 4581:  # Check if file has enough rows
                print(f"   ✅ Has data beyond row 4581")

                # Check for infinity values - handle different data types
                try:
                    # Convert to float for infinity checking
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) == 0:
                        print(f"   ❌ No numeric columns in this file")
                        continue

                    # Check for infinity values in numeric columns only
                    inf_count = 0
                    inf_cols = []

                    for col in numeric_df.columns:
                        try:
                            col_data = numeric_df[col].astype(float)
                            col_inf = np.isinf(col_data.values)
                            col_inf_count = np.sum(col_inf)
                            if col_inf_count > 0:
                                inf_count += col_inf_count
                                inf_cols.append(col)
                        except Exception as e:
                            print(f"      Warning checking {col}: {e}")
                            continue

                    if inf_count > 0:
                        print(f"   ⚠️ Found {inf_count} infinity values in {len(inf_cols)} columns!")

                        # Check infinity values around row 4581
                        start_row = max(0, 4581 - 5)
                        end_row = min(len(df), 4581 + 5)

                        subset = numeric_df.iloc[start_row:end_row]

                        # Check for infinity in the subset
                        subset_inf_cols = []
                        for col in inf_cols:
                            try:
                                col_data = subset[col].astype(float)
                                if np.any(np.isinf(col_data.values)):
                                    subset_inf_cols.append(col)
                            except:
                                continue

                        if subset_inf_cols:
                            print(f"   🎯 Infinity values found around row 4581 in columns: {subset_inf_cols[:5]}{'...' if len(subset_inf_cols) > 5 else ''}")

                            # Show specific values around row 4581
                            for col_name in subset_inf_cols[:3]:
                                try:
                                    values_around = subset[col_name].astype(float)
                                    inf_mask = np.isinf(values_around.values)
                                    if np.any(inf_mask):
                                        inf_rows = values_around.index[inf_mask]
                                        print(f"      {col_name}: infinity at rows {list(inf_rows)}")
                                        # Show the actual values
                                        print(f"         Values: {values_around[inf_mask].head(3).tolist()}")
                                except Exception as e:
                                    print(f"         Error examining {col_name}: {e}")
                            break
                        else:
                            print(f"   ✅ No infinity values around row 4581 in this file")
                    else:
                        print(f"   ✅ No infinity values in numeric columns")

                except Exception as e:
                    print(f"   ❌ Error checking for infinity values: {e}")

            else:
                print(f"   ❌ File too small ({len(df)} rows < 4581)")

        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            continue

    # If we didn't find the problematic file, let's try to understand what might cause infinity values
    print("\n🔬 Analyzing potential causes of infinity values:")

    # Common mathematical operations that can cause infinity
    potential_causes = [
        "Division by zero (e.g., safe_divide with zero denominator)",
        "Percentage change calculations (pct_change) on zero or very small values",
        "Standard deviation calculations on constant data",
        "Rolling window calculations with insufficient data",
        "Logarithmic transformations of zero or negative values",
        "Power operations with extreme exponents",
        "Complex mathematical operations in feature engineering"
    ]

    for cause in potential_causes:
        print(f"   • {cause}")

    print("\n💡 Recommendation: Check the feature engineering pipeline for:")
    print("   - Division operations that might divide by zero")
    print("   - Percentage change calculations on problematic data")
    print("   - Rolling window operations with insufficient lookback")

if __name__ == "__main__":
    investigate_infinity_values()
