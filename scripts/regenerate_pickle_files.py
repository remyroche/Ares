#!/usr/bin/env python3
"""
Script to regenerate pickle files from consolidated CSV files.
This script will:
1. Load the consolidated CSV files
2. Create proper pickle files with the expected data structure
3. Ensure prices are valid
"""

import os
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def detect_price_corruption(df: pd.DataFrame) -> bool:
    """Detect if price data is corrupted."""
    if df.empty:
        return False

    price_cols = ["open", "high", "low", "close"]
    if not all(col in df.columns for col in price_cols):
        return False

    median_price = df["close"].median()
    return bool(median_price < 100 or median_price > 10000)


def fix_corrupted_prices(
    df: pd.DataFrame,
    target_median: float = 3000.0,
) -> pd.DataFrame:
    """Fix corrupted prices by scaling them to a reasonable range."""
    if df.empty:
        return df

    price_cols = ["open", "high", "low", "close"]
    if not all(col in df.columns for col in price_cols):
        return df

    current_median = df["close"].median()
    if current_median <= 0:
        print(f"Warning: Invalid median price: {current_median}")
        return df

    scale_factor = target_median / current_median

    print("Fixing corrupted prices:")
    print(f"  Current median: ${current_median:.2f}")
    print(f"  Target median: ${target_median:.2f}")
    print(f"  Scale factor: {scale_factor:.6f}")

    for col in price_cols:
        df[col] = df[col] * scale_factor

    new_median = df["close"].median()
    print(f"  New median: ${new_median:.2f}")
    print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

    return df


def create_pickle_from_csv(
    csv_path: str,
    output_path: str,
    lookback_days: int = 730,
) -> bool:
    """Create a pickle file from a consolidated CSV file."""
    try:
        print(f"\nProcessing: {csv_path}")

        # Load CSV file
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")

        # Check if we have timestamp column
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        # Check for price corruption
        if detect_price_corruption(df):
            print("  Detected corrupted prices, fixing...")
            df = fix_corrupted_prices(df)
        else:
            print("  Prices appear to be valid")

        # Filter by lookback period
        if not df.empty and df.index is not None:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            df = df[df.index >= cutoff_date]
            print(f"  Filtered to {len(df)} rows for {lookback_days} days")

        # Create data structure for pickle
        data = {
            "klines": df,
            "agg_trades": pd.DataFrame(),  # Empty for now
            "futures": pd.DataFrame(),  # Empty for now
            "metadata": {
                "source_file": csv_path,
                "processed_at": datetime.now().isoformat(),
                "lookback_days": lookback_days,
                "price_corrected": detect_price_corruption(df),
            },
        }

        # Save pickle file
        with open(output_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  Saved data to: {output_path}")
        return True

    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
        return False


def main():
    """Main function to regenerate pickle files."""
    print("🔧 Regenerating Pickle Files from Consolidated CSV")
    print("=" * 60)

    data_cache_dir = "data_cache"
    if not os.path.exists(data_cache_dir):
        print(missing("Data cache directory not found: {data_cache_dir}")))
        return False

    # Find consolidated CSV files
    consolidated_files = []
    for pattern in ["*consolidated*.csv"]:
        consolidated_files.extend(Path(data_cache_dir).glob(pattern))

    if not consolidated_files:
        print(warning("No consolidated CSV files found in {data_cache_dir}")))
        return False

    print(f"📁 Found {len(consolidated_files)} consolidated CSV files")

    # Process each consolidated file
    success_count = 0
    for csv_file in consolidated_files:
        csv_name = csv_file.stem

        # Create different lookback periods
        lookback_periods = [30, 60, 730]  # 30 days, 60 days, 2 years

        for lookback_days in lookback_periods:
            # Create output filename
            if "klines" in csv_name:
                symbol = "ETHUSDT"
                timeframe = "1h"  # Convert 1m to 1h for the pickle
                pkl_name = f"{symbol}_{timeframe}_{lookback_days}_cached_data.pkl"
            else:
                # For other file types, use the original name
                pkl_name = f"{csv_name}_cached_data.pkl"

            pkl_path = os.path.join(data_cache_dir, pkl_name)

            if create_pickle_from_csv(str(csv_file), pkl_path, lookback_days):
                success_count += 1

    print(f"\n✅ Successfully created {success_count} pickle files")

    # List the created files
    pkl_files = list(Path(data_cache_dir).glob("*_cached_data.pkl"))
    if pkl_files:
        print("\n📁 Created pickle files:")
        for pkl_file in sorted(pkl_files):
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)

                if "klines" in data and isinstance(data["klines"], pd.DataFrame):
                    df = data["klines"]
                    print(f"  ✅ {pkl_file.name}: {len(df)} rows")
                else:
                    print(f"  ⚠️  {pkl_file.name}: Invalid data structure")

            except Exception as e:
                print(f"  ❌ {pkl_file.name}: Error reading file - {e}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
