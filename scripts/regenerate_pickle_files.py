#!/usr/bin/env python3
"""
Script to regenerate pickle files from consolidated CSV files.
This script will:
1. Load the consolidated CSV files
2. Create proper pickle files with the expected data structure
3. Ensure prices are valid and timestamps are indexed
"""

from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import missing, warning

# Add the project root to the path
project_root=Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def detect_price_corruption(df: pd.DataFrame) -> bool:
    """Detect if price data is likely corrupted based on median range."""
    if df.empty:
        return False

    price_cols=["open", "high", "low", "close"]
    if not all(col in df.columns for col in price_cols):
        return False

    median_price=float(pd.to_numeric(df["close"], errors="coerce").median())
    return bool(median_price < 100.0 or median_price > 10000.0)


def fix_corrupted_prices(df: pd.DataFrame, target_median: float=3000.0) -> pd.DataFrame:
    """Fix corrupted prices by scaling them to a reasonable range.

    If prices appear way out of range, scale all OHLC columns so that the
    median close price equals target_median.
    """
    if df.empty:
        return df

    price_cols=["open", "high", "low", "close"]
    if not all(col in df.columns for col in price_cols):
        return df

    numeric_df=df.copy()
    for col in price_cols:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    current_median=float(numeric_df["close"].median())
    if current_median <= 0 or not pd.notna(current_median):
        print(warning(f"Invalid median price: {current_median}"))
        return df

    scale_factor=float(target_median / current_median)

    print("Fixing corrupted prices:")
    print(f"  Current median: ${current_median:.2f}")
    print(f"  Target median: ${target_median:.2f}")
    print(f"  Scale factor: {scale_factor:.6f}")

    fixed_df=df.copy()
    for col in price_cols:
        fixed_df[col] = pd.to_numeric(fixed_df[col], errors="coerce") * scale_factor

    new_median=float(pd.to_numeric(fixed_df["close"], errors="coerce").median())
    print(f"  New median: ${new_median:.2f}")
    print(
        f"  Price range: ${pd.to_numeric(fixed_df['close'], errors='coerce').min():.2f} "
        f"to ${pd.to_numeric(fixed_df['close'], errors='coerce').max():.2f}",
    )

    return fixed_df


@handle_errors(default_return=False, context="create_pickle_from_csv")
def create_pickle_from_csv(csv_path: str, output_path: str, lookback_days: int=730) -> bool:
    """Create a pickle file from a consolidated CSV file."""
    print(f"\nProcessing: {csv_path}")

    # Load CSV file
    df=pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Timestamp handling
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df=df.dropna(subset=["timestamp"])  # remove unparsable timestamps
        df.set_index("timestamp", inplace=True)

    # Check for price corruption
    if detect_price_corruption(df):
        print("  Detected corrupted prices, fixing...")
        df=fix_corrupted_prices(df)
    else:
        print("  Prices appear to be valid")

    # Filter by lookback period
    if not df.empty and isinstance(df.index, pd.DatetimeIndex):
        cutoff_date=datetime.now() - timedelta(days=lookback_days)
        df=df[df.index > cutoff_date]
        print(f"  Filtered to {len(df)} rows for {lookback_days} days")

    # Create data structure for pickle
    data: dict[str, Any] = {
        "klines": df,
        "agg_trades": pd.DataFrame(),  # Empty for now
        "futures": pd.DataFrame(),     # Empty for now
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


@handle_errors(default_return=False, context="regenerate_pickle_main")
def main() -> bool:
    """Main function to regenerate pickle files."""
    print("Regenerating Pickle Files from Consolidated CSV")
    print("=" * 60)

    data_cache_dir="data_cache"
    if not os.path.exists(data_cache_dir):
        print(missing(f"Data cache directory not found: {data_cache_dir}"))
        return False

    # Find consolidated CSV files
    consolidated_files: list[Path] = []
    for pattern in ["*consolidated*.csv"]:
        consolidated_files.extend(Path(data_cache_dir).glob(pattern))

    if not consolidated_files:
        print(warning(f"No consolidated CSV files found in {data_cache_dir}"))
        return False

    print(f"Found {len(consolidated_files)} consolidated CSV files")

    # Process each consolidated file
    success_count=0
    for csv_file in consolidated_files:
        csv_name = csv_file.stem

        # Create different lookback periods
        lookback_periods = [30, 60, 730]  # 30 days, 60 days, 2 years

        for lookback_days in lookback_periods:
            # Create output filename
            if "klines" in csv_name:
                symbol="ETHUSDT"
                timeframe = "1h"  # Map to an hourly timeframe for cached pickle
                pkl_name = f"{symbol}_{timeframe}_{lookback_days}_cached_data.pkl"
            else:
                # For other file types, use the original name
                pkl_name=f"{csv_name}_{lookback_days}_cached_data.pkl"

            pkl_path = os.path.join(data_cache_dir, pkl_name)

            if create_pickle_from_csv(str(csv_file), pkl_path, lookback_days):
                success_count += 1

    print(f"\nSuccessfully created {success_count} pickle files")

    # List the created files
    pkl_files=list(Path(data_cache_dir).glob("*_cached_data.pkl"))
    if pkl_files:
        print("\nCreated pickle files:")
        for pkl_file in sorted(pkl_files):
            try:
                with open(pkl_file, "rb") as f:
                    data=pickle.load(f)
                if isinstance(data, dict) and "klines" in data and isinstance(data["klines"], pd.DataFrame):
                    df=data["klines"]
                    print(f"  ✅ {pkl_file.name}: {len(df)} rows")
                else:
                    print(f"  ⚠️  {pkl_file.name}: Invalid data structure")
            except Exception as e:
                print(f"  ❌ {pkl_file.name}: Error reading file - {e}")

    return True


if __name__== "__main__":
    success = main()
    sys.exit(0 if success else 1)
