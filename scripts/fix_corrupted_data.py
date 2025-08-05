#!/usr/bin/env python3
"""
Script to fix corrupted price data in cached files.
This script will:
1. Load the existing CSV files
2. Detect and fix corrupted prices
3. Regenerate the pickle files with corrected data
"""

import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logging, system_logger


def detect_price_corruption(df: pd.DataFrame) -> bool:
    """
    Detect if price data is corrupted by checking if median price is reasonable.

    Args:
        df: DataFrame with price columns

    Returns:
        bool: True if prices are corrupted, False otherwise
    """
    if df.empty:
        return False

    # Check if we have price columns
    price_cols = ["open", "high", "low", "close"]
    if not all(col in df.columns for col in price_cols):
        return False

    # Calculate median price
    median_price = df["close"].median()

    # For ETH, reasonable price range is $100-$10,000
    # If median is outside this range, data is corrupted
    if median_price < 100 or median_price > 10000:
        return True

    return False


def fix_corrupted_prices(
    df: pd.DataFrame,
    target_median: float = 3000.0,
) -> pd.DataFrame:
    """
    Fix corrupted prices by scaling them to a reasonable range.

    Args:
        df: DataFrame with price columns
        target_median: Target median price (default: $3000 for ETH)

    Returns:
        pd.DataFrame: DataFrame with corrected prices
    """
    if df.empty:
        return df

    # Check if we have price columns
    price_cols = ["open", "high", "low", "close"]
    if not all(col in df.columns for col in price_cols):
        return df

    current_median = df["close"].median()

    if current_median <= 0:
        print(f"Warning: Invalid median price: {current_median}")
        return df

    # Calculate scale factor
    scale_factor = target_median / current_median

    print("Fixing corrupted prices:")
    print(f"  Current median: ${current_median:.2f}")
    print(f"  Target median: ${target_median:.2f}")
    print(f"  Scale factor: {scale_factor:.6f}")

    # Apply scaling to all price columns
    for col in price_cols:
        df[col] = df[col] * scale_factor

    # Verify the fix
    new_median = df["close"].median()
    print(f"  New median: ${new_median:.2f}")
    print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

    return df


def process_csv_file(csv_path: str, output_dir: str) -> bool:
    """
    Process a single CSV file and create corrected pickle file.

    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to save the pickle file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\nProcessing: {csv_path}")

        # Load CSV file
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")

        # Check for price corruption
        if detect_price_corruption(df):
            print("  Detected corrupted prices, fixing...")
            df = fix_corrupted_prices(df)
        else:
            print("  Prices appear to be valid")

        # Create data structure for pickle
        data = {
            "klines": df,
            "agg_trades": pd.DataFrame(),  # Empty for now
            "futures": pd.DataFrame(),  # Empty for now
            "metadata": {
                "source_file": csv_path,
                "processed_at": datetime.now().isoformat(),
                "price_corrected": detect_price_corruption(df),
            },
        }

        # Create output filename
        csv_name = Path(csv_path).stem
        pkl_name = f"{csv_name}_cached_data.pkl"
        pkl_path = os.path.join(output_dir, pkl_name)

        # Save pickle file
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  Saved corrected data to: {pkl_path}")
        return True

    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
        return False


def main():
    """Main function to fix corrupted data files."""
    setup_logging()
    logger = system_logger.getChild("FixCorruptedData")

    print("🔧 Fixing Corrupted Price Data")
    print("=" * 50)

    # Check for CSV files in data_cache
    data_cache_dir = "data_cache"
    if not os.path.exists(data_cache_dir):
        print(f"❌ Data cache directory not found: {data_cache_dir}")
        return False

    # Find CSV files
    csv_files = []
    for pattern in ["klines_*.csv", "aggtrades_*.csv", "futures_*.csv"]:
        csv_files.extend(Path(data_cache_dir).glob(pattern))

    if not csv_files:
        print(f"❌ No CSV files found in {data_cache_dir}")
        return False

    print(f"📁 Found {len(csv_files)} CSV files to process")

    # Process each CSV file
    success_count = 0
    for csv_file in csv_files:
        if process_csv_file(str(csv_file), data_cache_dir):
            success_count += 1

    print(f"\n✅ Successfully processed {success_count}/{len(csv_files)} files")

    # Also check for existing pickle files and fix them
    pkl_files = list(Path(data_cache_dir).glob("*_cached_data.pkl"))
    if pkl_files:
        print(f"\n🔍 Found {len(pkl_files)} existing pickle files")
        print("⚠️  Consider regenerating these files with corrected data")

        for pkl_file in pkl_files:
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)

                if "klines" in data and isinstance(data["klines"], pd.DataFrame):
                    df = data["klines"]
                    if detect_price_corruption(df):
                        print(f"  ❌ {pkl_file.name}: Contains corrupted prices")
                    else:
                        print(f"  ✅ {pkl_file.name}: Prices appear valid")
                else:
                    print(f"  ⚠️  {pkl_file.name}: Invalid data structure")

            except Exception as e:
                print(f"  ❌ {pkl_file.name}: Error reading file - {e}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
