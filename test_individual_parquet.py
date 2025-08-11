#!/usr/bin/env python3
"""
Test script to verify individual parquet files loading works on Mac M1
"""

import os
import glob
import pandas as pd
from datetime import datetime


def load_individual_parquet_files(
    exchange: str, symbol: str, max_files: int = 10
) -> pd.DataFrame:
    """
    Load individual parquet files from the partitioned directory structure.
    This avoids the bus error issues with large consolidated files on Mac M1.
    """
    base_dir = f"data_cache/parquet/aggtrades_{exchange}_{symbol}"

    if not os.path.exists(base_dir):
        print(f"❌ Partitioned parquet directory not found: {base_dir}")
        return pd.DataFrame()

    # Find all parquet files in the directory structure
    pattern = f"{base_dir}/**/*.parquet"
    parquet_files = glob.glob(pattern, recursive=True)

    if not parquet_files:
        print(f"❌ No parquet files found in: {base_dir}")
        return pd.DataFrame()

    # Sort files by date (newest first)
    parquet_files.sort(reverse=True)

    # For blank mode, limit the number of files, but respect the max_files parameter
    if os.environ.get("BLANK_TRAINING_MODE", "0") == "1":
        # Use the provided max_files parameter instead of defaulting to 10
        parquet_files = parquet_files[:max_files]
        print(f"📊 Blank mode: Loading {len(parquet_files)} most recent files (max_files={max_files})")

    print(f"📁 Loading {len(parquet_files)} individual parquet files from: {base_dir}")

    # Load files one by one to avoid memory issues
    all_data = []
    total_rows = 0

    for i, file_path in enumerate(parquet_files):
        try:
            if i % 10 == 0:
                print(
                    f"📂 Loading file {i+1}/{len(parquet_files)}: {os.path.basename(file_path)}"
                )

            # Load individual file using pandas
            df = pd.read_parquet(file_path)

            if not df.empty:
                all_data.append(df)
                total_rows += len(df)
                print(f"✅ Loaded {len(df):,} rows from {os.path.basename(file_path)}")

        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            continue

    if not all_data:
        print("❌ No data loaded from any parquet files")
        return pd.DataFrame()

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    print(
        f"🎉 Successfully loaded {len(combined_df):,} total rows from {len(all_data)} files"
    )

    # Show data info
    if not combined_df.empty:
        print(f"📊 Data columns: {list(combined_df.columns)}")
        print(
            f"📅 Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}"
        )
        print(
            f"💰 Price range: {combined_df['price'].min():.2f} to {combined_df['price'].max():.2f}"
        )

    return combined_df


if __name__ == "__main__":
    print("🧪 Testing individual parquet files loading...")
    print("=" * 60)

    # Test with individual files
    print("📁 Testing individual parquet files...")
    df_individual = load_individual_parquet_files("BINANCE", "ETHUSDT", max_files=5)

    if not df_individual.empty:
        print("\n✅ Individual parquet files loading successful!")
        print(f"📊 Total rows: {len(df_individual):,}")

        # Test converting to OHLCV
        print("\n🔄 Testing OHLCV conversion...")
        try:
            # Convert to OHLCV (1h timeframe)
            df_ohlcv = (
                df_individual.set_index("timestamp")
                .resample("1H")
                .agg({"price": ["first", "max", "min", "last"], "quantity": "sum"})
                .dropna()
            )

            # Flatten column names
            df_ohlcv.columns = ["open", "high", "low", "close", "volume"]
            df_ohlcv = df_ohlcv.reset_index()

            print(f"✅ OHLCV conversion successful: {len(df_ohlcv)} records")
            print(
                f"📅 OHLCV date range: {df_ohlcv['timestamp'].min()} to {df_ohlcv['timestamp'].max()}"
            )

        except Exception as e:
            print(f"❌ OHLCV conversion failed: {e}")

    else:
        print("\n❌ Individual parquet files loading failed!")

    print("\n" + "=" * 60)
    print("🏁 Test completed!")
