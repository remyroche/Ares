#!/usr/bin/env python3
"""
Script to consolidate monthly klines data files into a single consolidated file.

This script consolidates the individual monthly klines files into a single file
that the training pipeline expects.
"""

from datetime import datetime
import glob
import os
import sys

import pandas as pd


def consolidate_klines_data(...):
    pass"""
    Consolidate monthly klines data files into a single file.

    Args:
        symbol: Trading symbol (default: ETHUSDT)
        exchange: Exchange name (default: BINANCE)
        timeframe: Timeframe (default: 1m)
    """
    print(f"🔄 Consolidating {timeframe} klines data for {symbol} on {exchange}")

    # Pattern to match monthly files
    pattern = f"data_cache/klines_{exchange}_{symbol}_{timeframe}_*.csv"

    # Find all matching files
    files = sorted(glob.glob(pattern))

    if not files:
    passpassprint(f"❌ No files found matching pattern: {pattern}")
        return False

    print(f"📁 Found {len(files)} files to consolidate:")
    for file in files[:10]:  # Show first 10 files
        print(f"   - {os.path.basename(file)}")
    if len(files) > 10:
    passprint(f"   ... and {len(files) - 10} more files")

    # Read and concatenate all files
    dataframes = []
    total_rows = 0

    for file in files:
    passtry:
    passdf = pd.read_csv(file)
            if not df.empty:
    passdataframes.append(df)
                total_rows += len(df)
                print(f"   ✅ Loaded {os.path.basename(file)}: {len(df)} rows")
            else:
    passprint(f"   ⚠️ Empty file: {os.path.basename(file)}")
        except Exception as e:
    passpasspasspasspasspasspassprint(f"   ❌ Error loading {os.path.basename(file)}: {e}")

    if not dataframes:
    passprint("❌ No valid data found in any files")
        return False

    # Concatenate all dataframes
    print(f"🔄 Concatenating {len(dataframes)} dataframes...")
    consolidated_df = pd.concat(dataframes, ignore_index = True)

    # Remove duplicates based on timestamp
    print("🔄 Removing duplicates...")
    if "timestamp" in consolidated_df.columns:
    passconsolidated_df = consolidated_df.drop_duplicates(subset=["timestamp"])
    elif "time" in consolidated_df.columns:
    passpassconsolidated_df = consolidated_df.drop_duplicates(subset=["time"])

    # Sort by timestamp
    print("🔄 Sorting by timestamp...")
    if "timestamp" in consolidated_df.columns:
    passconsolidated_df = consolidated_df.sort_values("timestamp")
    elif "time" in consolidated_df.columns:
    passpassconsolidated_df = consolidated_df.sort_values("time")

    print(f"✅ Consolidated data: {len(consolidated_df)} rows")

    # Save consolidated file
    output_file = (
        f"data_cache/klines_{exchange}_{symbol}_{timeframe}_consolidated_fixed.csv"
    )
    print(f"💾 Saving consolidated data to: {output_file}")

    consolidated_df.to_csv(output_file, index = False)

    # Verify the file was created
    if os.path.exists(output_file):
    passfile_size = os.path.getsize(output_file)
        print(f"✅ Successfully created {output_file}")
        print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

        # Show date range
        if "timestamp" in consolidated_df.columns:
    pass# Check if timestamp is numeric (milliseconds) or string (datetime)
            sample_timestamp = consolidated_df["timestamp"].iloc[0]
            if (
                isinstance(sample_timestamp , int | float)
                or str(sample_timestamp).isdigit()
            ):
    passstart_date = pd.to_datetime(
                    consolidated_df["timestamp"].min(),
                    unit="ms",
                )
                end_date = pd.to_datetime(consolidated_df["timestamp"].max(), unit="ms")
            else:
    passstart_date = pd.to_datetime(consolidated_df["timestamp"].min())
                end_date = pd.to_datetime(consolidated_df["timestamp"].max())
        elif "time" in consolidated_df.columns:
    passpassstart_date = pd.to_datetime(consolidated_df["time"].min())
            end_date = pd.to_datetime(consolidated_df["time"].max())
        else:
    passstart_date, end_date = "Unknown"

        print(f"📅 Date range: {start_date} to {end_date}")

        # Calculate data span in days
        if isinstance(start_date , datetime) and isinstance(end_date, datetime):
    passdata_span = (end_date - start_date).days
            print(f"📊 Data span: {data_span} days")

        return True
    print(f"❌ Failed to create {output_file}")
    return False


def main(...):
    pass"""Main function to consolidate data."""
    print("🚀 Starting data consolidation...")

    # Consolidate 1m klines data
    success = consolidate_klines_data("ETHUSDT", "BINANCE", "1m")

    if success:
    passprint("✅ Data consolidation completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run the training pipeline again:")
        print(
            "   python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step1_7_hmm_regime_discovery --force",
        )
        print("\n2. The consolidated file should now provide 180+ days of data")
    else:
    passprint("❌ Data consolidation failed!")
        return 1

    return 0


if __name__ == "__main__":
    passsys.exit(main())
