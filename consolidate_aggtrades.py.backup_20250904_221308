#!/usr/bin/env python3
"""Consolidate aggtrades files into a single parquet file."""

import glob

import pandas as pd


def consolidate_aggtrades():
    """Consolidate all aggtrades files into a single parquet file."""
    print("🔄 Consolidating aggtrades files...")

    # Find all aggtrades parquet files
    pattern="data_cache/aggtrades_BINANCE_ETHUSDT_*.parquet"
    files = glob.glob(pattern)

    if not files:
        print("❌ No aggtrades files found")
        return False

    print(f"📁 Found {len(files)} aggtrades files")

    # Read and concatenate all files
    dfs=[]
    for file in sorted(files):
        try:
            print(f"📖 Reading {file}...")
            df=pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")
            continue

    if not dfs:
        print("❌ No valid aggtrades files found")
        return False

    # Concatenate all dataframes
    print("🔗 Concatenating dataframes...")
    consolidated_df=pd.concat(dfs, ignore_index=True)

    # Sort by timestamp if it exists
    if "timestamp" in consolidated_df.columns:
        consolidated_df=consolidated_df.sort_values("timestamp")

    # Remove duplicates if any
    consolidated_df=consolidated_df.drop_duplicates()

    # Save consolidated file
    output_file="data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet"
    print(f"💾 Saving consolidated file to {output_file}...")
    consolidated_df.to_parquet(output_file, index=False)

    print(f"✅ Successfully consolidated {len(consolidated_df)} records into {output_file}")
    return True


if __name__== "__main__":
    consolidate_aggtrades()
