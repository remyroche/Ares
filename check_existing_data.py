#!/usr/bin/env python3

import os
import glob
import pandas as pd


def check_existing_data():
    unified_base = "data_cache/unified/binance/ETHUSDT/1m"
    print("🔍 CHECKING EXISTING UNIFIED DATA:")
    print(f"Path exists: {os.path.exists(unified_base)}")

    if os.path.exists(unified_base):
        parquet_files = []
        for root, dirs, files in os.walk(unified_base):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))

        print(f"Found {len(parquet_files)} parquet files")

        if parquet_files:
            print(f"Sample files: {parquet_files[:3]}")

            # Check the latest timestamp
            latest_ts = None
            for file_path in parquet_files[-5:]:  # Check last 5 files
                try:
                    df = pd.read_parquet(file_path)
                    if "timestamp" in df.columns:
                        file_latest = df["timestamp"].max()
                        if latest_ts is None or file_latest > latest_ts:
                            latest_ts = file_latest
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

            if latest_ts:
                from datetime import datetime, timezone

                latest_date = datetime.fromtimestamp(
                    latest_ts / 1000, tz=timezone.utc
                ).date()
                print(f"Latest timestamp: {latest_ts}")
                print(f"Latest date: {latest_date}")
            else:
                print("Could not determine latest timestamp")
    else:
        print("No existing unified data found")


if __name__ == "__main__":
    check_existing_data()
