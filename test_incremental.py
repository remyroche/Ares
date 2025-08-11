#!/usr/bin/env python3

import os
import glob
import pandas as pd
from datetime import datetime, timedelta, timezone, date


def test_incremental_logic():
    print("🔍 TESTING INCREMENTAL PROCESSING LOGIC")
    print("=" * 50)

    # Check existing unified data
    unified_base = "data_cache/unified/binance/ETHUSDT/1m"
    print(f"Unified base path: {unified_base}")
    print(f"Path exists: {os.path.exists(unified_base)}")

    if os.path.exists(unified_base):
        parquet_files = glob.glob(
            os.path.join(unified_base, "**/*.parquet"), recursive=True
        )
        print(f"Found {len(parquet_files)} parquet files")

        if parquet_files:
            # Find all existing dates in unified data
            unified_dates = set()
            for file_path in parquet_files:
                try:
                    # Extract date from path like: .../year=2025/month=07/day=15/...
                    path_parts = file_path.split("/")
                    for i, part in enumerate(path_parts):
                        if part.startswith("year="):
                            year = int(part.split("=")[1])
                            month = int(path_parts[i + 1].split("=")[1])
                            day = int(path_parts[i + 2].split("=")[1])
                            unified_dates.add(date(year, month, day))
                            break
                            break
                except Exception as e:
                    print(f"Error parsing date from {file_path}: {e}")

            if unified_dates:
                print(f"Found {len(unified_dates)} unified dates")
                print(f"Date range: {min(unified_dates)} to {max(unified_dates)}")

                # Check klines data
                klines_path = (
                    "data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet"
                )
                if os.path.exists(klines_path):
                    print(f"Klines file exists: {klines_path}")
                    try:
                        klines_df = pd.read_parquet(klines_path)
                        # Convert timestamps to dates
                        klines_df["date"] = pd.to_datetime(
                            klines_df["timestamp"], unit="ms", utc=True
                        ).dt.date
                        klines_dates = set(klines_df["date"].unique())

                        print(f"Found {len(klines_dates)} klines dates")
                        print(
                            f"Klines date range: {min(klines_dates)} to {max(klines_dates)}"
                        )

                        # Find missing dates
                        missing_dates = klines_dates - unified_dates
                        missing_dates = sorted(missing_dates)

                        if missing_dates:
                            print(f"✅ MISSING DATES FOUND: {len(missing_dates)} dates")
                            print(
                                f"Missing dates: {missing_dates[:10]}{'...' if len(missing_dates) > 10 else ''}"
                            )
                            print(
                                f"Should process from: {min(missing_dates)} to {max(missing_dates)}"
                            )
                        else:
                            print(
                                "✅ No missing dates found - unified dataset is complete"
                            )
                    except Exception as e:
                        print(f"Error reading klines: {e}")
                else:
                    print(f"❌ Klines file not found: {klines_path}")
            else:
                print("❌ Could not determine existing unified dates")
    else:
        print("❌ No existing unified data found")


if __name__ == "__main__":
    test_incremental_logic()
