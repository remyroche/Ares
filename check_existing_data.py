#!/usr/bin/env python3

from datetime import datetime, UTC
import os

import pandas as pd

def check_existing_data():
    pass
    pass
    unified_base = "data_cache/unified/binance/ETHUSDT/1m"
    print("🔍 CHECKING EXISTING UNIFIED DATA:")
    print(f"Path exists: {os.path.exists(unified_base)}")

    if os.path.exists(unified_base):
    pass
    pass
        parquet_files = []
        for root, _dirs, files in os.walk(unified_base):
    pass
    pass
            for file in files:
    pass
    pass
                if file.endswith(".parquet"):
    pass
    pass
                    parquet_files.append(os.path.join(root, file))

        print(f"Found {len(parquet_files)} parquet files")

        if parquet_files:
    pass
    pass
            print(f"Sample files: {parquet_files[:3]}")

            # Check the latest timestamp
            latest_ts = None
            for file_path in parquet_files[-5:]:  # Check last 5 files
                try:
                    df = pd.read_parquet(file_path)
    except Exception as e:
        pass
    except Exception as e:
        pass
                    if "timestamp" in df.columns:
    pass
    pass
                        file_latest = df["timestamp"].max()
                        if latest_ts is None or file_latest > latest_ts:
    pass
    pass
                            latest_ts = file_latest
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

            if latest_ts:
    pass
    pass
    pass  # TODO: Add proper implementation
                latest_date = datetime.fromtimestamp(
                    latest_ts / 1000,
                    tz, UTC = ).date()
                print(f"Latest timestamp: {latest_ts}")
                print(f"Latest date: {latest_date}")
            else:
                print("Could not determine latest timestamp")
    else:
        print("No existing unified data found")

if __name__ == "__main__":
    pass
    pass
    check_existing_data()
