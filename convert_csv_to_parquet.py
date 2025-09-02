#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd


def convert_csv_to_parquet():
    print("🔄 Converting CSV klines to Parquet...")

    csv_path="data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.csv"
    parquet_path = "data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet"

    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False

    if os.path.exists(parquet_path):
        print(f"✅ Parquet file already exists: {parquet_path}")
        return True

    try:
        print(f"📖 Reading CSV: {csv_path}")
        df=pd.read_csv(csv_path)
        print(f"   📊 Loaded {len(df)} rows")

        # Convert timestamp to numeric if needed
        if df["timestamp"].dtype== "object":
            # Convert datetime strings to timestamps
            df["timestamp"] = (
                pd.to_datetime(df["timestamp"], utc=True).astype(np.int64) // 10**6
            )
            print("   🔄 Converted datetime strings to millisecond timestamps")

        print(f"💾 Writing Parquet: {parquet_path}")
        df.to_parquet(parquet_path, index=False)
        print("   ✅ Successfully converted to parquet")

        # Verify the file
        df_check=pd.read_parquet(parquet_path)
        print(f"   ✅ Verified: {len(df_check)} rows in parquet")

        return True

    except Exception as e:
        print(f"❌ Error converting CSV to parquet: {e}")
        return False


if __name__== "__main__":
    convert_csv_to_parquet()
