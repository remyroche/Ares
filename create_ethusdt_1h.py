#!/usr/bin/env python3
"""
Script to convert downloaded data from data_cache to the expected format for regime training.
"""

import os

import pandas as pd


def create_ethusdt_1h_csv():
    """Convert downloaded klines data to the expected ETHUSDT_1h.csv format."""

    # Check if the consolidated klines file exists
    klines_file = "data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.csv"

    if not os.path.exists(klines_file):
        print(f"❌ Klines file not found: {klines_file}")
        return False

    print(f"📖 Reading klines data from: {klines_file}")

    try:
        # Read the consolidated klines data
        df = pd.read_csv(klines_file)
        print(f"📊 Loaded {len(df)} records")
        print(f"📋 Columns: {list(df.columns)}")

        # Ensure we have the required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False

        # Convert timestamp to datetime if it's not already
        if "timestamp" in df.columns:
            # Check if timestamp is already datetime
            if df["timestamp"].dtype == "object":
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                # Assume it's milliseconds
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Rename timestamp to open_time to match expected format
        df = df.rename(columns={"timestamp": "open_time"})

        # Sort by open_time
        df = df.sort_values("open_time").reset_index(drop=True)

        # Set open_time as index for resampling
        df.set_index("open_time", inplace=True)

        # Resample to 1-hour data for regime classification
        print("🔄 Resampling 1-minute data to 1-hour data...")
        df_1h = (
            df.resample("1H")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                },
            )
            .dropna()
        )

        print(f"📊 Original 1-minute data: {len(df)} records")
        print(f"📊 Resampled 1-hour data: {len(df_1h)} records")

        # Create the data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Save to the expected location
        output_file = "data/ETHUSDT_1h.csv"
        df_1h.to_csv(output_file)

        print(f"✅ Successfully created: {output_file}")
        print(f"📊 File contains {len(df_1h)} records")
        print(f"📅 Date range: {df_1h.index.min()} to {df_1h.index.max()}")

        return True

    except Exception as e:
        print(f"❌ Error creating ETHUSDT_1h.csv: {e}")
        return False


if __name__ == "__main__":
    success = create_ethusdt_1h_csv()
    if success:
        print("✅ Data conversion completed successfully")
    else:
        print("❌ Data conversion failed")
