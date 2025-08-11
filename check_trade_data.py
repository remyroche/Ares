import pandas as pd
import glob

files = glob.glob(
    "data_cache/unified/binance/ETHUSDT/1m/exchange=BINANCE/symbol=ETHUSDT/timeframe=1m/**/*.parquet",
    recursive=True,
)
non_zero_count = 0

print(f"Checking {len(files)} total files...")

# Check files from 2025 specifically (where aggtrades data exists)
for file in files:
    if "year=2025" in file:
        df = pd.read_parquet(file)
        trade_volume_sum = df["trade_volume"].sum()
        if trade_volume_sum > 0:
            non_zero_count += 1
            print(
                f'File: {file.split("/")[-3:]} - Trade volume: {trade_volume_sum:.2f}'
            )
            print(
                f'  Sample data: {df[["timestamp", "trade_volume", "trade_count"]].head()}'
            )
            break  # Just show the first one

print(f"Found {non_zero_count} files with trade data in 2025")

# Also check a few specific 2025 files
print("\nChecking specific 2025 files:")
for file in files[:50]:
    if "year=2025" in file and "month=06" in file:
        df = pd.read_parquet(file)
        trade_volume_sum = df["trade_volume"].sum()
        print(f'File: {file.split("/")[-3:]} - Trade volume: {trade_volume_sum:.2f}')
        if trade_volume_sum > 0:
            print(f"  ✅ Found trade data!")
            break
