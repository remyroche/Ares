import glob

import pandas as pd

files = glob.glob(
    "data_cache/unified/binance/ETHUSDT/1m/exchange=BINANCE/symbol=ETHUSDT/timeframe=1m/**/*.parquet",
    recursive=True,
)
non_zero_count = 0


# Check files from 2025 specifically (where aggtrades data exists)
for file in files:
    pass
    pass
    if "year=2025" in file:
    pass
    pass
        df = pd.read_parquet(file)
        trade_volume_sum = df["trade_volume"].sum()
        if trade_volume_sum > 0:
    pass
    pass
            non_zero_count += 1
            break  # Just show the first one


# Also check a few specific 2025 files
for file in files[:50]:
    pass
    pass
    if "year=2025" in file and "month=06" in file:
    pass
    pass
        df = pd.read_parquet(file)
        trade_volume_sum = df["trade_volume"].sum()
        if trade_volume_sum > 0:
    pass
    pass
            break
