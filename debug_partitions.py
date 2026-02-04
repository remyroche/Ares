import pandas as pd
import os
import glob

def check_partition(symbol, year, month):
    path = f"data/ohlcv/symbol={symbol}/year={year}/month={month}"
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return

    files = sorted(glob.glob(f"{path}/*.parquet"))
    print(f"Analyzing {symbol} {year}-{month} ({len(files)} files)")
    
    total_rows = 0
    for f in files:
        df = pd.read_parquet(f)
        count = len(df)
        total_rows += count
        
        ts_min = df["ts"].min()
        ts_max = df["ts"].max()
        
        print(f"File: {os.path.basename(f)}")
        print(f"  Rows: {count}")
        print(f"  Start: {ts_min}")
        print(f"  End:   {ts_max}")
        print(f"  Duration: {ts_max - ts_min}")
        print("-" * 20)

print("Checking 1MBABYDOGE_USDT for Jan 2026...")
check_partition("1MBABYDOGE_USDT", "2026", "01")
