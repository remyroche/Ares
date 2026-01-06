
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime, timedelta

def detect_historical_gaps(symbol="ETHUSDT", timeframe="15m", base_dir="historical_data/binance/ethusdt/raw"):
    print(f"🔍 Analyzing gaps for {symbol} in {base_dir}...")
    
    # 1. Collect all raw 1m files
    raw_files = list(Path(base_dir).glob("*.parquet"))
    if not raw_files:
        print("❌ No raw 1m parquet files found.")
        return
    
    # 2. Load all timestamps
    all_timestamps = []
    for f in sorted(raw_files):
        try:
            df = pd.read_parquet(f, columns=['timestamp'])
            all_timestamps.append(df['timestamp'])
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
            
    if not all_timestamps:
        return
        
    full_ts = pd.concat(all_timestamps).sort_values().unique()
    full_ts = pd.to_datetime(full_ts)
    
    # 3. Define the expected range
    start_date = full_ts.min()
    end_date = full_ts.max()
    print(f"📊 Range: {start_date} to {end_date}")
    
    # 4. Check 1m gaps first
    expected_1m = pd.date_range(start=start_date, end=end_date, freq='1min')
    missing_1m = expected_1m.difference(full_ts)
    
    print(f"🔴 Found {len(missing_1m)} missing 1m samples ({len(missing_1m)/len(expected_1m)*100:.2f}%)")
    
    # 5. Map to 15m gaps
    # A 15m bar is "missing" if any of its 1m components are missing, 
    # OR if the 15m boundary itself is missing.
    expected_15m = pd.date_range(start=start_date.floor('15min'), end=end_date.floor('15min'), freq='15min')
    
    # Check if we have the 15m points
    available_15m = full_ts[full_ts.isin(expected_15m)]
    missing_15m = expected_15m.difference(available_15m)
    
    print(f"🔴 Found {len(missing_15m)} missing 15m samples ({len(missing_15m)/len(expected_15m)*100:.2f}%)")
    
    # 6. Group into ranges for re-downloading
    if len(missing_15m) > 0:
        ranges = []
        if len(missing_15m) == 1:
            ranges.append((missing_15m[0], missing_15m[0]))
        else:
            start_range = missing_15m[0]
            for i in range(1, len(missing_15m)):
                if missing_15m[i] - missing_15m[i-1] > timedelta(minutes=15):
                    ranges.append((start_range, missing_15m[i-1]))
                    start_range = missing_15m[i]
            ranges.append((start_range, missing_15m[-1]))
            
        print("\n🚀 Gap Ranges identified for re-download:")
        for s, e in ranges:
            duration = e - s
            print(f"  - {s} to {e} ({duration})")
            
        return ranges
    else:
        print("✅ No 15m gaps detected in the existing raw data.")
        return []

if __name__ == "__main__":
    detect_historical_gaps()
