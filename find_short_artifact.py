
import os
import pandas as pd
from pathlib import Path

def check_artifacts():
    print("Searching for artifacts with ETHUSDT data in Jan 2026...")
    artifacts_dir = Path("artifacts")
    
    for file_path in artifacts_dir.rglob("*.parquet"):
        try:
            # Skip likely irrelevant files to speed up
            if "events" in file_path.name or "metrics" in file_path.name or "predictions" in file_path.name:
                continue
                
            # Read parquet
            # We only need metadata really, but let's read index/cols
            # columns=['symbol', 'timestamp'] if possible or just read index
            try:
                df = pd.read_parquet(file_path)
            except:
                continue
                
            if df.empty:
                continue
                
            # Check for ETHUSDT
            is_eth = False
            if 'symbol' in df.columns and (df['symbol'] == 'ETHUSDT').any():
                is_eth = True
            elif 'ETHUSDT' in file_path.name:
                # Name might indicate it's ETH specific
                is_eth = True
                
            if not is_eth:
                continue
                
            # Check date range
            start_date = None
            end_date = None
            
            if isinstance(df.index, pd.DatetimeIndex):
                start_date = df.index.min()
                end_date = df.index.max()
            elif 'timestamp' in df.columns:
                start_date = pd.to_datetime(df['timestamp']).min()
                end_date = pd.to_datetime(df['timestamp']).max()
            elif 'open_time' in df.columns:
                start_date = pd.to_datetime(df['open_time']).min()
                end_date = pd.to_datetime(df['open_time']).max()
                
            if start_date is None:
                continue
                
            # Look for Jan 2026 overlap with ~10 days duration
            if str(start_date).startswith("2026-01"):
                duration_days = (end_date - start_date).days
                print(f"MATCH: {file_path}")
                print(f"  Range: {start_date} -> {end_date} ({duration_days} days)")
                print(f"  Rows: {len(df)}")
                
        except Exception as e:
            # print(f"Error reading {file_path}: {e}")
            pass

if __name__ == "__main__":
    check_artifacts()
