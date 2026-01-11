import pandas as pd
import sys
from pathlib import Path

def inspect_file(symbol):
    base_dir = Path("historical_data/binance") / symbol.lower() / "raw"
    files = sorted(list(base_dir.glob("*.parquet")))
    if not files:
        print(f"No files found for {symbol}")
        return

    # Check first and last file to see if schema changes over time
    for f in [files[-1]]:
        print(f"--- {symbol.upper()} ({f.name}) ---")
        try:
            df = pd.read_parquet(f)
            print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
        except Exception as e:
            print(f"Error: {e}")
    print("\n")

for s in ["ethusdt", "btcusdt", "linkusdt", "solusdt", "bnbusdt", "avaxusdt"]:
    inspect_file(s)
