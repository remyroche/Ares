"""
Quick test script for 15m OHLCV data download and trailing profit simulation.
"""

import pandas as pd
import ccxt
from extreme_price_movements.hf_data_loader import get_15m_ohlcv

# Test 15m data download
print("=" * 60)
print("Testing 15m OHLCV Data Download")
print("=" * 60)

exchange = ccxt.binance()
symbol = "BTC/USDT"
entry_ts = pd.Timestamp("2026-02-07 12:00:00", tz="UTC")

print(f"\nDownloading 15m data for {symbol} starting at {entry_ts}")
df_15m = get_15m_ohlcv(exchange, symbol, entry_ts, max_hold_hours=12)

if not df_15m.empty:
    print(f"✅ Successfully downloaded {len(df_15m)} bars")
    print(f"   Time range: {df_15m.index.min()} to {df_15m.index.max()}")
    print(f"   Columns: {list(df_15m.columns)}")
    print(f"\nFirst 5 bars:")
    print(df_15m.head())
    print(f"\nData types:")
    print(df_15m.dtypes)
else:
    print("❌ Failed to download data")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
