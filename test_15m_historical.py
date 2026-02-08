"""
Test 15m download with historical data (should get full 48 bars).
"""

import pandas as pd
import ccxt
from extreme_price_movements.hf_data_loader import get_15m_ohlcv

print("=" * 60)
print("Testing 15m OHLCV Download - Historical Data")
print("=" * 60)

exchange = ccxt.binance()
symbol = "BTC/USDT"

# Use historical timestamp (yesterday) to ensure 12 hours of data exists
entry_ts = pd.Timestamp("2026-02-06 00:00:00", tz="UTC")

print(f"\nDownloading 15m data for {symbol} starting at {entry_ts}")
print(f"Requesting 12 hours = 48 bars")

df_15m = get_15m_ohlcv(exchange, symbol, entry_ts, max_hold_hours=12)

if not df_15m.empty:
    print(f"\n✅ Successfully downloaded {len(df_15m)} bars")
    print(f"   Time range: {df_15m.index.min()} to {df_15m.index.max()}")
    
    # Calculate expected vs actual
    time_span = (df_15m.index.max() - df_15m.index.min()).total_seconds() / 3600
    expected_bars = 48
    
    print(f"\n   Expected: {expected_bars} bars (12 hours)")
    print(f"   Actual: {len(df_15m)} bars ({time_span:.1f} hours)")
    
    if len(df_15m) == expected_bars:
        print(f"   ✅ PERFECT: Got exactly 48 bars!")
    elif len(df_15m) >= 47:  # Allow for minor timing differences
        print(f"   ✅ GOOD: Got {len(df_15m)} bars (close to 48)")
    else:
        print(f"   ⚠️  WARNING: Expected 48 bars, got {len(df_15m)}")
else:
    print("❌ Failed to download data")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
