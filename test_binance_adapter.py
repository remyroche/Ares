#!/usr/bin/env python3
"""Quick test of Binance Klines Adapter"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from exchanges.binance.klines_adapter import BinanceKlinesAdapter


async def main():
    print("🧪 Testing Binance Klines Adapter...")
    
    # Initialize adapter
    adapter = BinanceKlinesAdapter(api_key=None, secret_key=None)
    print("✅ Adapter initialized")
    
    # Test fetch
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    print(f"📥 Fetching 1 hour of ETHUSDT data...")
    df = await adapter.get_klines_data(
        symbol="ETHUSDT",
        interval="1m",
        start_time=start_time,
        end_time=end_time,
        limit=100
    )
    
    if df is not None and len(df) > 0:
        print(f"✅ Success! Received {len(df)} candles")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        print(f"   First 3 rows:")
        print(df.head(3))
        return 0
    else:
        print(f"❌ No data received")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

