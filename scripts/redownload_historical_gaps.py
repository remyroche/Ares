
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from exchanges.binance.klines_adapter import BinanceKlinesAdapter
from src.utils.tprint import tprint_info, tprint_success, tprint_error
from scripts.detect_historical_gaps import detect_historical_gaps

async def redownload_missing_data(symbol="ETHUSDT"):
    # 1. Detect gaps
    gap_ranges = detect_historical_gaps(symbol=symbol, timeframe="15m")
    if not gap_ranges:
        tprint_success("✅ No gaps to download.")
        return

    # Use BinanceKlinesAdapter for public REST fallback
    adapter = BinanceKlinesAdapter()
    base_path = Path(f"historical_data/binance/{symbol.lower()}/raw")
    base_path.mkdir(parents=True, exist_ok=True)

    tprint_info(f"🚀 Starting re-download of {len(gap_ranges)} gap ranges for {symbol}...")

    for start_ts, end_ts in gap_ranges:
        # Buffer the range slightly to ensure overlap
        fetch_start = start_ts - timedelta(minutes=15)
        fetch_end = end_ts + timedelta(minutes=15)
        
        tprint_info(f"📥 Fetching: {fetch_start} to {fetch_end}")
        
        try:
            # Download 1m data using adapter which handles public/auth logic
            df = await adapter.get_klines_data(
                symbol=symbol,
                interval="1m",
                start_time=fetch_start,
                end_time=fetch_end,
                limit=1000
            )
            
            if df is not None and not df.empty:
                # Save to a temporary recovery file
                timestamp_str = fetch_start.strftime("%Y%m%d_%H%M%S")
                filename = base_path / f"recovery_{symbol.lower()}_1m_{timestamp_str}.parquet"
                df.to_parquet(filename)
                tprint_success(f"💾 Saved {len(df)} samples to {filename}")
            else:
                tprint_error(f"⚠️ No data returned for range {fetch_start} to {fetch_end}")
                
        except Exception as e:
            tprint_error(f"❌ Failed to download range {fetch_start}-{fetch_end}: {e}")
        
        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(redownload_missing_data())
