import asyncio
import pandas as pd
from datetime import datetime
from pathlib import Path
import ccxt.async_support as ccxt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def download_btc_recovery():
    symbol = "BTCUSDT"
    interval = "1m"
    start_date = datetime(2023, 4, 1)
    end_date = datetime(2025, 1, 1)
    
    data_dir = Path("historical_data/binance/btcusdt/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting recovery for {symbol} {interval} from {start_date} to {end_date}")

    exchange = ccxt.binance({'enableRateLimit': True})
    
    current_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    batch_limit = 1000
    month_data = []
    current_month = start_date.month
    current_year = start_date.year
    
    total_downloaded = 0

    try:
        while current_ts < end_ts:
            # Retry loop
            for attempt in range(5):
                try:
                    ohlcv = await exchange.fetch_ohlcv(
                        symbol.replace('USDT', '/USDT'), 
                        timeframe=interval, 
                        since=current_ts, 
                        limit=batch_limit
                    )
                    break
                except Exception as e:
                    print(f"⚠️ Error (attempt {attempt+1}/5): {e}")
                    await asyncio.sleep(2 * (attempt + 1))
            else:
                print("❌ Failed after 5 attempts. Exiting.")
                break

            if not ohlcv:
                print("⚠️ No data returned. Break.")
                break

            # Process
            for candle in ohlcv:
                record = {
                    'timestamp': pd.to_datetime(candle[0], unit='ms'),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]),
                }
                month_data.append(record)

            total_downloaded += len(ohlcv)
            
            last_ts = ohlcv[-1][0]
            if last_ts <= current_ts:
                current_ts += 60000 # Force advance 1m
            else:
                current_ts = last_ts + 1

            # Save check
            last_dt = datetime.fromtimestamp(last_ts / 1000)
            if last_dt.month != current_month or last_dt.year != current_year:
                if month_data:
                    save_month(month_data, symbol, interval, current_year, current_month, data_dir)
                    month_data = []
                current_month = last_dt.month
                current_year = last_dt.year
            
            if total_downloaded % 10000 == 0:
                print(f"📊 Progress: {last_dt} ({total_downloaded} records)")

            await asyncio.sleep(0.05)

        # Final save
        if month_data:
             save_month(month_data, symbol, interval, current_year, current_month, data_dir)

    finally:
        await exchange.close()
        print("✅ Recovery finished.")

def save_month(data, symbol, interval, year, month, data_dir):
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    df['symbol'] = symbol
    df['interval'] = interval
    df['exchange'] = 'binance'
    
    filename = f"{symbol.lower()}_{interval}_{year}_{month:02d}.parquet"
    path = data_dir / filename
    df.to_parquet(path)
    print(f"💾 Saved {len(df)} records to {filename}")

if __name__ == "__main__":
    asyncio.run(download_btc_recovery())
