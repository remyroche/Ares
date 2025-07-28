import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import os
import time
# Import the main CONFIG dictionary
from config import CONFIG
from ares_mailer import send_email

# --- Configuration ---
CACHE_DIR = "data_cache" # Directory to store monthly chunks
MAX_RETRIES = 3 # Number of times to retry a failed API call
RETRY_DELAY_SECONDS = 5 # Seconds to wait between retries

def get_monthly_periods(years_back):
    """Generate start and end datetimes for each month in the lookback period."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    
    periods = []
    current_start = datetime(start_date.year, start_date.month, 1)
    
    while current_start < end_date:
        next_month = current_start.month + 1
        next_year = current_start.year
        if next_month > 12:
            next_month = 1
            next_year += 1
        current_end = datetime(next_year, next_month, 1)
        periods.append((current_start, current_end))
        current_start = current_end
    return periods

def get_daily_periods(years_back):
    """Generate start and end datetimes for each day in the lookback period."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    
    periods = []
    current_day = start_date.date()
    
    while current_day < end_date.date():
        start_dt = datetime.combine(current_day, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        periods.append((start_dt, end_dt))
        current_day += timedelta(days=1)
    return periods

def download_with_retry(api_call, description):
    """Wrapper to retry an API call on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            result = api_call()
            return result
        except Exception as e:
            print(f"An error occurred while fetching {description}: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"Failed to fetch {description} after {MAX_RETRIES} attempts.")
                return None

def download_klines_data(symbol, interval, periods, final_filename):
    """Downloads k-line data in monthly chunks, skipping existing non-empty files."""
    print(f"--- Downloading K-line Data ---")
    all_monthly_files = []
    client = Client()

    for start_dt, end_dt in periods:
        month_str = start_dt.strftime('%Y-%m')
        chunk_filename = os.path.join(CACHE_DIR, f"klines_{symbol}_{interval}_{month_str}.csv")
        all_monthly_files.append(chunk_filename)

        # CORRECTED: Check if a non-empty file already exists before fetching.
        if os.path.exists(chunk_filename) and os.path.getsize(chunk_filename) > 0:
            print(f"K-line chunk for {month_str} already exists. Skipping.")
            continue

        print(f"Fetching k-lines for {month_str}...")
        klines = download_with_retry(
            lambda: client.get_historical_klines(symbol, interval, str(start_dt), str(end_dt)),
            f"k-lines for {month_str}"
        )

        if not klines: continue

        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 
                                           'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        df.to_csv(chunk_filename)
        print(f"Saved chunk to '{chunk_filename}'")
    
    print("\nCombining all k-line chunks into final file...")
    all_dfs = [pd.read_csv(f, index_col='open_time', parse_dates=True) for f in all_monthly_files if os.path.exists(f)]
    if all_dfs:
        combined_df = pd.concat(all_dfs).sort_index()
        combined_df.to_csv(final_filename)
        print(f"Final k-line data saved to '{final_filename}'.\n")

def download_agg_trades_data(symbol, daily_periods, final_filename):
    """Downloads aggregated trade data in daily chunks, skipping existing non-empty files."""
    print(f"--- Downloading Aggregated Trades Data (Daily Chunks) ---")
    all_daily_files = []
    client = Client()

    for start_dt, end_dt in daily_periods:
        day_str = start_dt.strftime('%Y-%m-%d')
        chunk_filename = os.path.join(CACHE_DIR, f"aggtrades_{symbol}_{day_str}.csv")
        all_daily_files.append(chunk_filename)
        
        # CORRECTED: Check if a non-empty file already exists before fetching.
        if os.path.exists(chunk_filename) and os.path.getsize(chunk_filename) > 0:
            print(f"Agg trades chunk for {day_str} already exists. Skipping.")
            continue
        
        print(f"Fetching all agg trades for {day_str}...")
        
        all_trades_for_day = []
        current_start_time = int(start_dt.timestamp() * 1000)
        end_time_ms = int(end_dt.timestamp() * 1000)

        while current_start_time < end_time_ms:
            trades = download_with_retry(
                lambda: client.get_aggregate_trades(symbol=symbol, startTime=current_start_time, endTime=end_time_ms, limit=1000),
                f"agg trades from {datetime.fromtimestamp(current_start_time/1000)}"
            )
            
            if not trades:
                break # No more trades in this period
            
            all_trades_for_day.extend(trades)
            # Set the start time for the next fetch to be after the last trade received
            current_start_time = trades[-1]['T'] + 1
        
        if not all_trades_for_day: 
            print(f"  No trades found for {day_str}.")
            continue

        df = pd.DataFrame(all_trades_for_day)
        df.drop_duplicates(subset=['a'], inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['T'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'p': 'price', 'q': 'quantity', 'm': 'is_buyer_maker'}, inplace=True)
        df[['price', 'quantity']] = df[['price', 'quantity']].apply(pd.to_numeric)
        df_cleaned = df[['price', 'quantity', 'is_buyer_maker']]
        df_cleaned.to_csv(chunk_filename)
        print(f"Saved chunk with {len(df_cleaned)} trades to '{chunk_filename}'")
            
    print("\nCombining all daily agg trade chunks into final file...")
    all_dfs = [pd.read_csv(f, index_col='timestamp', parse_dates=True) for f in all_daily_files if os.path.exists(f)]
    if all_dfs:
        combined_df = pd.concat(all_dfs).sort_index()
        combined_df.to_csv(final_filename)
        print(f"Final agg trades data saved to '{final_filename}'.\n")

def download_futures_data(symbol, years_back, filename):
    """Downloads futures-specific data like funding rates and open interest."""
    if os.path.exists(filename):
        print(f"Futures data file '{filename}' already exists. Skipping download.")
        return

    print(f"--- Downloading Futures Data ---")
    client = Client()
    start_ts = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
    
    funding_rates = download_with_retry(
        lambda: client.futures_funding_rate(symbol=symbol, startTime=start_ts, limit=1000),
        "funding rate history"
    )
    open_interest = download_with_retry(
        lambda: client.futures_open_interest_hist(symbol=symbol, period='5m', limit=500, startTime=start_ts),
        "open interest history"
    )

    funding_df = pd.DataFrame()
    oi_df = pd.DataFrame()
    if funding_rates:
        funding_df = pd.DataFrame(funding_rates)
        funding_df['timestamp'] = pd.to_datetime(funding_df['fundingTime'], unit='ms')
        funding_df.set_index('timestamp', inplace=True)
        funding_df['fundingRate'] = pd.to_numeric(funding_df['fundingRate'])

    if open_interest:
        oi_df = pd.DataFrame(open_interest)
        oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
        oi_df.set_index('timestamp', inplace=True)
        oi_df['openInterest'] = pd.to_numeric(oi_df['sumOpenInterest'])
    
    if not funding_df.empty or not oi_df.empty:
        # Merge funding_df and oi_df, forward-filling to align timestamps
        futures_df = pd.concat([funding_df.get('fundingRate'), oi_df.get('openInterest')], axis=1).ffill()
        futures_df.to_csv(filename)
        print(f"Futures data download complete and saved to {filename}.\n")
    else:
        print("Could not retrieve any futures data.")

def main():
    """Main function to download all necessary data."""
    print("--- Ares Data Downloader ---")
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Access config values through the CONFIG dictionary
    symbol = CONFIG['SYMBOL']
    interval = CONFIG['INTERVAL']
    lookback_years = CONFIG['LOOKBACK_YEARS']
    klines_filename = CONFIG['KLINES_FILENAME']
    agg_trades_filename = CONFIG['AGG_TRADES_FILENAME']
    futures_filename = CONFIG['FUTURES_FILENAME']

    # Generate monthly periods for klines and daily for agg trades
    monthly_periods = get_monthly_periods(lookback_years)
    daily_periods = get_daily_periods(lookback_years)

    download_klines_data(symbol, interval, monthly_periods, klines_filename)
    download_agg_trades_data(symbol, daily_periods, agg_trades_filename)
    download_futures_data(symbol, lookback_years, futures_filename)
    
    print("\nAll data has been downloaded and cached locally.")

if __name__ == "__main__":
    main()
