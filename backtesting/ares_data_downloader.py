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

def get_monthly_periods(years_back, start_from_date=None):
    """Generate start and end datetimes for each month in the lookback period, optionally starting from a specific date."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    
    # If start_from_date is provided, ensure we don't go earlier than that
    if start_from_date:
        start_date = max(start_date, start_from_date)

    periods = []
    current_start = datetime(start_date.year, start_date.month, 1)
    
    while current_start < end_date:
        next_month = current_start.month + 1
        next_year = current_start.year
        if next_month > 12:
            next_month = 1
            next_year += 1
        current_end = datetime(next_year, next_month, 1)
        # Ensure the period doesn't go past the end_date if the last chunk is partial
        periods.append((current_start, min(current_end, end_date)))
        current_start = current_end
    return periods

def get_daily_periods(years_back, start_from_date=None):
    """Generate start and end datetimes for each day in the lookback period, optionally starting from a specific date."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)

    # If start_from_date is provided, ensure we don't go earlier than that
    if start_from_date:
        start_date = max(start_date, start_from_date)
    
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

def download_klines_data(symbol, interval, final_filename):
    """Downloads k-line data incrementally, appending to existing file."""
    print(f"--- Downloading K-line Data for {symbol} ({interval}) ---")
    client = Client()
    
    existing_df = pd.DataFrame()
    last_timestamp = None
    if os.path.exists(final_filename) and os.path.getsize(final_filename) > 0:
        try:
            existing_df = pd.read_csv(final_filename, index_col='open_time', parse_dates=True)
            if not existing_df.empty:
                last_timestamp = existing_df.index.max()
                print(f"Existing K-line data found up to {last_timestamp}. Downloading new data from this point.")
            else:
                print("Existing K-line file is empty. Starting full download.")
        except Exception as e:
            print(f"Error reading existing K-line file: {e}. Re-downloading full history.")
            existing_df = pd.DataFrame() # Reset to empty if read fails

    # Calculate periods based on whether we're resuming or starting fresh
    start_from_dt = last_timestamp + timedelta(minutes=1) if last_timestamp else None # Start from next minute
    monthly_periods = get_monthly_periods(CONFIG['LOOKBACK_YEARS'], start_from_date=start_from_dt)

    new_klines_data = []
    for start_dt, end_dt in monthly_periods:
        print(f"Fetching k-lines for {start_dt.strftime('%Y-%m')} to {end_dt.strftime('%Y-%m')}...")
        klines = download_with_retry(
            lambda: client.get_historical_klines(symbol, interval, str(start_dt), str(end_dt)),
            f"k-lines for {start_dt.strftime('%Y-%m')}"
        )
        if klines:
            new_klines_data.extend(klines)

    if not new_klines_data:
        print("No new k-line data downloaded.")
        return

    df_new = pd.DataFrame(new_klines_data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 
                                                   'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df_new['open_time'] = pd.to_datetime(df_new['open_time'], unit='ms')
    df_new.set_index('open_time', inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df_new[numeric_cols] = df_new[numeric_cols].apply(pd.to_numeric, axis=1)
    
    # Combine existing and new data, remove duplicates, and sort
    combined_df = pd.concat([existing_df, df_new]).drop_duplicates(subset=['close_time']).sort_index()
    combined_df.to_csv(final_filename)
    print(f"Updated k-line data saved to '{final_filename}'. Total records: {len(combined_df)}\n")

def download_agg_trades_data(symbol, final_filename):
    """Downloads aggregated trade data incrementally, appending to existing file."""
    print(f"--- Downloading Aggregated Trades Data for {symbol} ---")
    client = Client()

    existing_df = pd.DataFrame()
    last_timestamp = None
    if os.path.exists(final_filename) and os.path.getsize(final_filename) > 0:
        try:
            existing_df = pd.read_csv(final_filename, index_col='timestamp', parse_dates=True)
            if not existing_df.empty:
                last_timestamp = existing_df.index.max()
                print(f"Existing Agg Trades data found up to {last_timestamp}. Downloading new data from this point.")
            else:
                print("Existing Agg Trades file is empty. Starting full download.")
        except Exception as e:
            print(f"Error reading existing Agg Trades file: {e}. Re-downloading full history.")
            existing_df = pd.DataFrame() # Reset to empty if read fails

    # Calculate periods based on whether we're resuming or starting fresh
    start_from_dt = last_timestamp + timedelta(milliseconds=1) if last_timestamp else None
    daily_periods = get_daily_periods(CONFIG['LOOKBACK_YEARS'], start_from_date=start_from_dt)
    
    all_new_trades = []
    for start_dt, end_dt in daily_periods:
        print(f"Fetching agg trades for {start_dt.strftime('%Y-%m-%d')}...")
        
        current_start_time = int(start_dt.timestamp() * 1000)
        end_time_ms = int(end_dt.timestamp() * 1000)

        # If resuming, adjust the initial current_start_time for the very first fetch
        if last_timestamp and start_dt.date() == last_timestamp.date():
            current_start_time = int(last_timestamp.timestamp() * 1000) + 1 # Start from 1ms after last recorded trade

        trades_for_day = []
        while current_start_time < end_time_ms:
            trades = download_with_retry(
                lambda: client.get_aggregate_trades(symbol=symbol, startTime=current_start_time, endTime=end_time_ms, limit=1000),
                f"agg trades from {datetime.fromtimestamp(current_start_time/1000)}"
            )
            
            if not trades:
                break # No more trades in this period or API limit reached
            
            trades_for_day.extend(trades)
            # Set the start time for the next fetch to be after the last trade received
            current_start_time = trades[-1]['T'] + 1
        
        if trades_for_day: 
            df_day = pd.DataFrame(trades_for_day)
            df_day.drop_duplicates(subset=['a'], inplace=True) # Drop duplicates by aggregate trade ID
            all_new_trades.extend(df_day.to_dict('records')) # Append as dicts to avoid concat issues

    if not all_new_trades:
        print("No new aggregated trade data downloaded.")
        return

    df_new = pd.DataFrame(all_new_trades)
    df_new['timestamp'] = pd.to_datetime(df_new['T'], unit='ms')
    df_new.set_index('timestamp', inplace=True)
    df_new.rename(columns={'p': 'price', 'q': 'quantity', 'm': 'is_buyer_maker'}, inplace=True)
    df_new[['price', 'quantity']] = df_new[['price', 'quantity']].apply(pd.to_numeric)
    df_cleaned_new = df_new[['price', 'quantity', 'is_buyer_maker']]
    
    # Combine existing and new data, remove duplicates, and sort
    combined_df = pd.concat([existing_df, df_cleaned_new]).drop_duplicates(subset=['T']).sort_index() # Use 'T' for aggTrade ID
    combined_df.to_csv(final_filename)
    print(f"Updated agg trades data saved to '{final_filename}'. Total records: {len(combined_df)}\n")

def download_futures_data(symbol, final_filename):
    """Downloads futures-specific data like funding rates and open interest incrementally."""
    print(f"--- Downloading Futures Data for {symbol} ---")
    client = Client()

    existing_funding_df = pd.DataFrame()
    existing_oi_df = pd.DataFrame()
    last_funding_timestamp = None
    last_oi_timestamp = None

    # Load existing data if available
    if os.path.exists(final_filename) and os.path.getsize(final_filename) > 0:
        try:
            full_existing_df = pd.read_csv(final_filename, index_col='timestamp', parse_dates=True)
            if not full_existing_df.empty:
                if 'fundingRate' in full_existing_df.columns and not full_existing_df['fundingRate'].isnull().all():
                    existing_funding_df = full_existing_df[['fundingRate']].dropna()
                    if not existing_funding_df.empty:
                        last_funding_timestamp = existing_funding_df.index.max()
                if 'openInterest' in full_existing_df.columns and not full_existing_df['openInterest'].isnull().all():
                    existing_oi_df = full_existing_df[['openInterest']].dropna()
                    if not existing_oi_df.empty:
                        last_oi_timestamp = existing_oi_df.index.max()
                print(f"Existing Futures data found. Funding up to {last_funding_timestamp}, OI up to {last_oi_timestamp}.")
            else:
                print("Existing Futures file is empty. Starting full download.")
        except Exception as e:
            print(f"Error reading existing Futures file: {e}. Re-downloading full history.")
            existing_funding_df = pd.DataFrame()
            existing_oi_df = pd.DataFrame()

    # Determine start times for new downloads
    # Funding rate history is typically limited to 3 months (90 days) by Binance API
    # Open interest history is also limited (e.g., 500 points for 5m period)
    # So, we always try to fetch from a recent past, not necessarily last_timestamp.
    # We'll rely on drop_duplicates to handle overlaps.
    
    # For funding rates, Binance API typically allows max 3 months back.
    # For open interest, it's usually around 500-1000 points depending on period.
    # Let's fetch from 90 days back and rely on deduplication.
    start_ts_funding = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
    start_ts_oi = int((datetime.now() - timedelta(days=90)).timestamp() * 1000) # Same for simplicity

    # Fetch new funding rates
    new_funding_rates = download_with_retry(
        lambda: client.futures_funding_rate(symbol=symbol, startTime=start_ts_funding, limit=1000),
        "funding rate history"
    )
    funding_df_new = pd.DataFrame()
    if new_funding_rates:
        funding_df_new = pd.DataFrame(new_funding_rates)
        funding_df_new['timestamp'] = pd.to_datetime(funding_df_new['fundingTime'], unit='ms')
        funding_df_new.set_index('timestamp', inplace=True)
        funding_df_new['fundingRate'] = pd.to_numeric(funding_df_new['fundingRate'])
        funding_df_new = funding_df_new[['fundingRate']] # Keep only relevant column

    # Fetch new open interest
    new_open_interest = download_with_retry(
        lambda: client.futures_open_interest_hist(symbol=symbol, period='5m', limit=1000, startTime=start_ts_oi), # Increased limit
        "open interest history"
    )
    oi_df_new = pd.DataFrame()
    if new_open_interest:
        oi_df_new = pd.DataFrame(new_open_interest)
        oi_df_new['timestamp'] = pd.to_datetime(oi_df_new['timestamp'], unit='ms')
        oi_df_new.set_index('timestamp', inplace=True)
        oi_df_new['openInterest'] = pd.to_numeric(oi_df_new['sumOpenInterest'])
        oi_df_new = oi_df_new[['openInterest']] # Keep only relevant column
    
    # Combine existing and new data
    combined_funding_df = pd.concat([existing_funding_df, funding_df_new]).drop_duplicates().sort_index()
    combined_oi_df = pd.concat([existing_oi_df, oi_df_new]).drop_duplicates().sort_index()

    if not combined_funding_df.empty or not combined_oi_df.empty:
        # Merge the two combined dataframes, forward-filling to align timestamps
        futures_df = pd.concat([combined_funding_df.get('fundingRate'), combined_oi_df.get('openInterest')], axis=1)
        futures_df = futures_df.ffill().bfill() # Forward and backward fill to handle NaNs from merging
        
        # Save only if there's actual data
        if not futures_df.empty:
            futures_df.to_csv(final_filename)
            print(f"Futures data download complete and saved to {final_filename}. Total records: {len(futures_df)}\n")
        else:
            print("No futures data to save after combining.")
    else:
        print("Could not retrieve any futures data to combine.")

def main():
    """Main function to download all necessary data."""
    print("--- Ares Data Downloader ---")
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Access config values through the CONFIG dictionary
    symbol = CONFIG['SYMBOL']
    interval = CONFIG['INTERVAL']
    klines_filename = CONFIG['KLINES_FILENAME']
    agg_trades_filename = CONFIG['AGG_TRADES_FILENAME']
    futures_filename = CONFIG['FUTURES_FILENAME']

    download_klines_data(symbol, interval, klines_filename)
    download_agg_trades_data(symbol, agg_trades_filename)
    download_futures_data(symbol, futures_filename)
    
    print("\nAll data has been downloaded and cached locally.")

if __name__ == "__main__":
    main()
