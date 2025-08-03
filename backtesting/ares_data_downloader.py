# backtesting/ares_data_downloader.py

import asyncio
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

from src.utils.error_handler import (
    handle_file_operations,
    handle_network_operations,
)
from src.utils.logger import system_logger

# --- Configuration ---
CACHE_DIR = "data_cache"  # Directory to store data files
MAX_RETRIES = 3  # Number of times to retry a failed API call
RETRY_DELAY_SECONDS = 5  # Seconds to wait between retries

# Import required modules
try:
    import sys
    import os
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from exchange.factory import ExchangeFactory
    from src.config import CONFIG
    print("✅ Successfully imported ExchangeFactory and CONFIG")
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Please ensure all dependencies are installed and the project structure is correct.")
    sys.exit(1)


def get_monthly_periods(years_back, start_from_date=None):
    """Generate start and end datetimes for each month in the lookback period with optimized batches."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)

    if start_from_date:
        start_date = max(start_date, start_from_date)

    periods = []
    current_start = datetime(start_date.year, start_date.month, 1)

    # Use larger batches for better performance (3 months at a time)
    batch_size = 3
    batch_start = current_start

    while batch_start < end_date:
        # Calculate end of batch (3 months later)
        batch_end = batch_start
        for _ in range(batch_size):
            next_month = batch_end.month + 1
            next_year = batch_end.year
            if next_month > 12:
                next_month = 1
                next_year += 1
            batch_end = datetime(next_year, next_month, 1)
        
        # Ensure we don't go past end_date
        batch_end = min(batch_end, end_date)
        periods.append((batch_start, batch_end))
        batch_start = batch_end
    
    return periods


def get_daily_periods(years_back, start_from_date=None):
    """Generate start and end datetimes for each day in the lookback period with optimized batches."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)

    if start_from_date:
        start_date = max(start_date, start_from_date)

    periods = []
    current_day = start_date.date()

    # Use larger batches for better performance (7 days at a time)
    batch_size = 7
    batch_start = current_day

    while batch_start < end_date.date():
        # Calculate end of batch (7 days later)
        batch_end = batch_start + timedelta(days=batch_size)
        
        # Ensure we don't go past end_date
        if batch_end > end_date.date():
            batch_end = end_date.date()
        
        start_dt = datetime.combine(batch_start, datetime.min.time())
        end_dt = datetime.combine(batch_end, datetime.min.time())
        periods.append((start_dt, end_dt))
        batch_start = batch_end
    
    return periods


@handle_network_operations(
    max_retries=MAX_RETRIES,
    default_return=None,
)
def download_with_retry(api_call, description):
    """Wrapper to retry an API call on failure."""
    # The decorator handles retries and logging, so just call the API
    return api_call()


@handle_file_operations(
    default_return=None,  # Return None on failure
    context="download_klines_data",
)
async def download_klines_data(client, exchange_name, symbol, interval, final_filename, lookback_years):
    """Downloads k-line data incrementally, saving to a CSV file."""
    logger.info(f"--- Downloading K-line Data for {symbol} ({interval}) from {exchange_name} ---")

    # Find the latest timestamp from existing monthly CSV files
    last_timestamp = None
    print(f"🔍 Checking for existing monthly CSV files...")
    logger.info(f"🔍 Checking for existing monthly CSV files...")
    
    # Look for existing monthly CSV files
    import glob
    pattern = f"klines_{exchange_name}_{symbol}_{interval}_*.csv"
    existing_files = glob.glob(os.path.join("data_cache", pattern))
    
    if existing_files:
        print(f"📁 Found {len(existing_files)} existing monthly files:")
        logger.info(f"📁 Found {len(existing_files)} existing monthly files:")
        for file in sorted(existing_files):
            file_size = os.path.getsize(file)
            print(f"   📄 {os.path.basename(file)} ({file_size:,} bytes)")
            logger.info(f"   📄 {os.path.basename(file)} ({file_size:,} bytes)")
        
        # Find the latest timestamp across all files
        latest_timestamps = []
        for file in existing_files:
            try:
                df = pd.read_csv(file)
                if not df.empty and 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    latest_timestamps.append(df['timestamp'].max())
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
                logger.warning(f"⚠️ Error reading {file}: {e}")
        
        if latest_timestamps:
            last_timestamp = max(latest_timestamps)
            print(f"✅ Latest timestamp found: {last_timestamp}")
            logger.info(f"✅ Latest timestamp found: {last_timestamp}")
        else:
            print(f"⚠️ No valid timestamps found in existing files")
            logger.warning(f"⚠️ No valid timestamps found in existing files")
    else:
        print(f"📝 No existing monthly files found. Starting fresh download.")
        logger.info(f"📝 No existing monthly files found. Starting fresh download.")

    start_from_dt = last_timestamp + timedelta(minutes=1) if last_timestamp else None
    logger.info(f"📅 Using monthly periods for klines download")
    monthly_periods = get_monthly_periods(lookback_years, start_from_date=start_from_dt)
    logger.info(f"📊 Will download data for {len(monthly_periods)} monthly periods")
    logger.info(f"📅 Date range: {monthly_periods[0][0].strftime('%Y-%m')} to {monthly_periods[-1][1].strftime('%Y-%m')}")
    logger.info(f"⏱️ Estimated time: ~{len(monthly_periods) * 2} minutes")

    for i, (start_dt, end_dt) in enumerate(monthly_periods, 1):
        # Check if we already have this month's data
        expected_filename = f"klines_{exchange_name}_{symbol}_{interval}_{start_dt.strftime('%Y-%m')}.csv"
        expected_filepath = os.path.join("data_cache", expected_filename)
        
        if os.path.exists(expected_filepath):
            file_size = os.path.getsize(expected_filepath)
            print(f"📁 [{i}/{len(monthly_periods)}] SKIPPING {start_dt.strftime('%Y-%m')} - File exists ({file_size:,} bytes)")
            logger.info(f"📁 [{i}/{len(monthly_periods)}] SKIPPING {start_dt.strftime('%Y-%m')} - File exists ({file_size:,} bytes)")
            continue
        
        print(f"📥 [{i}/{len(monthly_periods)}] Fetching k-lines for {start_dt.strftime('%Y-%m')}...")
        logger.info(f"📥 [{i}/{len(monthly_periods)}] Fetching k-lines for {start_dt.strftime('%Y-%m')}...")
        logger.info(f"   📊 Progress: {i}/{len(monthly_periods)} ({i/len(monthly_periods)*100:.1f}%)")
        
        # Convert to milliseconds for API calls
        start_time_ms = int(start_dt.timestamp() * 1000)
        end_time_ms = int(end_dt.timestamp() * 1000)
        
        print(f"   ⏰ Time range: {start_dt} to {end_dt}")
        print(f"   📊 Expected duration: {(end_dt - start_dt).total_seconds() / 3600:.1f} hours")
        print(f"   📈 Expected klines: ~{int((end_dt - start_dt).total_seconds() / 60)} records")
        logger.info(f"   ⏰ Time range: {start_dt} to {end_dt}")
        logger.info(f"   🔢 Timestamps: {start_time_ms} to {end_time_ms}")
        logger.info(f"   📊 Expected duration: {(end_dt - start_dt).total_seconds() / 3600:.1f} hours")
        logger.info(f"   📈 Expected klines: ~{int((end_dt - start_dt).total_seconds() / 60)} records")
        logger.info(f"   🔄 API call parameters:")
        logger.info(f"      - Symbol: {symbol}")
        logger.info(f"      - Interval: {interval}")
        logger.info(f"      - Limit: 1000")
        
        try:
            print(f"   🔌 Making API call to {exchange_name}...")
            print(f"   ⏱️ Starting download at {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"   🔌 Making API call to {exchange_name}...")
            logger.info(f"   ⏱️ Starting download at {datetime.now().strftime('%H:%M:%S')}")
            
            if exchange_name.upper() == "BINANCE":
                print(f"   🏦 Using Binance API method")
                logger.info(f"   🏦 Using Binance API method")
                klines = await client.get_historical_klines(
                    symbol,
                    interval,
                    start_time_ms,
                    end_time_ms,
                    limit=5000,  # Much larger limit for faster downloads
                )
            else:
                print(f"   🏦 Using {exchange_name} API method")
                logger.info(f"   🏦 Using {exchange_name} API method")
                # MEXC and Gate.io use async methods with internal pagination
                klines = await client.get_historical_klines(
                    symbol,
                    interval,
                    start_time_ms,
                    end_time_ms,
                    limit=5000,  # Much larger limit for faster downloads
                )
            
            print(f"   ⏱️ Download completed at {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"   ⏱️ Download completed at {datetime.now().strftime('%H:%M:%S')}")
            
            if klines:
                print(f"   ✅ Received {len(klines)} klines for {start_dt.strftime('%Y-%m')}")
                logger.info(f"   ✅ Received {len(klines)} klines for {start_dt.strftime('%Y-%m')}")
                print(f"   📊 Data validation:")
                print(f"      - Response type: {type(klines)}")
                print(f"      - Response length: {len(klines)}")
                logger.info(f"   📊 Data validation:")
                logger.info(f"      - Response type: {type(klines)}")
                logger.info(f"      - Response length: {len(klines)}")
                if klines:
                    print(f"      - First kline fields: {len(klines[0])}")
                    print(f"      - Sample first kline: {klines[0][:6]}...")  # Show first 6 fields
                    logger.info(f"      - First kline fields: {len(klines[0])}")
                    logger.info(f"      - Sample first kline: {klines[0][:6]}...")  # Show first 6 fields
                
                # Save this month's data immediately
                print(f"   💾 Saving {start_dt.strftime('%Y-%m')} data immediately...")
                logger.info(f"   💾 Saving {start_dt.strftime('%Y-%m')} data immediately...")
                
                # Create DataFrame for this month
                df_month = pd.DataFrame(
                    klines,
                    columns=[
                        "open_time",
                        "open",
                        "high", 
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_volume",
                        "trades",
                        "taker_buy_base",
                        "taker_buy_quote",
                        "ignore"
                    ],
                )
                
                # Convert timestamp and set index
                df_month["timestamp"] = pd.to_datetime(df_month["open_time"], unit="ms")
                df_month.set_index("timestamp", inplace=True)
                
                # Convert numeric columns
                numeric_cols = ["open", "high", "low", "close", "volume"]
                df_month[numeric_cols] = df_month[numeric_cols].apply(pd.to_numeric, errors="coerce")
                
                # Save to monthly file
                monthly_filename = f"klines_{exchange_name}_{symbol}_{interval}_{start_dt.strftime('%Y-%m')}.csv"
                monthly_filepath = os.path.join("data_cache", monthly_filename)
                
                # Select only the columns we need
                df_month_final = df_month[["open", "high", "low", "close", "volume"]].reset_index()
                df_month_final.to_csv(monthly_filepath, index=False)
                
                file_size = os.path.getsize(monthly_filepath)
                print(f"   ✅ SAVED {start_dt.strftime('%Y-%m')} to {monthly_filename} ({file_size:,} bytes)")
                logger.info(f"   ✅ SAVED {start_dt.strftime('%Y-%m')} to {monthly_filename} ({file_size:,} bytes)")
                print(f"   📈 Progress: {i}/{len(monthly_periods)} months completed")
                logger.info(f"   📈 Progress: {i}/{len(monthly_periods)} months completed")
            else:
                print(f"   ⚠️ No klines received for {start_dt.strftime('%Y-%m')}")
                print(f"   🔍 Response was empty or None")
                logger.warning(f"   ⚠️ No klines received for {start_dt.strftime('%Y-%m')}")
                logger.warning(f"   🔍 Response was empty or None")
        except Exception as e:
            logger.error(f"   ❌ Error downloading klines for {start_dt.strftime('%Y-%m')}: {e}")
            logger.error(f"   🔍 Exception type: {type(e).__name__}")
            logger.error(f"   📍 Exception location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
            continue

    print(f"🎉 All monthly files processed successfully!")
    logger.info(f"🎉 All monthly files processed successfully!")
    return True


@handle_file_operations(
    default_return=None,  # Return None on failure
    context="download_agg_trades_data",
)
async def download_agg_trades_data(client, exchange_name, symbol, final_filename, lookback_years):
    """Downloads aggregated trade data incrementally, saving to a CSV file."""
    logger.info(f"--- Downloading Aggregated Trades Data for {symbol} from {exchange_name} ---")
    logger.info(f"📊 Function parameters:")
    logger.info(f"   - Exchange: {exchange_name}")
    logger.info(f"   - Symbol: {symbol}")
    logger.info(f"   - Output file: {final_filename}")
    logger.info(f"   - Lookback years: {lookback_years}")
    logger.info(f"   - Client type: {type(client).__name__}")
    logger.info(f"   - Client methods: {[m for m in dir(client) if not m.startswith('_')]}")
    
    # Add direct print statements for debugging
    print(f"🔍 DEBUG: download_agg_trades_data called")
    print(f"🔍 DEBUG: Exchange: {exchange_name}")
    print(f"🔍 DEBUG: Symbol: {symbol}")
    print(f"🔍 DEBUG: Client type: {type(client).__name__}")
    print(f"🔍 DEBUG: Client methods: {[m for m in dir(client) if not m.startswith('_')]}")

    # Find the latest timestamp from existing daily CSV files
    existing_df = pd.DataFrame()
    last_timestamp = None
    print(f"🔍 Checking for existing daily agg trades CSV files...")
    logger.info(f"🔍 Checking for existing daily agg trades CSV files...")
    
    # Look for existing daily CSV files
    import glob
    pattern = f"aggtrades_{exchange_name}_{symbol}_*.csv"
    existing_files = glob.glob(os.path.join("data_cache", pattern))
    
    if existing_files:
        print(f"📁 Found {len(existing_files)} existing daily agg trades files:")
        logger.info(f"📁 Found {len(existing_files)} existing daily agg trades files:")
        for file in sorted(existing_files)[-5:]:  # Show last 5 files
            file_size = os.path.getsize(file)
            print(f"   📄 {os.path.basename(file)} ({file_size:,} bytes)")
            logger.info(f"   📄 {os.path.basename(file)} ({file_size:,} bytes)")
        
        # Find the latest timestamp across all files
        latest_timestamps = []
        for file in existing_files:
            try:
                df = pd.read_csv(file)
                if not df.empty and 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    latest_timestamps.append(df['timestamp'].max())
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
                logger.warning(f"⚠️ Error reading {file}: {e}")
        
        if latest_timestamps:
            last_timestamp = max(latest_timestamps)
            print(f"✅ Latest timestamp found: {last_timestamp}")
            logger.info(f"✅ Latest timestamp found: {last_timestamp}")
            
            # Load existing data from the consolidated file if it exists
            if os.path.exists(final_filename) and os.path.getsize(final_filename) > 0:
                try:
                    logger.info(f"📖 Reading existing consolidated agg trades file: {final_filename}")
                    existing_df = pd.read_csv(
                        final_filename,
                        index_col="timestamp",
                        parse_dates=True,
                    )
                    logger.info(f"✅ Loaded existing consolidated data with {len(existing_df)} records")
                except Exception as e:
                    logger.warning(f"⚠️ Error reading consolidated file: {e}")
        else:
            print(f"⚠️ No valid timestamps found in existing files")
            logger.warning(f"⚠️ No valid timestamps found in existing files")
    else:
        print(f"📝 No existing daily agg trades files found. Starting fresh download.")
        logger.info(f"📝 No existing daily agg trades files found. Starting fresh download.")

    start_from_dt = (
        last_timestamp + timedelta(milliseconds=1) if last_timestamp else None
    )
    daily_periods = get_daily_periods(lookback_years, start_from_date=start_from_dt)
    logger.info(f"📊 Will download agg trades for {len(daily_periods)} daily periods")

    for i, (start_dt, end_dt) in enumerate(daily_periods, 1):
        logger.info(f"📥 [{i}/{len(daily_periods)}] Fetching agg trades for {start_dt.strftime('%Y-%m-%d')}...")
        
        start_time_ms = int(start_dt.timestamp() * 1000)
        end_time_ms = int(end_dt.timestamp() * 1000)

        if last_timestamp and start_dt.date() == last_timestamp.date():
            start_time_ms = int(last_timestamp.timestamp() * 1000) + 1
            logger.info(f"   ⏰ Resuming from last timestamp: {last_timestamp}")

        logger.info(f"   ⏰ Time range: {start_dt} to {end_dt}")
        logger.info(f"   🔢 Timestamps: {start_time_ms} to {end_time_ms}")

        try:
            if exchange_name.upper() == "BINANCE":
                # Binance uses async API
                trades = await client.get_aggregate_trades(
                    symbol=symbol,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                    limit=5000,  # Much larger limit for faster downloads
                )
            elif exchange_name.upper() == "MEXC":
                # MEXC uses async method with proper error handling
                try:
                    logger.info(f"   🔄 Calling MEXC get_historical_agg_trades...")
                    logger.info(f"   📊 Parameters: symbol={symbol}, start_time_ms={start_time_ms}, end_time_ms={end_time_ms}, limit=1000")
                    logger.info(f"   ⏰ Time range: {datetime.fromtimestamp(start_time_ms / 1000)} to {datetime.fromtimestamp(end_time_ms / 1000)}")
                    
                    print(f"🔍 DEBUG: About to call MEXC get_historical_agg_trades")
                    print(f"🔍 DEBUG: Parameters: symbol={symbol}, start_time_ms={start_time_ms}, end_time_ms={end_time_ms}")
                    
                    trades = await client.get_historical_agg_trades(
                        symbol=symbol,
                        start_time_ms=start_time_ms,
                        end_time_ms=end_time_ms,
                        limit=5000,  # Much larger limit for faster downloads
                    )
                    logger.info(f"   ✅ MEXC get_historical_agg_trades completed successfully")
                    logger.info(f"   📊 Returned {len(trades)} trades")
                    if trades:
                        logger.info(f"   📋 Sample trade: {trades[0]}")
                        print(f"🔍 DEBUG: Got {len(trades)} trades from MEXC")
                        print(f"🔍 DEBUG: Sample trade: {trades[0]}")
                    else:
                        print(f"🔍 DEBUG: No trades returned from MEXC")
                except Exception as e:
                    logger.warning(f"   ⚠️ MEXC historical agg trades failed: {e}")
                    logger.warning(f"   🔍 Exception type: {type(e).__name__}")
                    logger.warning(f"   📋 Exception details: {str(e)}")
                    print(f"🔍 DEBUG: MEXC exception: {type(e).__name__}: {e}")
                    trades = []  # Return empty list instead of retrying
            else:
                # Gate.io and others use async methods with internal pagination
                trades = await client.get_historical_agg_trades(
                    symbol=symbol,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                    limit=5000,  # Much larger limit for faster downloads
                )
            
            if trades:
                logger.info(f"   ✅ Received {len(trades)} trades for {start_dt.strftime('%Y-%m-%d')}")
                
                # Save this day's data immediately
                print(f"   💾 Saving {start_dt.strftime('%Y-%m-%d')} agg trades data immediately...")
                logger.info(f"   💾 Saving {start_dt.strftime('%Y-%m-%d')} agg trades data immediately...")
                
                # Create DataFrame for this day
                df_day = pd.DataFrame(trades)
                logger.info(f"📈 DataFrame created with {len(df_day)} rows")
                logger.info(f"📋 Columns: {list(df_day.columns)}")

                # Ensure consistent column names across exchanges
                if exchange_name.upper() == "BINANCE":
                    # Binance uses different column names
                    df_day.rename(
                        columns={
                            "a": "agg_trade_id",
                            "p": "price",
                            "q": "quantity",
                            "T": "timestamp",
                            "m": "is_buyer_maker",
                        },
                        inplace=True,
                    )
                    logger.info("🔄 Renamed Binance columns to standard format")
                elif exchange_name.upper() == "MEXC":
                    # MEXC uses 'T' for timestamp, similar to Binance
                    df_day.rename(
                        columns={
                            "a": "agg_trade_id",
                            "p": "price",
                            "q": "quantity",
                            "T": "timestamp",
                            "m": "is_buyer_maker",
                        },
                        inplace=True,
                    )
                    logger.info("🔄 Renamed MEXC columns to standard format")
                else:
                    # Gate.io and others already use standard column names
                    logger.info("✅ Using standard column names")

                df_day["timestamp"] = pd.to_datetime(df_day["timestamp"], unit="ms")
                # Ensure consistent timestamp format by converting to string and back
                df_day["timestamp"] = pd.to_datetime(
                    df_day["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                )
                df_day.set_index("timestamp", inplace=True)
                
                numeric_cols = ["price", "quantity"]
                df_day[numeric_cols] = df_day[numeric_cols].apply(pd.to_numeric, errors="coerce")
                
                logger.info(f"🔢 Numeric conversion completed")
                logger.info(f"📊 Data range: {df_day.index.min()} to {df_day.index.max()}")

                # Combine with existing data
                logger.info(f"🔄 Combining with existing data...")
                combined_df = pd.concat(
                    [existing_df, df_day[["agg_trade_id", "price", "quantity", "is_buyer_maker"]]],
                )
                combined_df = combined_df[
                    ~combined_df["agg_trade_id"].duplicated(keep="last")
                ].sort_index()
                
                logger.info(f"📊 Combined data: {len(combined_df)} rows")
                logger.info(f"📈 Data range: {combined_df.index.min()} to {combined_df.index.max()}")

                # Save to file immediately
                logger.info(f"💾 Saving to {final_filename}...")
                combined_df.to_csv(final_filename)
                
                file_size = os.path.getsize(final_filename)
                print(f"   ✅ SAVED {start_dt.strftime('%Y-%m-%d')} agg trades to {final_filename} ({file_size:,} bytes)")
                logger.info(f"   ✅ SAVED {start_dt.strftime('%Y-%m-%d')} agg trades to {final_filename} ({file_size:,} bytes)")
                print(f"   📈 Progress: {i}/{len(daily_periods)} days completed")
                logger.info(f"   📈 Progress: {i}/{len(daily_periods)} days completed")
                
                # Update existing_df for next iteration
                existing_df = combined_df
            else:
                logger.warning(f"   ⚠️ No trades received for {start_dt.strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.error(f"   ❌ Error downloading trades for {start_dt.strftime('%Y-%m-%d')}: {e}")
            continue

    print(f"🎉 All agg trades data processed successfully!")
    logger.info(f"🎉 All agg trades data processed successfully!")
    return True


@handle_file_operations(
    default_return=None,  # Return None on failure
    context="download_futures_data",
)
async def download_futures_data(client, exchange_name, symbol, final_filename, lookback_years: int):
    """Downloads futures data like funding rates."""
    logger.info(f"--- Downloading Futures Data (Funding Rates) for {symbol} from {exchange_name} ---")

    existing_df = pd.DataFrame()
    last_timestamp = None
    if os.path.exists(final_filename) and os.path.getsize(final_filename) > 0:
        try:
            logger.info(f"📖 Reading existing futures file: {final_filename}")
            existing_df = pd.read_csv(
                final_filename,
                index_col="timestamp",
                parse_dates=True,
            )
            if not existing_df.empty:
                last_timestamp = existing_df.index.max()
                logger.info(
                    f"✅ Existing Futures data found up to {last_timestamp}. Will download new data from this point.",
                )
            else:
                logger.info("📝 Existing futures file is empty. Starting fresh download.")
        except pd.errors.EmptyDataError:
            logger.warning(
                f"⚠️ Existing Futures file {final_filename} is empty. Starting fresh.",
            )
            existing_df = pd.DataFrame()
        except Exception as e:
            logger.error(
                f"❌ Error reading {final_filename}: {e}. Re-downloading full history.",
                exc_info=True,
            )
            existing_df = pd.DataFrame()

    start_from_dt = last_timestamp + timedelta(hours=8) if last_timestamp else None
    logger.info(f"📅 Using monthly periods for futures download")
    monthly_periods = get_monthly_periods(lookback_years, start_from_date=start_from_dt)
    logger.info(f"📊 Will download futures data for {len(monthly_periods)} monthly periods")

    for i, (start_dt, end_dt) in enumerate(monthly_periods, 1):
        logger.info(f"📥 [{i}/{len(monthly_periods)}] Fetching futures data for {start_dt.strftime('%Y-%m')}...")
        
        start_time_ms = int(start_dt.timestamp() * 1000)
        end_time_ms = int(end_dt.timestamp() * 1000)
        
        logger.info(f"   ⏰ Time range: {start_dt} to {end_dt}")
        logger.info(f"   🔢 Timestamps: {start_time_ms} to {end_time_ms}")
        
        try:
            if exchange_name.upper() == "BINANCE":
                # Binance has direct funding rate endpoint
                rates = await client.futures_funding_rate(
                    symbol=symbol,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                    limit=1000,
                )
            elif exchange_name.upper() == "GATEIO":
                # Gate.io has funding rate endpoint but may require authentication
                try:
                    rates = await client.get_historical_futures_data(
                        symbol=symbol,
                        start_time_ms=start_time_ms,
                        end_time_ms=end_time_ms,
                    )
                except Exception as e:
                    logger.info(f"   ℹ️ Gate.io futures data failed (likely auth required): {e}")
                    rates = []
            else:
                # MEXC and other exchanges don't have direct funding rate endpoints
                logger.info(f"   ℹ️ {exchange_name} doesn't have direct funding rate endpoint. Skipping.")
                rates = []
            
            if rates:
                logger.info(f"   ✅ Received {len(rates)} funding rates for {start_dt.strftime('%Y-%m')}")
                
                # Save this month's data immediately
                print(f"   💾 Saving {start_dt.strftime('%Y-%m')} futures data immediately...")
                logger.info(f"   💾 Saving {start_dt.strftime('%Y-%m')} futures data immediately...")
                
                # Create DataFrame for this month
                df_month = pd.DataFrame(rates)
                logger.info(f"📈 DataFrame created with {len(df_month)} rows")
                logger.info(f"📋 Columns: {list(df_month.columns)}")

                # Process funding rates data
                if exchange_name.upper() == "BINANCE":
                    df_month["timestamp"] = pd.to_datetime(df_month["fundingTime"], unit="ms")
                    df_month.set_index("timestamp", inplace=True)
                    df_month["fundingRate"] = pd.to_numeric(df_month["fundingRate"])
                    df_month = df_month[["fundingRate"]]
                    logger.info("✅ Processed Binance funding rates data")
                else:
                    # For MEXC and Gate.io, create empty DataFrame with correct structure
                    df_month = pd.DataFrame(columns=["fundingRate"])
                    df_month.index.name = "timestamp"
                    logger.info(f"ℹ️ No funding rates available for {exchange_name}")

                logger.info(f"🔢 Numeric conversion completed")
                if not df_month.empty:
                    logger.info(f"📊 Data range: {df_month.index.min()} to {df_month.index.max()}")

                # Combine with existing data
                logger.info(f"🔄 Combining with existing data...")
                combined_df = pd.concat([existing_df, df_month])
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")].sort_index()
                
                logger.info(f"📊 Combined data: {len(combined_df)} rows")
                if not combined_df.empty:
                    logger.info(f"📈 Data range: {combined_df.index.min()} to {combined_df.index.max()}")

                # Save to file immediately
                logger.info(f"💾 Saving to {final_filename}...")
                combined_df.to_csv(final_filename)
                
                file_size = os.path.getsize(final_filename)
                print(f"   ✅ SAVED {start_dt.strftime('%Y-%m')} futures data to {final_filename} ({file_size:,} bytes)")
                logger.info(f"   ✅ SAVED {start_dt.strftime('%Y-%m')} futures data to {final_filename} ({file_size:,} bytes)")
                print(f"   📈 Progress: {i}/{len(monthly_periods)} months completed")
                logger.info(f"   📈 Progress: {i}/{len(monthly_periods)} months completed")
                
                # Update existing_df for next iteration
                existing_df = combined_df
            else:
                logger.warning(f"   ⚠️ No funding rates received for {start_dt.strftime('%Y-%m')}")
        except Exception as e:
            logger.error(f"   ❌ Error downloading futures data for {start_dt.strftime('%Y-%m')}: {e}")
            continue

    print(f"🎉 All futures data processed successfully!")
    logger.info(f"🎉 All futures data processed successfully!")
    return True


async def main():
    """Main function to download all necessary data."""
    import argparse

    print("🚀 ARES DATA DOWNLOADER - MAIN FUNCTION STARTED")
    print(f"⏰ Main function start time: {datetime.now().strftime('%H:%M:%S')}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Download trading data (klines, aggtrades, futures) for all exchanges",
    )
    parser.add_argument("--symbol", type=str, help="Trading symbol (e.g., ETHUSDT)")
    parser.add_argument("--exchange", type=str, help="Exchange name (e.g., BINANCE, MEXC, GATEIO)")
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="K-line interval (default: 1m)",
    )
    parser.add_argument(
        "--lookback-years",
        type=float,
        default=None,
        help="Number of years of data to download (default: from CONFIG)",
    )

    args = parser.parse_args()
    print(f"📊 Parsed arguments: symbol={args.symbol}, exchange={args.exchange}, interval={args.interval}")

    start_time = datetime.now()
    print("=" * 80)
    print("🚀 ARES MULTI-EXCHANGE DATA DOWNLOADER")
    print("=" * 80)
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Downloading data for {args.symbol} on {args.exchange}")
    print(f"📈 Interval: {args.interval}")
    print("=" * 80)
    
    logger.info("=" * 80)
    logger.info("🚀 ARES MULTI-EXCHANGE DATA DOWNLOADER")
    logger.info("=" * 80)
    logger.info(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📊 Downloading data for {args.symbol} on {args.exchange}")
    logger.info(f"📈 Interval: {args.interval}")
    logger.info("=" * 80)

    # Ensure cache directory exists
    logger.info("📁 Step 1: Setting up cache directory...")
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"✅ Cache directory '{CACHE_DIR}' ready")
    except PermissionError as e:
        logger.critical(
            f"Permission Error: Could not create/access '{CACHE_DIR}' directory: {e}",
        )
        logger.critical(
            f"Please ensure you have write permissions to the project directory or run: chmod -R u+rw {CACHE_DIR}",
        )
        sys.exit(1)  # Exit immediately on critical permission error

    # Get symbol, exchange, and interval from command line args or config
    symbol = args.symbol or CONFIG.get("SYMBOL", "ETHUSDT")
    exchange = args.exchange or CONFIG.get("EXCHANGE", "BINANCE")
    interval = args.interval or CONFIG.get("INTERVAL", "1m")
    lookback_years = args.lookback_years or CONFIG.get("LOOKBACK_YEARS", 2)

    logger.info("📊 Step 2: Configuration validation...")
    logger.info(f"   Symbol: {symbol}")
    logger.info(f"   Exchange: {exchange}")
    logger.info(f"   Interval: {interval}")
    logger.info(f"   Lookback Years: {lookback_years}")
    logger.info(f"   Cache Directory: {CACHE_DIR}")
    logger.info("✅ Configuration validated")

    # Validate exchange
    logger.info("🔍 Step 3: Exchange validation...")
    supported_exchanges = ["BINANCE", "MEXC", "GATEIO"]
    if exchange.upper() not in supported_exchanges:
        logger.error(f"❌ Unsupported exchange: {exchange}")
        logger.error(f"   Supported exchanges: {supported_exchanges}")
        sys.exit(1)
    logger.info(f"✅ Exchange {exchange} is supported")

    # Initialize exchange client
    print("🔧 Step 4: Initializing exchange client...")
    logger.info("🔧 Step 4: Initializing exchange client...")
    try:
        print(f"   🔌 Connecting to {exchange}...")
        logger.info(f"   🔌 Connecting to {exchange}...")
        print(f"   🏭 Creating ExchangeFactory instance...")
        client = ExchangeFactory.get_exchange(exchange.lower())
        print(f"   ✅ {exchange} client created successfully")
        print(f"   📊 Client type: {type(client).__name__}")
        logger.info(f"✅ {exchange} client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize {exchange} client: {e}")
        print(f"🔍 Exception type: {type(e).__name__}")
        logger.error(f"❌ Failed to initialize {exchange} client: {e}")
        sys.exit(1)

    # Define filenames with exchange prefix and correct period format
    logger.info("📁 Step 5: Setting up file paths...")
    current_date = datetime.now()
    current_month = current_date.strftime("%Y-%m")
    current_day = current_date.strftime("%Y-%m-%d")

    klines_filename = os.path.join(
        CACHE_DIR,
        f"klines_{exchange}_{symbol}_{interval}_{current_month}.csv",
    )
    agg_trades_filename = os.path.join(
        CACHE_DIR,
        f"aggtrades_{exchange}_{symbol}_consolidated.csv",
    )
    futures_filename = os.path.join(CACHE_DIR, f"futures_{exchange}_{symbol}_consolidated.csv")

    logger.info(f"   📄 Klines: {klines_filename}")
    logger.info(f"   📄 Agg Trades: {agg_trades_filename}")
    logger.info(f"   📄 Futures: {futures_filename}")
    logger.info("✅ File paths configured")

    try:
        # Download klines data
        print("=" * 60)
        print("📈 PHASE 1: DOWNLOADING KLINES DATA")
        print("=" * 60)
        logger.info("=" * 60)
        logger.info("📈 PHASE 1: DOWNLOADING KLINES DATA")
        logger.info("=" * 60)
        
        print(f"⏱️ Starting klines download at: {datetime.now().strftime('%H:%M:%S')}")
        klines_start = datetime.now()
        print(f"🔍 Calling download_klines_data function...")
        klines_success = await download_klines_data(client, exchange, symbol, interval, klines_filename, lookback_years)
        klines_duration = (datetime.now() - klines_start).total_seconds()
        print(f"⏱️ Klines download completed at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏱️ Klines duration: {klines_duration:.2f} seconds")
        
        if not klines_success:
            print("❌ Klines download failed")
            logger.error("❌ Klines download failed")
            return False
        print("✅ Klines download completed successfully")
        logger.info(f"✅ Klines download completed in {klines_duration:.2f} seconds")

        # Download aggregated trades data
        logger.info("=" * 60)
        logger.info("📊 PHASE 2: DOWNLOADING AGGREGATED TRADES DATA")
        logger.info("=" * 60)
        logger.info(f"📊 Exchange: {exchange}")
        logger.info(f"📊 Symbol: {symbol}")
        logger.info(f"📊 Lookback years: {lookback_years}")
        logger.info(f"📊 Output file: {agg_trades_filename}")
        
        trades_start = datetime.now()
        trades_success = await download_agg_trades_data(client, exchange, symbol, agg_trades_filename, lookback_years)
        trades_duration = (datetime.now() - trades_start).total_seconds()
        
        if not trades_success:
            logger.error("❌ Aggregated trades download failed")
            return False
        logger.info(f"✅ Aggregated trades download completed in {trades_duration:.2f} seconds")

        # Download futures data
        logger.info("=" * 60)
        logger.info("📈 PHASE 3: DOWNLOADING FUTURES DATA")
        logger.info("=" * 60)
        futures_start = datetime.now()
        futures_success = await download_futures_data(client, exchange, symbol, futures_filename, lookback_years)
        futures_duration = (datetime.now() - futures_start).total_seconds()
        
        if not futures_success:
            logger.error("❌ Futures data download failed")
            return False
        logger.info(f"✅ Futures data download completed in {futures_duration:.2f} seconds")

        # Final summary
        total_duration = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info("🎉 ALL DOWNLOADS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 Summary for {exchange} {symbol}:")
        logger.info(f"   ✅ Klines: {klines_filename} ({klines_duration:.2f}s)")
        logger.info(f"   ✅ Agg Trades: {agg_trades_filename} ({trades_duration:.2f}s)")
        logger.info(f"   ✅ Futures: {futures_filename} ({futures_duration:.2f}s)")
        logger.info(f"   ⏱️ Total duration: {total_duration:.2f} seconds")
        logger.info("=" * 60)
        
        return True

    except Exception as e:
        logger.error(f"❌ Critical error during download process: {e}")
        return False
    finally:
        # Close client connection
        if hasattr(client, 'close'):
            try:
                if asyncio.iscoroutinefunction(client.close):
                    await client.close()
                else:
                    client.close()
                logger.info(f"🔌 {exchange} client connection closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing {exchange} client: {e}")


if __name__ == "__main__":
    # Ensure logging is set up if this script is run directly
    try:
        from src.utils.logger import setup_logging

        setup_logging()
    except ImportError:
        pass  # Continue without setup_logging if not available
    asyncio.run(main()) # Run the main function using asyncio.run

    logger.info("\nAll data has been downloaded and cached locally.")
