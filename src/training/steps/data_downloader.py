# src/training/steps/data_downloader.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import asyncio
import time
import traceback
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import sqlite3
import glob
import ccxt.async_support as ccxt

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.logger import system_logger
from src.utils.error_handler import (
    handle_file_operations,
    handle_network_operations,
)
from exchange.binance import BinanceExchange

# --- Configuration ---
CACHE_DIR = "data_cache"  # Directory to store data files
MAX_RETRIES = 3  # Number of times to retry a failed API call
RETRY_DELAY_SECONDS = 5  # Seconds to wait between retries

# Use system_logger
logger = system_logger.getChild("DataDownloader")


def get_monthly_periods(years_back, start_from_date=None):
    """Generate start and end datetimes for each month in the lookback period."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)

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
        periods.append((current_start, min(current_end, end_date)))
        current_start = current_end
    return periods


def get_daily_periods(years_back, start_from_date=None):
    """Generate start and end datetimes for each day in the lookback period."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)

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

def _get_interval_ms(interval: str) -> int:
    """Convert interval string to milliseconds."""
    interval_map = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '8h': 8 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '3d': 3 * 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000,
        '1M': 30 * 24 * 60 * 60 * 1000,  # Approximate
    }
    return interval_map.get(interval, 60 * 1000)  # Default to 1 minute

@handle_network_operations(
    max_retries=MAX_RETRIES, default_return=None, context="download_with_retry"
)
async def download_with_retry(api_call, description):
    """Wrapper to retry an API call on failure."""
    # The decorator handles retries and logging, so just call the API
    return await api_call()


@handle_file_operations(
    default_return=None,  # Return None on failure
    context="download_klines_data",
)
async def download_klines_data(client, exchange_name, symbol, interval, lookback_years):
    """Downloads k-line data into monthly CSV files, resuming for the current month."""
    logger.info(f"--- Downloading K-line Data for {symbol} ({interval}) into monthly files (incremental) ---")

    monthly_periods = get_monthly_periods(lookback_years)

    for start_dt, end_dt in monthly_periods:
        month_str = start_dt.strftime('%Y-%m')
        filename = os.path.join(CACHE_DIR, f"klines_{exchange_name}_{symbol}_{interval}_{month_str}.csv")

        # For past months, if the file exists, we assume it's complete and skip.
        is_current_month = (start_dt.year == datetime.now().year and start_dt.month == datetime.now().month)
        if os.path.exists(filename) and not is_current_month:
            logger.info(f"Skipping already downloaded k-lines for past month: {month_str}")
            continue

        # For the current month, or for past months where the file doesn't exist,
        # check for existing data to resume from.
        start_ts = int(start_dt.timestamp() * 1000)
        existing_df = pd.DataFrame()

        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename)
                if not existing_df.empty and 'timestamp' in existing_df.columns:
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                    last_timestamp = existing_df['timestamp'].max()
                    # Start downloading from the next kline interval
                    start_ts = int(last_timestamp.timestamp() * 1000) + _get_interval_ms(interval)
                    logger.info(f"Resuming k-line download for {month_str} from {last_timestamp}")
            except (pd.errors.EmptyDataError, FileNotFoundError):
                logger.warning(f"File {filename} is empty or corrupted. Re-downloading for {month_str}.")
                existing_df = pd.DataFrame()

        end_ts = int(end_dt.timestamp() * 1000)

        if start_ts >= end_ts:
            logger.info(f"K-line data for {month_str} is already up to date.")
            continue

        logger.info(f"Fetching k-lines for {month_str} from {datetime.fromtimestamp(start_ts / 1000)}...")
        klines = await download_with_retry(
            lambda: client.get_historical_klines(
                symbol, interval, start_ts, end_ts
            ),
            f"k-lines for {month_str}",
        )

        if klines:
            new_df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
                ],
            )
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")
            
            # Combine with existing data for the month
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            combined_df.sort_values('timestamp', inplace=True)
            
            # Save the combined data
            combined_df[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(filename, index=False)
            logger.info(f"Saved/Updated {len(combined_df)} klines to {filename}")
        else:
            logger.info(f"No new kline data for {month_str}.")
            # If the file didn't exist and we got no data, create an empty file to mark as complete for past months.
            if not os.path.exists(filename):
                 pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]).to_csv(filename, index=False)

    return True


@handle_file_operations(
    default_return=None,  # Return None on failure
    context="download_agg_trades_data",
)
async def download_agg_trades_data(client, exchange_name, symbol, lookback_years):
    """Downloads aggregated trade data into daily CSV files, resuming from the most recent data."""
    logger.info(f"--- Downloading Aggregated Trades Data for {symbol} into daily files (incremental) ---")

    # --- EFFICIENTLY FIND RESUME POINT ---
    most_recent_timestamp_ms = 0
    file_pattern = os.path.join(CACHE_DIR, f"aggtrades_{exchange_name}_{symbol}_????-??-??.csv")
    csv_files = sorted(glob.glob(file_pattern), reverse=True)

    if csv_files:
        logger.info(f"Found {len(csv_files)} aggtrade files. Checking most recent for resume point.")
        for file_path in csv_files:
            try:
                if os.path.getsize(file_path) < 80:  # Skip empty/header-only files
                    continue
                df = pd.read_csv(file_path)
                if not df.empty and 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df.dropna(subset=['timestamp'], inplace=True)
                    if not df.empty:
                        last_ts = df['timestamp'].max()
                        most_recent_timestamp_ms = int(last_ts.timestamp() * 1000)
                        logger.info(f"Found most recent data in {os.path.basename(file_path)} at {last_ts}")
                        break  # Found the latest, stop searching
            except Exception as e:
                logger.warning(f"Could not read or parse {os.path.basename(file_path)}: {e}. Checking next file.")
                continue

    # --- DETERMINE DATE RANGE TO DOWNLOAD ---
    if most_recent_timestamp_ms > 0:
        resume_datetime = datetime.fromtimestamp(most_recent_timestamp_ms / 1000)
        start_date = resume_datetime.date()
        logger.info(f"Resuming downloads from {resume_datetime}")
    else:
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).date()
        logger.info(f"No existing data found. Starting download from {start_date}")

    # Generate daily periods from the determined start date to now
    periods = []
    current_day = start_date
    while current_day <= datetime.now().date():
        start_dt = datetime.combine(current_day, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        periods.append((start_dt, end_dt))
        current_day += timedelta(days=1)

    # --- DOWNLOAD DATA FOR EACH PERIOD ---
    for i, (start_dt, end_dt) in enumerate(periods):
        day_str = start_dt.strftime('%Y-%m-%d')
        filename = os.path.join(CACHE_DIR, f"aggtrades_{exchange_name}_{symbol}_{day_str}.csv")

        # For the very first day of download, we might resume from a specific timestamp
        if i == 0 and most_recent_timestamp_ms > 0:
            start_ts = most_recent_timestamp_ms + 1
            logger.info(f"Resuming first day from timestamp: {datetime.fromtimestamp(start_ts / 1000)}")
        else:
            start_ts = int(start_dt.timestamp() * 1000)

        # Don't download for future dates
        now_ms = int(datetime.now().timestamp() * 1000)
        if start_ts > now_ms:
            continue

        end_ts = int(min(end_dt, datetime.now()).timestamp() * 1000)

        if start_ts >= end_ts:
            logger.info(f"Agg trades data for {day_str} is already up to date.")
            continue

        logger.info(f"Fetching agg trades for {day_str} from {datetime.fromtimestamp(start_ts / 1000)}...")
        
        trades_downloaded_for_day = 0
        current_start_ts = start_ts
        while current_start_ts < end_ts:
            trades = await download_with_retry(
                lambda: client.get_historical_agg_trades_ccxt(
                    symbol=symbol, start_time_ms=current_start_ts, end_time_ms=end_ts
                ),
                f"agg trades for {day_str} from {datetime.fromtimestamp(current_start_ts / 1000)}",
            )
            if not trades:
                logger.info(f"No more trades found for the period starting at {datetime.fromtimestamp(current_start_ts / 1000)}.")
                break

            # --- Incremental Save Logic ---
            # Process and save each batch immediately for robustness.
            batch_df = pd.DataFrame(trades)
            batch_df.rename(
                columns={"a": "agg_trade_id", "p": "price", "q": "quantity", "T": "timestamp", "m": "is_buyer_maker"},
                inplace=True,
            )
            batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"], unit="ms")
            
            # Clean the data to ensure proper structure
            batch_df = batch_df.dropna(subset=['timestamp', 'price', 'quantity'])
            batch_df['trade_date'] = batch_df['timestamp'].dt.date
            
            for trade_date, trades_group in batch_df.groupby('trade_date'):
                date_str_for_file = trade_date.strftime('%Y-%m-%d')
                filename_for_date = os.path.join(CACHE_DIR, f"aggtrades_{exchange_name}_{symbol}_{date_str_for_file}.csv")
                trades_to_save = trades_group.drop('trade_date', axis=1)

                existing_df = pd.DataFrame()
                if os.path.exists(filename_for_date):
                    try:
                        # Use error_bad_lines=False to skip malformed lines
                        existing_df = pd.read_csv(filename_for_date, on_bad_lines='skip')
                        if 'timestamp' in existing_df.columns:
                            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], errors='coerce')
                            # Remove rows with invalid timestamps
                            existing_df = existing_df.dropna(subset=['timestamp'])
                    except (pd.errors.EmptyDataError, Exception) as e:
                        logger.warning(f"Could not read existing file {filename_for_date}: {e}")
                        existing_df = pd.DataFrame()
                
                combined_df = pd.concat([existing_df, trades_to_save], ignore_index=True)
                combined_df.drop_duplicates(subset=['agg_trade_id'], keep='last', inplace=True)
                combined_df.sort_values('timestamp', inplace=True)
                
                # Ensure we have the correct columns and clean data
                cols_to_keep = ["timestamp", "price", "quantity", "is_buyer_maker", "agg_trade_id"]
                final_df = combined_df[[col for col in cols_to_keep if col in combined_df.columns]]
                
                # Clean the data before saving
                final_df = final_df.dropna(subset=['timestamp', 'price', 'quantity'])
                
                # Save with proper error handling
                try:
                    final_df.to_csv(filename_for_date, index=False)
                    logger.info(f"Saved/Updated {len(trades_to_save)} trades to {os.path.basename(filename_for_date)}")
                except Exception as e:
                    logger.error(f"Failed to save {filename_for_date}: {e}")
                    # Try to save with a backup name
                    backup_filename = filename_for_date.replace('.csv', '_backup.csv')
                    final_df.to_csv(backup_filename, index=False)
                    logger.info(f"Saved backup to {backup_filename}")

            trades_downloaded_for_day += len(trades)
            last_trade_ts = int(trades[-1]['T'])
            current_start_ts = last_trade_ts + 1
            if len(trades) < 1000:
                break

        if trades_downloaded_for_day > 0:
            logger.info(f"Finished downloading for {day_str}. Total new trades: {trades_downloaded_for_day}.")
        else:
            logger.info(f"No new agg trades for {day_str}.")
            if not os.path.exists(filename):
                pd.DataFrame(columns=["timestamp", "price", "quantity", "is_buyer_maker", "agg_trade_id"]).to_csv(filename, index=False)

    return True


@handle_file_operations(
    default_return=None,  # Return None on failure
    context="download_futures_data",
)
async def download_futures_data(client, exchange_name, symbol, lookback_years: float):
    """Downloads futures data (funding rates only) into monthly CSV files, resuming for the current month."""
    logger.info(f"--- Downloading Futures Data (Funding Rates) for {symbol} into monthly files (incremental) ---")

    monthly_periods = get_monthly_periods(lookback_years)

    for start_dt, end_dt in monthly_periods:
        month_str = start_dt.strftime('%Y-%m')
        filename = os.path.join(CACHE_DIR, f"futures_{exchange_name}_{symbol}_{month_str}.csv")

        is_current_month = (start_dt.year == datetime.now().year and start_dt.month == datetime.now().month)
        if os.path.exists(filename) and not is_current_month:
            logger.info(f"Skipping already downloaded futures data for past month: {month_str}")
            continue

        start_ts = int(start_dt.timestamp() * 1000)
        existing_df = pd.DataFrame()

        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename)
                if not existing_df.empty and 'timestamp' in existing_df.columns:
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                    last_timestamp = existing_df['timestamp'].max()
                    start_ts = int(last_timestamp.timestamp() * 1000) + 1
                    logger.info(f"Resuming futures data download for {month_str} from {last_timestamp}")
            except (pd.errors.EmptyDataError, FileNotFoundError):
                logger.warning(f"File {filename} is empty or corrupted. Re-downloading for {month_str}.")
                existing_df = pd.DataFrame()

        end_ts = int(end_dt.timestamp() * 1000)

        if start_ts >= end_ts:
            logger.info(f"Futures data for {month_str} is already up to date.")
            continue

        logger.info(f"Fetching futures data for {month_str} from {datetime.fromtimestamp(start_ts / 1000)}...")
        
        # Use CCXT for futures data
        exchange = None
        try:
            # Initialize CCXT exchange for futures
            exchange = ccxt.binance({
                'apiKey': CONFIG.get('binance_api_key', ''),
                'secret': CONFIG.get('binance_api_secret', ''),
                'options': {
                    'defaultType': 'future',  # IMPORTANT: Specify futures market
                },
                'enableRateLimit': True,
            })
            
            # Convert symbol format for CCXT (e.g., ETHUSDT -> ETH/USDT)
            ccxt_symbol = f"{symbol[:-4]}/{symbol[-4:]}" if 'USDT' in symbol else symbol
            
            # --- Fetch Funding Rates with Pagination ---
            all_funding_rates = []
            current_fr_since = start_ts
            while current_fr_since < end_ts:
                logger.info(f"Fetching funding rates for {ccxt_symbol} from {datetime.fromtimestamp(current_fr_since / 1000)}")
                funding_rates = await download_with_retry(
                    lambda: exchange.fetch_funding_rate_history(
                        symbol=ccxt_symbol, since=current_fr_since, limit=1000
                    ),
                    f"funding rates for {month_str}",
                )
                if not funding_rates:
                    break
                all_funding_rates.extend(funding_rates)
                last_ts_fr = funding_rates[-1]['timestamp']
                current_fr_since = last_ts_fr + 1
                if len(funding_rates) < 1000:
                    break
            
            # --- Open Interest Fetching Removed ---
            logger.info(f"Skipping open interest fetch for {month_str} as it is no longer used.")
            
            # Process funding rates
            funding_df = pd.DataFrame()
            if all_funding_rates:
                logger.info(f"Received {len(all_funding_rates)} total funding rate records for {month_str}")
                funding_df = pd.DataFrame(all_funding_rates)
                if not funding_df.empty:
                    funding_df.rename(columns={'fundingRate': 'fundingRate', 'timestamp': 'timestamp'}, inplace=True)
                    funding_df["timestamp"] = pd.to_datetime(funding_df["timestamp"], unit="ms")
                    funding_df["fundingRate"] = pd.to_numeric(funding_df["fundingRate"])
                    funding_df = funding_df[["timestamp", "fundingRate"]]
            else:
                logger.warning(f"No funding rates received for {month_str}")

            # Combine data
            if not funding_df.empty:
                new_df = funding_df
            else:
                new_df = pd.DataFrame()

            if not new_df.empty:
                logger.info(f"Combining {len(new_df)} new records with {len(existing_df)} existing records")
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                combined_df.sort_values('timestamp', inplace=True)
                
                combined_df.to_csv(filename, index=False)
                logger.info(f"Saved/Updated {len(combined_df)} futures data points to {filename}")
            else:
                logger.warning(f"No new futures data to save for {month_str}.")
                # If the file didn't exist and we got no data, create an empty file to mark as complete for past months.
                if not os.path.exists(filename):
                    pd.DataFrame(columns=["timestamp", "fundingRate"]).to_csv(filename, index=False)

        except Exception as e:
            logger.error(f"Error fetching futures data for {month_str}: {e}")
            logger.error(traceback.format_exc())
        finally:
            if exchange:
                await exchange.close()

    return True

async def download_all_data(symbol: str, exchange_name: str, interval: str = "1m"):
    """
    Download all necessary data for a given symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange_name: The name of the exchange (e.g., 'BINANCE')
        interval: K-line interval (e.g., '1m')
        
    Returns:
        Dict containing DataFrames for klines, agg_trades, and futures data
    """
    logger.info("--- Ares Data Downloader ---")

    # The lookback period for the initial download is now sourced directly from the config.
    lookback_years = CONFIG["MODEL_TRAINING"]["data_retention_days"] / 365.0
    logger.info(f"Using a lookback of {lookback_years:.2f} years for any initial data downloads.")
    # Ensure cache directory exists
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except PermissionError as e:
        logger.error(
            f"Permission Error: Could not create/access '{CACHE_DIR}' directory: {e}"
        )
        logger.error(
            f"Please ensure you have write permissions to the project directory or run: chmod -R u+rw {CACHE_DIR}"
        )
        raise

    # Initialize exchange client
    try:
        from src.config import settings
        
        client = BinanceExchange(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            trade_symbol=symbol
        )
    except ImportError:
        logger.error("Could not import settings or BinanceExchange. Using mock client.")
        
        # Fallback for client if import fails
        class MockClient:
            def get_klines(self, *args, **kwargs):
                return []

            def get_historical_agg_trades(self, *args, **kwargs):
                return []

            def get_historical_futures_data(self, *args, **kwargs):
                return []

        client = MockClient()

    # This function now orchestrates the download of dated files.
    # The consolidation will happen in the calling script (e.g., step1_data_collection).
    kline_success = await download_klines_data(
        client, exchange_name, symbol, interval, lookback_years
    )
    agg_trades_success = await download_agg_trades_data(
        client, exchange_name, symbol, lookback_years
    )
    futures_success = await download_futures_data(
        client, exchange_name, symbol, lookback_years
    )

    success = all([kline_success, agg_trades_success, futures_success])
    logger.info(f"Download process completed. Success: {success}")
    return success


async def download_all_data_with_consolidation(symbol: str, exchange_name: str, interval: str = "1m"):
    """
    Download all necessary data and create a consolidated CSV file.
    
    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange_name: The name of the exchange (e.g., 'BINANCE')
        interval: K-line interval (e.g., '1m')
        
    Returns:
        bool: True if all downloads were successful, False otherwise.
    """
    # This function is now a simple wrapper. The name is kept for compatibility
    # with step1_data_collection.py, but consolidation now happens in that step.
    success = await download_all_data(symbol, exchange_name, interval)
    return success


def main():
    """Main function to download all necessary data."""
    try:
        from src.config import CONFIG
        symbol = CONFIG.get("SYMBOL", "ETHUSDT")
        exchange_name = CONFIG.get("EXCHANGE", "BINANCE")
        interval = CONFIG.get("INTERVAL", "1m")
    except ImportError:
        symbol = "ETHUSDT"
        exchange_name = "BINANCE"
        interval = "1m"

    success = asyncio.run(download_all_data_with_consolidation(symbol, exchange_name, interval))
    
    logger.info(f"\nAll data has been downloaded into dated files in '{CACHE_DIR}'.")
    logger.info(f"Success: {success}")
    return success


if __name__ == "__main__":
    # Ensure logging is set up if this script is run directly
    try:
        from src.utils.logger import setup_logging
        setup_logging()
    except ImportError:
        pass  # Continue without setup_logging if not available
    main() 