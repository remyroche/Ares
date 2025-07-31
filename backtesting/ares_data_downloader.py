# backtesting/ares_data_downloader.py

import pandas as pd
from datetime import datetime, timedelta
import os
import sys  # Import sys for sys.exit
import exchange.binance as BinanceExchange

# The new implementation expects a CONFIG dictionary.
# We assume a config file exists at `src/config.py` that defines this.
try:
    from src.config import CONFIG
    from src.utils.logger import system_logger
    from src.utils.error_handler import (
        handle_file_operations,
        handle_network_operations,
    )
except ImportError:
    print(
        "Could not import CONFIG or logging utilities. Using default values and basic print for logs."
    )
    CONFIG = {
        "SYMBOL": "BTCUSDT",
        "INTERVAL": "1h",
        "LOOKBACK_YEARS": 5,
    }

    class MockLogger:
        def info(self, msg):
            print(f"INFO: {msg}")

        def warning(self, msg):
            print(f"WARNING: {msg}")

        def error(self, msg, exc_info=False):
            print(f"ERROR: {msg}")

    system_logger = MockLogger()

    # Mock decorators if not imported
    def handle_file_operations_mock(default_return=None, context=""):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except PermissionError as e:
                    print(
                        f"PERMISSION ERROR in {context} ({func.__name__}): {e}. Please check file/directory permissions."
                    )
                    print(
                        f"Try running: chmod -R u+rw {os.path.dirname(args[0]) if args else '.'}"
                    )
                    sys.exit(1)  # Exit on permission error
                except Exception as e:
                    print(f"ERROR in {context} ({func.__name__}): {e}")
                    return default_return

            return wrapper

        return decorator

    handle_file_operations = handle_file_operations_mock
    def handle_network_operations(**kwargs):
        def decorator(func):
            return func
        return decorator


# --- Configuration ---
CACHE_DIR = "data_cache"  # Directory to store data files
MAX_RETRIES = 3  # Number of times to retry a failed API call
RETRY_DELAY_SECONDS = 5  # Seconds to wait between retries

# Use system_logger if available, otherwise fallback to print
logger = (
    system_logger.getChild("DataDownloader") if "system_logger" in locals() else None
)
if logger is None:

    class TempLogger:
        def info(self, msg):
            print(f"INFO: {msg}")

        def warning(self, msg):
            print(f"WARNING: {msg}")

        def error(self, msg, exc_info=False):
            print(f"ERROR: {msg}")

    logger = TempLogger()


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


@handle_network_operations(
    max_retries=MAX_RETRIES, default_return=None, context="download_with_retry"
)
def download_with_retry(api_call, description):
    """Wrapper to retry an API call on failure."""
    # The decorator handles retries and logging, so just call the API
    return api_call()


@handle_file_operations(
    default_return=None,  # Return None on failure
    context="download_klines_data",
)
def download_klines_data(client, symbol, interval, final_filename, lookback_years):
    """Downloads k-line data incrementally, saving to a CSV file."""
    logger.info(f"--- Downloading K-line Data for {symbol} ({interval}) ---")

    existing_df = pd.DataFrame()
    last_timestamp = None
    if os.path.exists(final_filename) and os.path.getsize(final_filename) > 0:
        try:
            existing_df = pd.read_csv(
                final_filename, index_col="timestamp", parse_dates=True
            )
            if not existing_df.empty:
                last_timestamp = existing_df.index.max()
                logger.info(
                    f"Existing K-line data found up to {last_timestamp}. Downloading new data."
                )
        except pd.errors.EmptyDataError:
            logger.warning(
                f"Existing K-line file {final_filename} is empty. Starting fresh."
            )
            existing_df = pd.DataFrame()
        except Exception as e:
            logger.error(
                f"Error reading {final_filename}: {e}. Re-downloading full history.",
                exc_info=True,
            )
            existing_df = pd.DataFrame()

    start_from_dt = last_timestamp + timedelta(minutes=1) if last_timestamp else None
    monthly_periods = get_monthly_periods(lookback_years, start_from_date=start_from_dt)

    new_klines_data = []
    for start_dt, end_dt in monthly_periods:
        logger.info(f"Fetching k-lines for {start_dt.strftime('%Y-%m')}...")
        klines = download_with_retry(
            lambda: client.get_historical_klines(
                symbol, interval, str(start_dt), str(end_dt)
            ),
            f"k-lines for {start_dt.strftime('%Y-%m')}",
        )
        if klines:
            new_klines_data.extend(klines)

    if not new_klines_data:
        logger.info("No new k-line data downloaded.")
        return True  # Considered successful if no new data needed

    df_new = pd.DataFrame(
        new_klines_data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")
    df_new.set_index("timestamp", inplace=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df_new[numeric_cols] = df_new[numeric_cols].apply(pd.to_numeric, errors="coerce")

    combined_df = pd.concat(
        [existing_df, df_new[["open", "high", "low", "close", "volume"]]]
    )
    combined_df = combined_df[~combined_df.index.duplicated(keep="last")].sort_index()

    combined_df.to_csv(final_filename)
    logger.info(
        f"Updated k-line data saved to '{final_filename}'. Total records: {len(combined_df)}\n"
    )
    return True


@handle_file_operations(
    default_return=None,  # Return None on failure
    context="download_agg_trades_data",
)
def download_agg_trades_data(client, symbol, final_filename, lookback_years):
    """Downloads aggregated trade data incrementally, saving to a CSV file."""
    logger.info(f"--- Downloading Aggregated Trades Data for {symbol} ---")

    existing_df = pd.DataFrame()
    last_timestamp = None
    if os.path.exists(final_filename) and os.path.getsize(final_filename) > 0:
        try:
            existing_df = pd.read_csv(
                final_filename, index_col="timestamp", parse_dates=True
            )
            if not existing_df.empty:
                last_timestamp = existing_df.index.max()
                logger.info(
                    f"Existing Agg Trades data found up to {last_timestamp}. Downloading new data."
                )
        except pd.errors.EmptyDataError:
            logger.warning(
                f"Existing Agg Trades file {final_filename} is empty. Starting fresh."
            )
            existing_df = pd.DataFrame()
        except Exception as e:
            logger.error(
                f"Error reading {final_filename}: {e}. Re-downloading full history.",
                exc_info=True,
            )
            existing_df = pd.DataFrame()

    start_from_dt = (
        last_timestamp + timedelta(milliseconds=1) if last_timestamp else None
    )
    daily_periods = get_daily_periods(lookback_years, start_from_date=start_from_dt)

    all_new_trades = []
    for start_dt, end_dt in daily_periods:
        logger.info(f"Fetching agg trades for {start_dt.strftime('%Y-%m-%d')}...")
        current_start_time = int(start_dt.timestamp() * 1000)
        end_time_ms = int(end_dt.timestamp() * 1000)

        if last_timestamp and start_dt.date() == last_timestamp.date():
            current_start_time = int(last_timestamp.timestamp() * 1000) + 1

        while current_start_time < end_time_ms:
            trades = download_with_retry(
                lambda: client.get_aggregate_trades(
                    symbol=symbol,
                    startTime=current_start_time,
                    endTime=end_time_ms,
                    limit=1000,
                ),
                f"agg trades from {datetime.fromtimestamp(current_start_time / 1000)}",
            )
            if not trades:
                break
            all_new_trades.extend(trades)
            current_start_time = trades[-1]["T"] + 1

    if not all_new_trades:
        logger.info("No new aggregated trade data downloaded.")
        return True  # Considered successful if no new data needed

    df_new = pd.DataFrame(all_new_trades)
    df_new.rename(
        columns={
            "a": "agg_trade_id",
            "p": "price",
            "q": "quantity",
            "T": "timestamp",
            "m": "is_buyer_maker",
        },
        inplace=True,
    )
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")
    df_new.set_index("timestamp", inplace=True)
    numeric_cols = ["price", "quantity"]
    df_new[numeric_cols] = df_new[numeric_cols].apply(pd.to_numeric, errors="coerce")

    combined_df = pd.concat(
        [existing_df, df_new[["agg_trade_id", "price", "quantity", "is_buyer_maker"]]]
    )
    combined_df = combined_df[
        ~combined_df["agg_trade_id"].duplicated(keep="last")
    ].sort_index()

    combined_df.to_csv(final_filename)
    logger.info(
        f"Updated agg trades data saved to '{final_filename}'. Total records: {len(combined_df)}\n"
    )
    return True


@handle_file_operations(
    default_return=None,  # Return None on failure
    context="download_futures_data",
)
def download_futures_data(client, symbol, final_filename, lookback_years: int):
    """Downloads futures data like funding rates."""
    logger.info(f"--- Downloading Futures Data (Funding Rates) for {symbol} ---")

    start_ts = int((datetime.now() - timedelta(days=365 * lookback_years)).timestamp() * 1000)

    # Fetch new funding rates
    new_funding_rates = download_with_retry(
        lambda: client.futures_funding_rate(
            symbol=symbol, startTime=start_ts, limit=1000
        ),
        "funding rate history",
    )
    funding_df = pd.DataFrame()
    if new_funding_rates:
        funding_df = pd.DataFrame(new_funding_rates)
        funding_df["timestamp"] = pd.to_datetime(funding_df["fundingTime"], unit="ms")
        funding_df.set_index("timestamp", inplace=True)
        funding_df["fundingRate"] = pd.to_numeric(funding_df["fundingRate"])
        funding_df = funding_df[["fundingRate"]]

    if funding_df.empty:
        logger.info("Could not retrieve any new futures data.")
        return True

    futures_df = funding_df.sort_index()
    futures_df.to_csv(final_filename)
    logger.info(
        f"Futures data saved to {final_filename}. Total records: {len(futures_df)}\n"
    )
    return True


def main():
    """Main function to download all necessary data."""
    logger.info("--- Ares Data Downloader ---")

    # Ensure cache directory exists
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except PermissionError as e:
        logger.critical(
            f"Permission Error: Could not create/access '{CACHE_DIR}' directory: {e}"
        )
        logger.critical(
            f"Please ensure you have write permissions to the project directory or run: chmod -R u+rw {CACHE_DIR}"
        )
        sys.exit(1)  # Exit immediately on critical permission error

    # This script expects a CONFIG dictionary, likely from `src/config.py`.
    # For demonstration, a default config is provided here.
    try:
        from src.config import CONFIG
        from exchange.binance import (
            exchange,
        )  # Assuming exchange client is properly initialized
    except ImportError:
        logger.error(
            "Could not import CONFIG or BinanceExchange from src.config.py/src.exchange.binance. Using default values."
        )

        # Fallback for client if import fails
        class MockClient:
            def get_historical_klines(self, *args, **kwargs):
                return []

            def get_aggregate_trades(self, *args, **kwargs):
                return []

            def futures_funding_rate(self, *args, **kwargs):
                return []

            def futures_open_interest_hist(self, *args, **kwargs):
                return []

        exchange = MockClient()
        # Ensure CONFIG has necessary keys for fallback
        CONFIG = {
            "SYMBOL": "BTCUSDT",
            "INTERVAL": "1h",
            "LOOKBACK_YEARS": 5,
        }

    symbol = CONFIG["SYMBOL"]
    interval = CONFIG["INTERVAL"]
    lookback_years = CONFIG["LOOKBACK_YEARS"]

    # Define filenames based on config with date format
    # Format: klines_ETHUSDT_1m_2023-10.csv, aggtrades_ETHUSDT_2023-07-28.csv, futures_ETHUSDT_2023-10.csv
    current_date = datetime.now()
    current_month = current_date.strftime("%Y-%m")
    current_day = current_date.strftime("%Y-%m-%d")
    
    klines_filename = os.path.join(CACHE_DIR, f"klines_{symbol}_{interval}_{current_month}.csv")
    agg_trades_filename = os.path.join(CACHE_DIR, f"aggtrades_{symbol}_{current_day}.csv")
    futures_filename = os.path.join(CACHE_DIR, f"futures_{symbol}_{current_month}.csv")

    # Initialize Binance client (already done via src.exchange.binance import)
    client = exchange

    try:
        # Pass the client instance to the download functions
        success_klines = download_klines_data(
            client, symbol, interval, klines_filename, lookback_years
        )
        success_agg_trades = download_agg_trades_data(
            client, symbol, agg_trades_filename, lookback_years
        )
        success_futures = download_futures_data(
            client, symbol, futures_filename, lookback_years
        )

        if all([success_klines, success_agg_trades, success_futures]):
            logger.info("All data downloads attempted and completed successfully.")
            sys.exit(0)  # Exit with success code
        else:
            logger.error("One or more data downloads failed. Check logs for details.")
            sys.exit(1)  # Exit with failure code

    except Exception as e:
        error_message = (
            f"A critical error occurred in the main data download process: {e}"
        )
        logger.critical(error_message, exc_info=True)
        sys.exit(1)  # Exit with failure code


if __name__ == "__main__":
    # Ensure logging is set up if this script is run directly
    try:
        from src.utils.logger import setup_logging
        setup_logging()
    except ImportError:
        pass  # Continue without setup_logging if not available
    main()

    logger.info("\nAll data has been downloaded and cached locally.")
