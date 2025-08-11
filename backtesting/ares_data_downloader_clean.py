#!/usr/bin/env python3
"""
Clean, full-featured data downloader for Ares trading bot.
Supports incremental downloads, parallel processing, and multiple data types.
Uses CCXT directly for better rate limiting.
"""

import argparse
import asyncio
import glob
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import ccxt.async_support as ccxt

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("log/ares_data_downloader.log"),
    ],
)
logger = logging.getLogger("OptimizedDataDownloader")

# Import dependencies
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.config import CONFIG
    from src.utils.logger import get_logger, setup_logging
    from src.utils.warning_symbols import (
        connection_error,
        critical,
        error,
        execution_error,
        failed,
        initialization_error,
        invalid,
        missing,
        problem,
        timeout,
        validation_error,
        warning,
    )

    logger = get_logger("OptimizedDataDownloader")
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    # Fallback configuration
    CONFIG = {
        "SYMBOL": "ETHUSDT",
        "INTERVAL": "1m",
        "LOOKBACK_YEARS": 2,
    }

    # Create fallback functions
    def error(msg):
        return f"❌ ERROR: {msg}"

    def warning(msg):
        return f"⚠️ WARNING: {msg}"

    def critical(msg):
        return f"🚨 CRITICAL: {msg}"

    def problem(msg):
        return f"🔧 PROBLEM: {msg}"

    def failed(msg):
        return f"💥 FAILED: {msg}"

    def invalid(msg):
        return f"❌ INVALID: {msg}"

    def missing(msg):
        return f"📭 MISSING: {msg}"

    def timeout(msg):
        return f"⏰ TIMEOUT: {msg}"

    def connection_error(msg):
        return f"🔌 CONNECTION ERROR: {msg}"

    def validation_error(msg):
        return f"✅ VALIDATION ERROR: {msg}"

    def initialization_error(msg):
        return f"🚀 INITIALIZATION ERROR: {msg}"

    def execution_error(msg):
        return f"⚡ EXECUTION ERROR: {msg}"


@dataclass
class DownloadConfig:
    """Configuration for optimized data downloading."""

    symbol: str
    exchange: str
    interval: str
    lookback_years: int
    max_concurrent_downloads: int = 5
    max_concurrent_requests: int = 10
    chunk_size: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    memory_threshold: float = 0.8
    force: bool = False


class CleanDataDownloader:
    """Clean, full-featured data downloader with parallel processing and incremental downloads."""

    def __init__(self, config: DownloadConfig):
        self.config = config
        self.session = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.download_semaphore = asyncio.Semaphore(config.max_concurrent_downloads)
        self.cache_dir = "data_cache"
        self.exchange = None

        # Initialize stats
        self.stats = {
            "klines_downloaded": 0,
            "aggtrades_downloaded": 0,
            "futures_downloaded": 0,
            "total_time": 0,
            "errors": 0,
        }

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    async def initialize(self):
        """Initialize the downloader and CCXT exchange."""
        print("🔧 STEP 1: Initializing clean downloader with CCXT...")
        logger.info("🔧 STEP 1: Initializing clean downloader with CCXT...")

        try:
            # Create CCXT exchange instance
            exchange_class = getattr(ccxt, self.config.exchange.lower())
            self.exchange = exchange_class(
                {
                    "enableRateLimit": True,  # Enable CCXT's built-in rate limiting
                    "rateLimit": 100,  # 100ms between requests (10 requests per second)
                    "timeout": 30000,  # 30 second timeout
                    "sandbox": False,  # Use live API
                }
            )

            print(f"✅ CCXT exchange created for {self.config.exchange}")
            logger.info(f"✅ CCXT exchange created for {self.config.exchange}")

            return True

        except Exception as e:
            print(failed(f"❌ Initialization failed: {e}"))
            logger.exception(f"❌ Initialization failed: {e}")
            return False

    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.exchange:
                await self.exchange.close()
                print("✅ CCXT exchange closed")
                logger.info("✅ CCXT exchange closed")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
            logger.warning(f"⚠️ Error during cleanup: {e}")

    def _find_latest_aggtrades_timestamp(self) -> datetime | None:
        """Find the latest timestamp in existing aggtrades files by reading actual trade timestamps."""
        try:
            # Find all aggtrades files for this symbol and exchange
            pattern = f"aggtrades_{self.config.exchange}_{self.config.symbol}_*.csv"
            files = glob.glob(os.path.join(self.cache_dir, pattern))

            if not files:
                print("🔍 DEBUG: No existing aggtrades files found")
                return None

            latest_timestamp = None
            latest_file = None

            for file_path in files:
                try:
                    # Read only the last few lines to find the latest trade timestamp efficiently
                    result = subprocess.run(
                        ["tail", "-100", file_path],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        print(f"🔍 DEBUG: Cannot read {file_path}")
                        continue

                    # Parse the CSV data from the tail output
                    lines = result.stdout.strip().split("\n")
                    if len(lines) < 2:  # Need at least header + 1 data line
                        continue

                    # Skip header line
                    data_lines = lines[1:]

                    # Extract timestamps from the data lines
                    timestamps = []
                    for line in data_lines:
                        if "," in line:
                            timestamp_str = line.split(",")[0]
                            try:
                                # Try to parse as datetime string (most common format)
                                timestamp = datetime.strptime(
                                    timestamp_str, "%Y-%m-%d %H:%M:%S.%f"
                                )
                                timestamps.append(timestamp)
                            except ValueError:
                                try:
                                    # Try without microseconds
                                    timestamp = datetime.strptime(
                                        timestamp_str, "%Y-%m-%d %H:%M:%S"
                                    )
                                    timestamps.append(timestamp)
                                except ValueError:
                                    try:
                                        # Try as millisecond timestamp
                                        timestamp = datetime.fromtimestamp(
                                            int(timestamp_str) / 1000
                                        )
                                        timestamps.append(timestamp)
                                    except (ValueError, TypeError):
                                        continue

                    if timestamps:
                        file_latest = max(timestamps)
                        if latest_timestamp is None or file_latest > latest_timestamp:
                            latest_timestamp = file_latest
                            latest_file = file_path

                except Exception as e:
                    print(f"🔍 DEBUG: Error reading {file_path}: {e}")
                    continue

            if latest_timestamp:
                print(
                    f"🔍 DEBUG: Latest timestamp found: {latest_timestamp} from {latest_file}"
                )
                # Return the next second after the latest timestamp
                return latest_timestamp + timedelta(seconds=1)
            print("🔍 DEBUG: No valid timestamps found in existing files")
            return None

        except Exception as e:
            print(f"🔍 DEBUG: Error finding latest timestamp: {e}")
            return None

    def get_time_periods(self, data_type: str) -> list[tuple[datetime, datetime]]:
        """Get time periods for downloading data, excluding already downloaded periods."""
        print(f"📅 STEP 2: Calculating time periods for {data_type}...")
        logger.info(f"📅 STEP 2: Calculating time periods for {data_type}...")

        # For aggtrades, find the latest timestamp in existing files
        if data_type == "aggtrades":
            latest_timestamp = self._find_latest_aggtrades_timestamp()
            if latest_timestamp:
                print(f"🔍 DEBUG: Found latest aggtrades timestamp: {latest_timestamp}")
                start_date = latest_timestamp
                end_date = datetime.now()
            else:
                # If no existing data, use the standard 2-year lookback
                end_date = datetime.now()
                max_days = 365 * self.config.lookback_years
                start_date = end_date - timedelta(days=max_days)
        else:
            # For other data types, use standard lookback
            end_date = datetime.now()
            max_days = 365 * self.config.lookback_years
            start_date = end_date - timedelta(days=max_days)

        print(
            f"📊 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        print(f"📊 Total days: {(end_date - start_date).days}")
        logger.info(
            f"📊 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        logger.info(f"📊 Total days: {(end_date - start_date).days}")

        if data_type == "klines":
            print("📈 Processing klines (monthly periods)...")
            logger.info("📈 Processing klines (monthly periods)...")
            # Monthly periods for klines
            periods = []
            current = start_date.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            month_count = 0
            skip_count = 0

            while current < end_date:
                next_month = current.replace(day=28) + timedelta(days=4)
                next_month = next_month.replace(day=1)
                period_end = min(next_month, end_date)

                # Check if this month's data already exists
                filename = f"klines_{self.config.exchange}_{self.config.symbol}_{self.config.interval}_{current.strftime('%Y-%m')}.csv"
                filepath = os.path.join(self.cache_dir, filename)

                force_mode = getattr(self.config, "force", False)
                if (
                    force_mode
                    or not os.path.exists(filepath)
                    or os.path.getsize(filepath) == 0
                ):
                    periods.append((current, period_end))
                    month_count += 1
                    print(f"📥 Will download: {filename}")
                    logger.info(f"📥 Will download: {filename}")
                else:
                    skip_count += 1
                    print(f"📁 Skipping existing: {filename}")
                    logger.info(f"📁 Skipping existing: {filename}")

                current = next_month

            print(
                f"📊 Summary: {month_count} months to download, {skip_count} months skipped"
            )
            logger.info(
                f"📊 Summary: {month_count} months to download, {skip_count} months skipped"
            )
            return periods

        if data_type == "aggtrades":
            print("📊 Processing aggtrades (daily periods)...")
            logger.info("📊 Processing aggtrades (daily periods)...")
            # Daily periods for aggtrades
            periods = []
            day_count = 0
            skip_count = 0

            # Find the latest timestamp in existing files
            latest_timestamp = self._find_latest_aggtrades_timestamp()
            if latest_timestamp:
                print(f"🔍 DEBUG: Found latest timestamp: {latest_timestamp}")
                # Start from the exact timestamp of the latest trade
                start_date = latest_timestamp

                print(f"🔍 DEBUG: Starting download from: {start_date}")
                current = start_date
            else:
                # If no existing data, use the standard start date
                end_date = datetime.now()
                max_days = 365 * self.config.lookback_years
                start_date = end_date - timedelta(days=max_days)
                current = start_date

            end_date = datetime.now()

            print(
                f"📊 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )
            print(f"📊 Total days: {(end_date - start_date).days}")

            # Create daily periods from current to end_date
            print(f"🔍 DEBUG: Starting daily period loop from {current} to {end_date}")
            while current < end_date:
                period_end = current + timedelta(days=1)
                filename = f"aggtrades_{self.config.exchange}_{self.config.symbol}_{current.strftime('%Y-%m-%d')}.csv"
                filepath = os.path.join(self.cache_dir, filename)

                print(f"🔍 DEBUG: Checking file: {filename}")

                # Only download if file doesn't exist
                if not os.path.exists(filepath):
                    periods.append((current, period_end))
                    day_count += 1
                    print(f"📥 Will download: {filename}")
                else:
                    skip_count += 1
                    print(f"📁 Skipping existing: {filename}")

                current = period_end
                print(f"🔍 DEBUG: Moving to next day: {current}")

            print(
                f"📊 Summary: {day_count} days to download, {skip_count} days skipped"
            )
            return periods

        if data_type == "futures":
            print("📈 Processing futures (monthly periods)...")
            logger.info("📈 Processing futures (monthly periods)...")
            # Monthly periods for futures
            periods = []
            current = start_date.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            month_count = 0
            skip_count = 0

            while current < end_date:
                next_month = current.replace(day=28) + timedelta(days=4)
                next_month = next_month.replace(day=1)
                period_end = min(next_month, end_date)

                # Check if this month's data already exists
                filename = f"futures_{self.config.exchange}_{self.config.symbol}_{current.strftime('%Y-%m')}.csv"
                filepath = os.path.join(self.cache_dir, filename)

                force_mode = getattr(self.config, "force", False)
                if (
                    force_mode
                    or not os.path.exists(filepath)
                    or os.path.getsize(filepath) == 0
                ):
                    periods.append((current, period_end))
                    month_count += 1
                    print(f"📥 Will download: {filename}")
                    logger.info(f"📥 Will download: {filename}")
                else:
                    skip_count += 1
                    print(f"📁 Skipping existing: {filename}")
                    logger.info(f"📁 Skipping existing: {filename}")

                current = next_month

            print(
                f"📊 Summary: {month_count} months to download, {skip_count} months skipped"
            )
            logger.info(
                f"📊 Summary: {month_count} months to download, {skip_count} months skipped"
            )
            return periods

        print(f"❌ Unknown data type: {data_type}")
        logger.error(f"❌ Unknown data type: {data_type}")
        return []

    async def download_klines_parallel(self) -> bool:
        """Download klines data in parallel."""
        print("📈 STEP 3A: Starting parallel klines download...")
        logger.info("📈 STEP 3A: Starting parallel klines download...")

        periods = self.get_time_periods("klines")
        if not periods:
            print("📁 No klines periods to download")
            logger.info("📁 No klines periods to download")
            return True

        print(f"📊 Found {len(periods)} periods to download")
        for start, end in periods:
            print(
                f"📅 Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
            )

        # Download periods in parallel
        tasks = [self._download_klines_period(start, end) for start, end in periods]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success_count = 0
        error_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_count += 1
                if i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                    print(f"❌ Task {i+1} failed: {result}")
                    print(failed(f"❌ Task {i+1} failed: {result}"))
                self.stats["errors"] += 1
            elif result:
                success_count += 1
                self.stats["klines_downloaded"] += 1
                if i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                    print(f"✅ Task {i+1} completed successfully")
                    logger.info(f"✅ Task {i+1} completed successfully")

        print("✅ STEP 3A COMPLETED: Klines download finished")
        print(f"📊 Success: {success_count}/{len(periods)} periods")
        print(f"📊 Errors: {error_count}")
        print(f"📁 CSV Files: {success_count} monthly klines files created")
        logger.info(
            f"✅ STEP 3A COMPLETED: Klines download finished - {success_count}/{len(periods)} periods successful, {error_count} errors"
        )
        logger.info(f"📁 CSV Files: {success_count} monthly klines files created")
        return success_count > 0

    async def download_aggtrades_parallel(self) -> bool:
        """Download aggregated trades data in parallel."""
        print("📊 STEP 3B: Starting parallel aggtrades download...")
        logger.info("📊 STEP 3B: Starting parallel aggtrades download...")

        periods = self.get_time_periods("aggtrades")
        if not periods:
            print("📁 No aggtrades periods to download")
            logger.info("📁 No aggtrades periods to download")
            return True

        print(f"📊 Found {len(periods)} periods to download")
        for start, end in periods:
            print(
                f"📅 Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
            )

        # Download periods in parallel
        tasks = [self._download_aggtrades_period(start, end) for start, end in periods]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success_count = 0
        error_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_count += 1
                if i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                    print(f"❌ Task {i+1} failed: {result}")
                    print(failed(f"❌ Task {i+1} failed: {result}"))
                self.stats["errors"] += 1
            elif result:
                success_count += 1
                self.stats["aggtrades_downloaded"] += 1
                if i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                    print(f"✅ Task {i+1} completed successfully")
                    logger.info(f"✅ Task {i+1} completed successfully")

        print("✅ STEP 3B COMPLETED: Aggtrades download finished")
        print(f"📊 Success: {success_count}/{len(periods)} periods")
        print(f"📊 Errors: {error_count}")
        print(f"📁 CSV Files: {success_count} daily aggtrades files created")
        logger.info(
            f"✅ STEP 3B COMPLETED: Aggtrades download finished - {success_count}/{len(periods)} periods successful, {error_count} errors"
        )
        logger.info(f"📁 CSV Files: {success_count} daily aggtrades files created")
        return success_count > 0

    async def download_futures_parallel(self) -> bool:
        """Download futures data in parallel."""
        print("📈 STEP 3C: Starting parallel futures download...")
        logger.info("📈 STEP 3C: Starting parallel futures download...")

        periods = self.get_time_periods("futures")
        if not periods:
            print("📁 No futures periods to download")
            logger.info("📁 No futures periods to download")
            return True

        print(f"📊 Found {len(periods)} periods to download")
        for start, end in periods:
            print(
                f"📅 Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
            )

        # Download periods in parallel
        tasks = [self._download_futures_period(start, end) for start, end in periods]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success_count = 0
        error_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_count += 1
                if i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                    print(f"❌ Task {i+1} failed: {result}")
                    print(failed(f"❌ Task {i+1} failed: {result}"))
                self.stats["errors"] += 1
            elif result:
                success_count += 1
                self.stats["futures_downloaded"] += 1
                if i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                    print(f"✅ Task {i+1} completed successfully")
                    logger.info(f"✅ Task {i+1} completed successfully")

        print("✅ STEP 3C COMPLETED: Futures download finished")
        print(f"📊 Success: {success_count}/{len(periods)} periods")
        print(f"📊 Errors: {error_count}")
        print(f"📁 CSV Files: {success_count} monthly futures files created")
        logger.info(
            f"✅ STEP 3C COMPLETED: Futures download finished - {success_count}/{len(periods)} periods successful, {error_count} errors"
        )
        logger.info(f"📁 CSV Files: {success_count} monthly futures files created")
        return success_count > 0

    async def _download_klines_period(
        self, start_dt: datetime, end_dt: datetime
    ) -> bool:
        """Download klines data for a specific time period using CCXT."""
        async with self.download_semaphore:
            try:
                # Generate filename for this day
                filename = f"klines_{self.config.exchange}_{self.config.symbol}_{start_dt.strftime('%Y-%m-%d')}.csv"
                filepath = os.path.join(self.cache_dir, filename)

                print(f"📥 Downloading klines for {start_dt.strftime('%Y-%m-%d')}...")
                logger.info(
                    f"📥 Downloading klines for {start_dt.strftime('%Y-%m-%d')}"
                )

                # Convert to milliseconds
                start_ms = int(start_dt.timestamp() * 1000)
                end_ms = int(end_dt.timestamp() * 1000)

                print(f"⏰ Time range: {start_dt} to {end_dt}")
                print(f"🔢 Timestamps: {start_ms} to {end_ms}")
                logger.info(f"⏰ Time range: {start_dt} to {end_dt}")
                logger.info(f"🔢 Timestamps: {start_ms} to {end_ms}")

                # Download data using CCXT with pagination
                print(f"🔌 Making API call to {self.config.exchange}...")
                logger.info(f"🔌 Making API call to {self.config.exchange}")

                all_klines = []
                since = start_ms

                while since < end_ms:
                    klines = await self.exchange.fetch_ohlcv(
                        self.config.symbol,
                        self.config.interval,
                        since=since,
                        limit=1000,
                    )

                    if not klines:
                        break

                    all_klines.extend(klines)

                    # Update since to the timestamp of the last kline
                    if klines:
                        since = klines[-1][0] + 1  # Next timestamp after the last one

                    # Small delay to respect rate limits (CCXT handles this automatically)
                    await asyncio.sleep(0.1)

                if not all_klines:
                    print(f"⚠️ No klines data found for {start_dt.strftime('%Y-%m-%d')}")
                    logger.warning(
                        f"⚠️ No klines data found for {start_dt.strftime('%Y-%m-%d')}"
                    )
                    return False

                # Process and save data
                df = self._process_klines_data(all_klines)
                df.to_csv(filepath, index=False)

                print(f"✅ Saved {len(df)} klines to {filename}")
                logger.info(f"✅ Saved {len(df)} klines to {filename}")
                return True

            except Exception as e:
                print(
                    f"❌ Error downloading klines for {start_dt.strftime('%Y-%m-%d')}: {e}"
                )
                logger.exception(
                    f"❌ Error downloading klines for {start_dt.strftime('%Y-%m-%d')}: {e}"
                )
                return False

    async def _download_aggtrades_period(
        self, start_dt: datetime, end_dt: datetime
    ) -> bool:
        """Download aggregated trades for a specific time period using CCXT."""
        async with self.download_semaphore:
            try:
                # Generate filename for this day
                filename = f"aggtrades_{self.config.exchange}_{self.config.symbol}_{start_dt.strftime('%Y-%m-%d')}.csv"
                filepath = os.path.join(self.cache_dir, filename)

                print(
                    f"📥 Downloading aggtrades for {start_dt.strftime('%Y-%m-%d')}..."
                )
                logger.info(
                    f"📥 Downloading aggtrades for {start_dt.strftime('%Y-%m-%d')}"
                )

                # Convert to milliseconds
                start_ms = int(start_dt.timestamp() * 1000)
                end_ms = int(end_dt.timestamp() * 1000)

                print(f"⏰ Time range: {start_dt} to {end_dt}")
                print(f"🔢 Timestamps: {start_ms} to {end_ms}")
                logger.info(f"⏰ Time range: {start_dt} to {end_dt}")
                logger.info(f"🔢 Timestamps: {start_ms} to {end_ms}")

                # Download data using CCXT with pagination
                print(f"🔌 Making API call to {self.config.exchange}...")
                logger.info(f"🔌 Making API call to {self.config.exchange}")

                all_trades = []
                since = start_ms

                while since < end_ms:
                    trades = await self.exchange.fetch_trades(
                        self.config.symbol,
                        since=since,
                        limit=1000,
                    )

                    if not trades:
                        break

                    # Filter trades within our time range
                    filtered_trades = [t for t in trades if t["timestamp"] <= end_ms]
                    all_trades.extend(filtered_trades)

                    # Update since to the timestamp of the last trade
                    if trades:
                        since = (
                            trades[-1]["timestamp"] + 1
                        )  # Next timestamp after the last one

                    # Small delay to respect rate limits (CCXT handles this automatically)
                    await asyncio.sleep(0.1)

                if not all_trades:
                    print(
                        f"⚠️ No aggtrades data found for {start_dt.strftime('%Y-%m-%d')}"
                    )
                    logger.warning(
                        f"⚠️ No aggtrades data found for {start_dt.strftime('%Y-%m-%d')}"
                    )
                    return False

                # Process and save data
                df = self._process_aggtrades_data(all_trades)
                df.to_csv(filepath, index=False)

                print(f"✅ Saved {len(df)} aggtrades to {filename}")
                logger.info(f"✅ Saved {len(df)} aggtrades to {filename}")
                return True

            except Exception as e:
                print(
                    f"❌ Error downloading aggtrades for {start_dt.strftime('%Y-%m-%d')}: {e}"
                )
                logger.exception(
                    f"❌ Error downloading aggtrades for {start_dt.strftime('%Y-%m-%d')}: {e}"
                )
                return False

    async def _download_futures_period(
        self, start_dt: datetime, end_dt: datetime
    ) -> bool:
        """Download futures-specific data (funding rates) for a specific time period."""
        async with self.download_semaphore:
            try:
                # Generate filename for this month
                filename = f"futures_{self.config.exchange}_{self.config.symbol}_{start_dt.strftime('%Y-%m')}.csv"
                filepath = os.path.join(self.cache_dir, filename)

                print(
                    f"📥 Downloading funding rates for {start_dt.strftime('%Y-%m')}..."
                )
                logger.info(
                    f"📥 Downloading funding rates for {start_dt.strftime('%Y-%m')}"
                )

                # Convert to milliseconds
                start_ms = int(start_dt.timestamp() * 1000)
                end_ms = int(end_dt.timestamp() * 1000)

                print(f"⏰ Time range: {start_dt} to {end_dt}")
                print(f"🔢 Timestamps: {start_ms} to {end_ms}")
                logger.info(f"⏰ Time range: {start_dt} to {end_dt}")
                logger.info(f"🔢 Timestamps: {start_ms} to {end_ms}")

                # Download funding rates data
                print(f"🔌 Making API calls to {self.config.exchange}...")
                logger.info(f"🔌 Making API calls to {self.config.exchange}")

                all_futures_data = []

                # Try different methods to get funding rates
                try:
                    print("📊 Downloading funding rates...")

                    # Method 1: Try fetch_funding_rate_history if available
                    if hasattr(self.exchange, "fetch_funding_rate_history"):
                        try:
                            funding_rates = (
                                await self.exchange.fetch_funding_rate_history(
                                    self.config.symbol, since=start_ms, limit=1000
                                )
                            )

                            for rate in funding_rates:
                                if start_ms <= rate["timestamp"] <= end_ms:
                                    all_futures_data.append(
                                        {
                                            "timestamp": rate["timestamp"],
                                            "funding_rate": rate.get("fundingRate", 0),
                                            "funding_time": rate.get(
                                                "fundingTime", rate["timestamp"]
                                            ),
                                            "data_type": "funding_rate",
                                        }
                                    )

                            print(
                                f"✅ Downloaded {len(funding_rates)} funding rate records via fetch_funding_rate_history"
                            )
                            logger.info(
                                f"✅ Downloaded {len(funding_rates)} funding rate records via fetch_funding_rate_history"
                            )

                        except Exception as e:
                            print(f"⚠️ fetch_funding_rate_history failed: {e}")
                            logger.warning(f"⚠️ fetch_funding_rate_history failed: {e}")

                    # Method 2: Try fetch_funding_rate if available
                    if not all_futures_data and hasattr(
                        self.exchange, "fetch_funding_rate"
                    ):
                        try:
                            funding_rate = await self.exchange.fetch_funding_rate(
                                self.config.symbol
                            )

                            if (
                                funding_rate
                                and start_ms <= funding_rate["timestamp"] <= end_ms
                            ):
                                all_futures_data.append(
                                    {
                                        "timestamp": funding_rate["timestamp"],
                                        "funding_rate": funding_rate.get(
                                            "fundingRate", 0
                                        ),
                                        "funding_time": funding_rate.get(
                                            "fundingTime", funding_rate["timestamp"]
                                        ),
                                        "data_type": "funding_rate",
                                    }
                                )

                            print(
                                f"✅ Downloaded current funding rate via fetch_funding_rate"
                            )
                            logger.info(
                                f"✅ Downloaded current funding rate via fetch_funding_rate"
                            )

                        except Exception as e:
                            print(f"⚠️ fetch_funding_rate failed: {e}")
                            logger.warning(f"⚠️ fetch_funding_rate failed: {e}")

                    # Method 3: Try direct API call for historical funding rates
                    if not all_futures_data and hasattr(
                        self.exchange, "fapiPublicGetFundingRate"
                    ):
                        try:
                            # Direct API call to Binance futures funding rate endpoint
                            params = {
                                "symbol": self.config.symbol.upper(),
                                "startTime": start_ms,
                                "endTime": end_ms,
                                "limit": 1000,
                            }

                            funding_rates = (
                                await self.exchange.fapiPublicGetFundingRate(params)
                            )

                            for rate in funding_rates:
                                all_futures_data.append(
                                    {
                                        "timestamp": int(rate["fundingTime"]),
                                        "funding_rate": float(rate["fundingRate"]),
                                        "funding_time": int(rate["fundingTime"]),
                                        "data_type": "funding_rate",
                                    }
                                )

                            print(
                                f"✅ Downloaded {len(funding_rates)} funding rate records via direct API"
                            )
                            logger.info(
                                f"✅ Downloaded {len(funding_rates)} funding rate records via direct API"
                            )

                        except Exception as e:
                            print(f"⚠️ Direct API call failed: {e}")
                            logger.warning(f"⚠️ Direct API call failed: {e}")

                except Exception as e:
                    print(f"⚠️ Warning: Could not download funding rates: {e}")
                    logger.warning(f"⚠️ Warning: Could not download funding rates: {e}")

                if not all_futures_data:
                    print(
                        f"⚠️ No funding rate data found for {start_dt.strftime('%Y-%m')}"
                    )
                    logger.warning(
                        f"⚠️ No funding rate data found for {start_dt.strftime('%Y-%m')}"
                    )
                    return False

                # Process and save data
                df = self._process_futures_data(all_futures_data)
                df.to_csv(filepath, index=False)

                print(f"✅ Saved {len(df)} funding rate records to {filename}")
                logger.info(f"✅ Saved {len(df)} funding rate records to {filename}")
                return True

            except Exception as e:
                print(
                    f"❌ Error downloading funding rates for {start_dt.strftime('%Y-%m')}: {e}"
                )
                logger.exception(
                    f"❌ Error downloading funding rates for {start_dt.strftime('%Y-%m')}: {e}"
                )
                return False

    def _process_klines_data(self, klines: list[list]) -> pd.DataFrame:
        """Process klines data from CCXT into a DataFrame."""
        if not klines:
            return pd.DataFrame()

        # CCXT format: [timestamp, open, high, low, close, volume, ...]
        data = []
        for kline in klines:
            try:
                data.append(
                    {
                        "timestamp": datetime.fromtimestamp(kline[0] / 1000),
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                    }
                )
            except (IndexError, ValueError, TypeError) as e:
                print(f"⚠️ Error processing kline: {e}, kline: {kline}")
                continue

        return pd.DataFrame(data)

    def _process_aggtrades_data(self, trades: list[dict]) -> pd.DataFrame:
        """Process trades data from CCXT into a DataFrame."""
        if not trades:
            return pd.DataFrame()

        data = []
        for trade in trades:
            try:
                data.append(
                    {
                        "timestamp": datetime.fromtimestamp(trade["timestamp"] / 1000),
                        "price": float(trade["price"]),
                        "quantity": float(trade["amount"]),
                        "is_buyer_maker": trade.get("side", "unknown") == "sell",
                        "agg_trade_id": str(trade.get("id", "")),
                    }
                )
            except (KeyError, ValueError, TypeError) as e:
                print(f"⚠️ Error processing trade: {e}, trade: {trade}")
                continue

        return pd.DataFrame(data)

    def _process_futures_data(self, futures_data: list[dict]) -> pd.DataFrame:
        """Process futures data (funding rates) from CCXT into a DataFrame."""
        if not futures_data:
            return pd.DataFrame()

        # Funding rate data structure - the data is already in the correct format
        data = []
        for item in futures_data:
            try:
                # The funding rate data is already properly structured
                data.append(
                    {
                        "timestamp": datetime.fromtimestamp(item["timestamp"] / 1000),
                        "funding_rate": float(item["funding_rate"]),
                        "funding_time": datetime.fromtimestamp(
                            item["funding_time"] / 1000
                        ),
                    }
                )
            except (KeyError, ValueError, TypeError) as e:
                print(f"⚠️ Error processing futures item: {e}, item: {item}")
                continue

        return pd.DataFrame(data)

    async def run_clean_download(self) -> bool:
        """Run the complete clean download process."""
        start_time = time.time()

        try:
            print("=" * 80)
            print("🚀 CLEAN DATA DOWNLOAD PROCESS STARTED")
            print("=" * 80)
            logger.info("🚀 CLEAN DATA DOWNLOAD PROCESS STARTED")

            # Step 1: Initialize
            print("🔧 STEP 1: Initializing...")
            logger.info("🔧 STEP 1: Initializing...")

            if not await self.initialize():
                print(failed("INITIALIZATION FAILED - Aborting download process"))
                print(failed("❌ INITIALIZATION FAILED - Aborting download process"))
                return False

            print("✅ INITIALIZATION COMPLETED SUCCESSFULLY")
            print("🔍 DEBUG: Exchange client ready:", self.exchange is not None)
            print("🔍 DEBUG: Cache directory exists:", os.path.exists(self.cache_dir))
            print(
                "🔍 DEBUG: Cache directory contents:",
                len(os.listdir(self.cache_dir))
                if os.path.exists(self.cache_dir)
                else "N/A",
            )

            print("🔄 STEP 4: Starting parallel downloads for all data types...")
            logger.info("🔄 STEP 4: Starting parallel downloads for all data types...")

            # Download all data types in parallel
            download_tasks = [
                self.download_klines_parallel(),
                self.download_aggtrades_parallel(),
                self.download_futures_parallel(),
            ]

            print("📋 Created 3 parallel download tasks:")
            print("📈 Task 1: Klines data")
            print("📊 Task 2: Aggregated trades data")
            print("📈 Task 3: Futures data")
            logger.info("📋 Created 3 parallel download tasks")

            print("⏳ Executing all tasks concurrently...")
            logger.info("⏳ Executing all tasks concurrently...")

            results = await asyncio.gather(*download_tasks, return_exceptions=True)

            print("📊 Processing final results...")
            logger.info("📊 Processing final results...")

            # Process results
            success_count = 0
            error_count = 0
            for i, result in enumerate(results):
                data_types = ["Klines", "Aggtrades", "Futures"]
                if isinstance(result, Exception):
                    error_count += 1
                    print(f"❌ {data_types[i]} download failed: {result}")
                    print(failed(f"❌ {data_types[i]} download failed: {result}"))
                elif result:
                    success_count += 1
                    print(f"✅ {data_types[i]} download completed successfully")
                    logger.info(f"✅ {data_types[i]} download completed successfully")

            # Calculate statistics
            self.stats["total_time"] = time.time() - start_time

            print("=" * 80)
            print("🎉 CLEAN DOWNLOAD PROCESS COMPLETED")
            print("=" * 80)
            print("📊 FINAL STATISTICS:")
            print(f"✅ Successful downloads: {success_count}/3")
            print(f"❌ Failed downloads: {error_count}/3")
            print(f"⏱️ Total time: {self.stats['total_time']:.2f} seconds")
            print(f"📈 Klines files downloaded: {self.stats['klines_downloaded']}")
            print(
                f"📊 Aggtrades files downloaded: {self.stats['aggtrades_downloaded']}"
            )
            print(f"📈 Futures files downloaded: {self.stats['futures_downloaded']}")
            print(f"❌ Total errors: {self.stats['errors']}")
            print()
            print("📁 CSV FILES CREATED:")
            total_files = (
                self.stats["klines_downloaded"]
                + self.stats["aggtrades_downloaded"]
                + self.stats["futures_downloaded"]
            )
            print(f"📈 Monthly klines files: {self.stats['klines_downloaded']}")
            print(f"📊 Daily aggtrades files: {self.stats['aggtrades_downloaded']}")
            print(f"📈 Monthly futures files: {self.stats['futures_downloaded']}")
            print(f"📁 Total CSV files: {total_files}")
            print("=" * 80)

            logger.info("🎉 CLEAN DOWNLOAD PROCESS COMPLETED")
            logger.info("📊 FINAL STATISTICS:")
            logger.info(f"✅ Successful downloads: {success_count}/3")
            logger.info(f"❌ Failed downloads: {error_count}/3")
            logger.info(f"⏱️ Total time: {self.stats['total_time']:.2f} seconds")
            logger.info(
                f"📈 Klines files downloaded: {self.stats['klines_downloaded']}"
            )
            logger.info(
                f"📊 Aggtrades files downloaded: {self.stats['aggtrades_downloaded']}"
            )
            logger.info(
                f"📈 Futures files downloaded: {self.stats['futures_downloaded']}"
            )
            logger.info(f"❌ Total errors: {self.stats['errors']}")

            return success_count > 0

        except Exception:
            print(critical("CRITICAL ERROR in clean download: {e}"))
            print(critical("❌ CRITICAL ERROR in clean download: {e}"))
            return False
        finally:
            print("🧹 Cleaning up resources...")
            logger.info("🧹 Cleaning up resources...")
            await self.cleanup()
            print("✅ Cleanup completed")
            logger.info("✅ Cleanup completed")


async def main():
    """Main function for the clean data downloader."""
    parser = argparse.ArgumentParser(
        description="Clean data downloader for Ares trading bot"
    )
    parser.add_argument(
        "--symbol", type=str, required=True, help="Trading symbol (e.g., ETHUSDT)"
    )
    parser.add_argument(
        "--exchange", type=str, required=True, help="Exchange name (e.g., BINANCE)"
    )
    parser.add_argument(
        "--interval", type=str, default="1m", help="K-line interval (default: 1m)"
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=2,
        help="Years of data to download (default: 2)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent downloads (default: 5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if files already exist",
    )

    args = parser.parse_args()

    # Create configuration
    config = DownloadConfig(
        symbol=args.symbol,
        exchange=args.exchange,
        interval=args.interval,
        lookback_years=args.lookback_years,
        max_concurrent_downloads=args.max_concurrent,
        force=args.force,
    )

    # Create and run downloader
    downloader = CleanDataDownloader(config)
    success = await downloader.run_clean_download()

    if success:
        logger.info("✅ Clean download completed successfully")
        sys.exit(0)
    else:
        print(failed("❌ Clean download failed"))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
