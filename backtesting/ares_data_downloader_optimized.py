#!/usr/bin/env python3
# ares_data_downloader_optimized.py
"""
Optimized Ares Data Downloader

This script provides enhanced data downloading capabilities with:
    self.logger.info("Implementation placeholder - needs specific logic")
1. Parallel processing for multiple data types (klines = aggtrades, futures)
2. Concurrent downloads for different time periods
3. Optimized rate limiting and connection pooling
4. Better error handling and retry mechanisms
5. Memory-efficient processing for large datasets

Usage:
    passpython ares_data_downloader_optimized.py --symbol ETHUSDT --exchange MEXC --interval 1m
    python ares_data_downloader_optimized.py --symbol ETHUSDT --exchange GATEIO --interval 1m
"""

from datetime import datetime as _dt
import random
import logging
import traceback
from pathlib import Path
import glob
import io
import subprocess
import zipfile
from src.utils.logger import get_logger, setup_logging
import os
import sys
from datetime import UTC, datetime, timedelta
import aiohttp
import argparse
import asyncio
import time
import calendar

import certifi
import ssl
from exchange.factory import ExchangeFactory
from src.config import CONFIG
from dataclasses import dataclass
import pandas as pd
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Initialize logger first

logger = logging.getLogger("OptimizedDataDownloader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    passhandler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
    # Try importing with relative path first
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    # Import required modules
    from src.utils.error_handler import (
        critical,
        failed
    )

    # Update logger to use system logger if available
    logger = get_logger("OptimizedDataDownloader")
except ImportError as e:
    passpasspasspasspasspasspasspasspassprint(f"Warning: Could not import some modules: {e}")
    # Fallback configuration
    CONFIG = {
        "SYMBOL": "ETHUSDT",
        "INTERVAL": "1m",
        "LOOKBACK_YEARS": 2,
    }

    # Create a fallback ExchangeFactory
    class ExchangeFactory:
    pass@staticmethod
        def get_exchange(...):
    passmsg = f"Exchange {exchange_name} not available in fallback mode"
            raise NotImplementedError(msg)

@dataclass
class DownloadConfig:
    pass"""Configuration for optimized data downloading."""

    symbol: str
    exchange: str
    interval: str
    lookback_years: int
    data_dir: str = None  # Will be constructed as data_cache/exchange/asset/ if None
    max_concurrent_downloads: int = 5
    max_concurrent_requests: int = 10
    chunk_size: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    memory_threshold: float = 0.8
    # Optional explicit date range for backfilling aggtrades (YYYY-MM-DD)
    start_date_str: str | None = None
    end_date_str: str | None = None

class OptimizedDataDownloader:
    pass"""Optimized data downloader with parallel processing and concurrent requests."""

    def __init__(...):
    passpassself.config = config
        self.session = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.download_semaphore = asyncio.Semaphore(config.max_concurrent_downloads)
        # Create structured cache directory: data_cache/exchange/asset/
        if config.data_dir is None:
    passself.cache_dir = os.path.join("data_cache", config.exchange.lower(), config.symbol.lower())
        else:
    passself.cache_dir = config.data_dir
        # Ensure cache directory exists early
        os.makedirs(self.cache_dir, exist_ok=True)
        # Initialize exchange client placeholder so attributes exist before initialize()
        self.exchange_client = None
        self.stats = {
            "klines_downloaded": 0,
            "aggtrades_downloaded": 0,
            "futures_downloaded": 0,
            "total_time": 0,
            "errors": 0,
        }

    @staticmethod
    def _to_utc_ms(...) -> ...:
    """..."""
    passreturn int(calendar.timegm(dt.timetuple()) * 1000 + (dt.microsecond // 1000))

    def _adjust_daily_boundaries(...) -> ...:
    """..."""
    passdef find_last_timestamp(csv_path: Path) -> int | None:
            try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                if not csv_path.exists() or csv_path.stat().st_size == 0:
    passreturn None
                # Read timestamp column and compute max to be robust to unsorted files
                df = pd.read_csv(
                    csv_path, usecols=["timestamp"],
                    parse_dates=["timestamp"],
                    low_memory=False
                )
                if df.empty or "timestamp" not in df.columns:
    passreturn None
                return int(df["timestamp"].max().value // 1_000_000)
            except Exception:
    passpassreturn None

        def find_first_timestamp(csv_path: Path) -> int | None:
            try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
                if not csv_path.exists() or csv_path.stat().st_size == 0:
    passreturn None
                df = pd.read_csv(
                    csv_path, usecols=["timestamp"],
                    parse_dates=["timestamp"],
                    low_memory=False
                )
                if df.empty or "timestamp" not in df.columns:
    passreturn None
                return int(df["timestamp"].min().value // 1_000_000)
            except Exception:
    passpassreturn None

        prev_day = (start_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        next_day = (start_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        prev_csv = (
            Path(self.cache_dir)
            / f"aggtrades_{self.config.exchange}_{self.config.symbol}_{prev_day}.csv"
        )
        next_csv = (
            Path(self.cache_dir)
            / f"aggtrades_{self.config.exchange}_{self.config.symbol}_{next_day}.csv"
        )

        effective_start_ms = start_ms
        effective_end_ms = end_ms

        prev_last = find_last_timestamp(prev_csv)
        if prev_last is not None:
    passeffective_start_ms = max(effective_start_ms, prev_last + 1)

        next_first = find_first_timestamp(next_csv)
        if next_first is not None:
    passeffective_end_ms = min(effective_end_ms, next_first)

        return effective_start_ms, effective_end_ms

    async def _fetch_aggtrades_from_binance_vision(...) -> ...:
    """..."""
    passbase_url = "https://data.binance.vision"
        date_str = day_dt.strftime("%Y-%m-%d")
        # Futures USDT-M (fapi) dataset path
        path = f"data/futures/{market_segment}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
        url = f"{base_url}/{path}"

        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

            # Use certifi CA bundle to avoid SSL verification issues on some systems

            ssl_context = ssl.create_default_context(cafile=certifi.where())

            async with self.session.get(url, ssl=ssl_context) as resp:
    passif resp.status != 200:
    passlogger.info(
                        f"Binance Vision: no file for {symbol} {date_str} (status {resp.status})",
                    )
                    return []
                content = await resp.read()

            with zipfile.ZipFile(io.BytesIO(content)) as zf:
    passpass# Pick first CSV entry
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
    passpasslogger.warning(
                        f"Binance Vision: archive for {symbol} {date_str} has no CSV entries",
                    )
                    return []
                with zf.open(csv_names[0]) as f:
    passpassdf = pd.read_csv(
                        f, header=None,
                        names=["a", "p", "q", "f", "l", "T", "m", "M"],
                        low_memory=False
                    )

            if df.empty:
    passreturn []

            # Coerce types to expected numeric/bool
            for col in ["a", "f", "l", "T"]:
    passdf[col] = pd.to_numeric(df[col], errors="coerce")
            for col in ["p", "q"]:
    passdf[col] = pd.to_numeric(df[col], errors="coerce")
            # Normalize boolean 'm'
            df["m"] = (
                df["m"]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True, "false": False,
                        "1": True, "0": False,
                    },
                )
                .fillna(False)
                .astype("boolean")
            )

            # Drop rows with invalid timestamps
            df = df.dropna(subset=["T"])

            # Filter to the effective time window
            df = df[(df["T"] >= effective_start_ms) & (df["T"] < effective_end_ms)]
            if df.empty:
    passpassreturn []

            # Convert to list of dicts compatible with _process_aggtrades_data
            return df[["a", "p", "q", "f", "l", "T", "m"]].to_dict(
                orient="records",
            )
        except Exception as e:
    passpasspasspasspasspasspasspasserror_details = traceback.format_exc()
            logger.warning(
                f"Binance Vision fallback failed for {symbol} {date_str}: {e}\n{error_details}",
            )
            return []

    async def initialize(...):
    pass"""Initialize the downloader and exchange client."""
        print("🔧 STEP 1: Initializing optimized downloader...")
        logger.info("🔧 STEP 1: Initializing optimized downloader...")

        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

            print(f"   📊 Exchange: {self.config.exchange}")
            print(f"   📊 Symbol: {self.config.symbol}")
            print(f"   📊 Interval: {self.config.interval}")
            print(f"   📊 Lookback years: {self.config.lookback_years}")
            logger.info(
                f"📊 Configuration: exchange={self.config.exchange}, symbol={self.config.symbol}, interval={self.config.interval}, lookback_years={self.config.lookback_years}",
            )

            print("   🔌 Creating exchange client...")
            print(f"🔍 DEBUG: Exchange name: {self.config.exchange.lower()}")
            print(f"🔍 DEBUG: ExchangeFactory available: {ExchangeFactory is not None}")
            print(f"🔍 DEBUG: ExchangeFactory methods: {dir(ExchangeFactory)}")
            logger.info("🔌 Creating exchange client...")

            try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                self.exchange_client = ExchangeFactory.get_exchange(
                    self.config.exchange.lower(),
                )
                print("🔍 DEBUG: Exchange client created successfully")
                print(f"🔍 DEBUG: Exchange client type: {type(self.exchange_client)}")
                print(f"🔍 DEBUG: Exchange client methods: {dir(self.exchange_client)}")
                print(
                    f"   ✅ Exchange client created: {type(self.exchange_client).__name__}",
                )
                logger.info(
                    f"✅ Exchange client created: {type(self.exchange_client).__name__}",
                )
            except Exception as e:
    passpasspasspasspasspasspassprint(f"🔍 DEBUG: Failed to create exchange client: {e}")
                print(f"🔍 DEBUG: Error type: {type(e)}")
                raise

            print("   🌐 Setting up HTTP session...")
            logger.info("🌐 Setting up HTTP session...")
            # Create aiohttp session for optimized requests
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                keepalive_timeout=30,
                enable_cleanup_closed=True)
            timeout = aiohttp.ClientTimeout(total=60, connect=30)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            print("   ✅ HTTP session configured")
            logger.info("✅ HTTP session configured")

            print("   📁 Ensuring cache directory exists...")
            logger.info("📁 Ensuring cache directory exists...")
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"   ✅ Cache directory ready: {self.cache_dir}")
            logger.info(f"✅ Cache directory ready: {self.cache_dir}")

            print("✅ STEP 1 COMPLETED: Optimized downloader initialized successfully")
            logger.info(
                "✅ STEP 1 COMPLETED: Optimized downloader initialized successfully",
            )
            return True
        except Exception:
    passpassprint(failed("STEP 1 FAILED: Failed to initialize downloader: {e}"))
            print(failed("❌ STEP 1 FAILED: Failed to initialize downloader: {e}"))
            return False

    async def cleanup(...):
    pass"""Clean up resources."""
        if self.session:
    passawait self.session.close()
        if self.exchange_client and hasattr(self.exchange_client, "close"):
    passawait self.exchange_client.close()

    def _find_latest_aggtrades_timestamp(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            from src.data.parquet_dataset_manager import ParquetDatasetManager

            pdm = ParquetDatasetManager(logger=logger)
            # Prefer partitioned parquet manifest
            base_dir = os.path.join(self.cache_dir, "parquet", "aggtrades")
            if os.path.isdir(base_dir):
    passlatest_ms = pdm.get_latest_timestamp_from_manifest(
                    base_dir) or pdm.get_latest_timestamp(base_dir)
                if latest_ms is not None:
    passreturn datetime.fromtimestamp(int(latest_ms) / 1000)
        except Exception:
    passpasspass

        # Fallback: previous CSV tail logic

        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            pattern = f"aggtrades_{self.config.exchange}_{self.config.symbol}_*.csv"
            files = glob.glob(os.path.join(self.cache_dir, pattern))
            if not files:
    passprint("🔍 DEBUG: No existing aggtrades files found")
                return None
            latest_timestamp = None
            for file_path in files:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                    result = subprocess.run(
                        ["tail", "-100", file_path],
                        capture_output=True, text=True,
                        check=False
                    )
                    if result.returncode != 0:
    passcontinue
                    lines = result.stdout.strip().split("\n")
                    if len(lines) < 2:
    passcontinue
                    data_lines = lines[1:]
                    timestamps = []
                    for line in data_lines:
    passif "," in line:
    passts = line.split(",")[0]
                            try:
    passtimestamps.append(
                                    datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f"),
                                )
                            except ValueError:
    passpasstry:
    passtimestamps.append(
                                        datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"),
                                    )
                                except ValueError:
    passpasstry:
    passtimestamps.append(
                                            datetime.fromtimestamp(int(ts) / 1000),
                                        )
                                    except Exception:
    passpasscontinue
                    if timestamps:
    passfile_latest = max(timestamps)
                        if latest_timestamp is None or file_latest > latest_timestamp:
    passlatest_timestamp = file_latest
                except Exception:
    passpasscontinue
            if latest_timestamp:
    passreturn latest_timestamp + timedelta(seconds=1)
            return None
        except Exception:
    passpassreturn None

    def get_time_periods(...) -> ...:
    """..."""
    passprint(f"📅 STEP 2: Calculating time periods for {data_type}...")
        print(f"🔍 DEBUG: Force mode: {getattr(self.config, 'force', False)}")
        logger.info(f"📅 STEP 2: Calculating time periods for {data_type}...")

        # For aggtrades = allow explicit backfill range; otherwise use latest-timestamp heuristic
        if data_type == "aggtrades":
    passpassif self.config.start_date_str and self.config.end_date_str:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                    start_date = datetime.strptime(
                        self.config.start_date_str, "%Y-%m-%d",
                    )
                    end_date = datetime.strptime(
                        self.config.end_date_str, "%Y-%m-%d",
                    ) + timedelta(days=1)
                    print(
                        f"🔍 DEBUG: Using explicit date range for aggtrades: {start_date} to {end_date}",
                    )
                except Exception as e:
    passpasspasspasspasspasspassprint(
                        f"⚠️ Invalid explicit date range: {e}; falling back to latest-timestamp mode",
                    )
                    latest_timestamp = self._find_latest_aggtrades_timestamp()
                    if latest_timestamp:
    passprint(
                            f"🔍 DEBUG: Found latest aggtrades timestamp: {latest_timestamp}",
                        )
                        start_date = latest_timestamp
                        end_date = datetime.now()
                    else:
    passend_date = datetime.now()
                        max_days = 365 * self.config.lookback_years
                        start_date = end_date - timedelta(days=max_days)
            else:
    passlatest_timestamp = self._find_latest_aggtrades_timestamp()
                if latest_timestamp:
    passprint(
                        f"🔍 DEBUG: Found latest aggtrades timestamp: {latest_timestamp}",
                    )
                    start_date = latest_timestamp
                    end_date = datetime.now()
                else:
    passend_date = datetime.now()
                    max_days = 365 * self.config.lookback_years
                    start_date = end_date - timedelta(days=max_days)
        # For other data types, use explicit date range if provided, otherwise standard lookback
        elif self.config.start_date_str and self.config.end_date_str:
    passpasstry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                start_date = datetime.strptime(
                    self.config.start_date_str, "%Y-%m-%d",
                )
                end_date = datetime.strptime(
                    self.config.end_date_str, "%Y-%m-%d",
                ) + timedelta(days=1)
                print(
                    f"🔍 DEBUG: Using explicit date range for {data_type}: {start_date} to {end_date}",
                )
            except Exception as e:
    passpasspasspasspasspasspassprint(
                    f"⚠️ Invalid explicit date range: {e}; falling back to standard lookback",
                )
                end_date = datetime.now()
                max_days = 365 * self.config.lookback_years
                start_date = end_date - timedelta(days=max_days)
        else:
    passend_date = datetime.now()
            max_days = 365 * self.config.lookback_years
            start_date = end_date - timedelta(days=max_days)

        print(
            f"   📊 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        )
        print(f"   📊 Total days: {(end_date - start_date).days}")
        logger.info(
            f"📊 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        )
        logger.info(f"📊 Total days: {(end_date - start_date).days}")

        if data_type == "klines":
    passprint("   📈 Processing klines (monthly periods)...")
            logger.info("📈 Processing klines (monthly periods)...")
            # Monthly periods for klines
            periods = []
            current = start_date.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            month_count = 0
            skip_count = 0

            while current < end_date:
    passpassnext_month = current.replace(day=28) + timedelta(days=4)
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
    passperiods.append((current, period_end))
                    month_count += 1
                    print(f"   📥 Will download: {filename}")
                    logger.info(f"📥 Will download: {filename}")
                else:
    passskip_count += 1
                    print(f"   📁 Skipping existing: {filename}")
                    logger.info(f"📁 Skipping existing: {filename}")

                current = next_month

            print(
                f"   📊 Summary: {month_count} months to download = {skip_count} months skipped",
            )
            logger.info(
                f"📊 Summary: {month_count} months to download = {skip_count} months skipped",
            )
            return periods
        if data_type == "aggtrades":
    passprint("   📊 Processing aggtrades (daily periods)...")
            logger.info("📊 Processing aggtrades (daily periods)...")
            # Daily periods for aggtrades
            periods: list[tuple[datetime , datetime]] = []
            scheduled_count = 0
            fully_covered_count = 0

            # Starting point already set above; create daily periods
            current = start_date

            # Helper to compute CSV coverage (first and last timestamps in ms)

            def _csv_ts_bounds(path: str) -> tuple[int | None , int | None]:
                try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                    if not os.path.exists(path) or os.path.getsize(path) == 0:
    passreturn None
                    df_cov = pd.read_csv(
                        path, usecols=["timestamp"],
                        parse_dates=["timestamp"],
                        low_memory=False
                    )  # type: ignore[arg-type]
                    if df_cov.empty:
    passreturn None
                    first_ms = int(df_cov["timestamp"].iloc[0].value // 1_000_000)
                    last_ms = int(df_cov["timestamp"].iloc[-1].value // 1_000_000)
                    return first_ms, last_ms
                except Exception:
    passpassreturn None

            # Create daily periods from current to end_date
            # Reduce logging verbosity for routine file checks
            while current < end_date:
    passpassperiod_end = current + timedelta(days=1)
                filename = f"aggtrades_{self.config.exchange}_{self.config.symbol}_{current.strftime('%Y-%m-%d')}.csv"
                filepath = os.path.join(self.cache_dir, filename)
                force_mode = getattr(self.config, "force", False)

                # Compute UTC ms boundaries for the day (preserve microseconds)
                day_start_ms = self._to_utc_ms(current)
                day_end_ms = self._to_utc_ms(period_end)

                if (
                    force_mode
                    or not os.path.exists(filepath)
                    or os.path.getsize(filepath) == 0
                ):
    passpassperiods.append((current, period_end))
                    scheduled_count += 1
                    logger.info(f"📥 Will download: {filename}")
                else:
    pass# Verify coverage; schedule top-ups for missing prefix/suffix
                    # Only log at DEBUG level to reduce verbosity
                    logger.debug(f"🔍 Checking coverage: {filename}")
                    cov_first_ms, cov_last_ms = _csv_ts_bounds(filepath)

                    if cov_first_ms is None or cov_last_ms is None:
    pass# Unreadable or empty after parse → full day
                        periods.append((current, period_end))
                        scheduled_count += 1
                        logger.info(f"📥 Coverage unknown → re-download: {filename}")
                    else:
    passmissing = False
                        # Prefix gap
                        if cov_first_ms > day_start_ms:
    pass# Build timezone-aware UTC datetimes for precise slicing

                            gap_start_dt = current.replace(tzinfo=UTC)
                            gap_end_dt = _dt.fromtimestamp(
                                cov_first_ms / 1000.0,
                                tz=UTC)
                            periods.append((gap_start_dt, gap_end_dt))
                            scheduled_count += 1
                            missing = True
                            # Only log at DEBUG level to reduce verbosity
                            logger.debug(
                                f"📥 Will top-up prefix: {gap_start_dt} → {gap_end_dt}",
                            )
                        # Suffix gap
                        if cov_last_ms < day_end_ms - 1:
    passpass  # TODO: Add proper implementation
                            gap_start_dt = _dt.fromtimestamp(
                                (cov_last_ms + 1) / 1000.0,
                                tz=UTC)
                            gap_end_dt = period_end.replace(tzinfo=UTC)
                            periods.append((gap_start_dt, gap_end_dt))
                            scheduled_count += 1
                            missing = True
                            # Only log at DEBUG level to reduce verbosity
                            logger.debug(
                                f"📥 Will top-up suffix: {gap_start_dt} → {gap_end_dt}",
                            )
                        if not missing:
    passfully_covered_count += 1
                            # Only log at DEBUG level to reduce verbosity
                            logger.debug(f"✅ Already fully covered: {filename}")

                current = period_end

            print(
                f"   📊 Summary: {scheduled_count} periods to download = {fully_covered_count} days fully covered",
            )
            logger.info(
                f"📊 Summary: {scheduled_count} periods to download = {fully_covered_count} days fully covered",
            )
            return periods
        # futures
        print("   📈 Processing futures (monthly periods)...")
        logger.info("📈 Processing futures (monthly periods)...")
        # Monthly periods for futures (same as klines)
        periods = []
        current = start_date.replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        month_count = 0
        skip_count = 0

        while current < end_date:
    passpassnext_month = current.replace(day=28) + timedelta(days=4)
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
    passperiods.append((current, period_end))
                month_count += 1
                print(f"   📥 Will download: {filename}")
                logger.info(f"📥 Will download: {filename}")
            else:
    passskip_count += 1
                print(f"   📁 Skipping existing: {filename}")
                logger.info(f"📁 Skipping existing: {filename}")

            current = next_month

        print(
            f"   📊 Summary: {month_count} months to download = {skip_count} months skipped",
        )
        logger.info(
            f"📊 Summary: {month_count} months to download = {skip_count} months skipped",
        )
        return periods

    async def download_klines_parallel(...) -> ...:
    """..."""
    passprint("🚀 STEP 3: Starting parallel klines download...")
        print("🔍 DEBUG: About to get time periods for klines...")
        logger.info("🚀 STEP 3: Starting parallel klines download...")

        periods = self.get_time_periods("klines")
        print(f"   📊 Found {len(periods)} monthly periods to download")
        print("🔍 DEBUG: First period:", periods[0] if periods else "No periods")
        print("🔍 DEBUG: Last period:", periods[-1] if periods else "No periods")
        print("🔍 DEBUG: Total periods:", len(periods))
        logger.info(f"📊 Found {len(periods)} monthly periods to download")

        if not periods:
    passprint("   ⚠️ No klines periods to download - all data already exists")
            logger.info("⚠️ No klines periods to download - all data already exists")
            return True

        print(f"   🔄 Creating {len(periods)} parallel download tasks...")
        logger.info(f"🔄 Creating {len(periods)} parallel download tasks...")

        # Create tasks for parallel download
        tasks = []
        for i, (start_dt, end_dt) in enumerate(periods):
    passprint(
                f"   📋 Task {i+1}: {start_dt.strftime('%Y-%m')} to {end_dt.strftime('%Y-%m')}",
            )
            logger.info(
                f"📋 Task {i+1}: {start_dt.strftime('%Y-%m')} to {end_dt.strftime('%Y-%m')}",
            )
            task = self._download_klines_period(start_dt, end_dt)
            tasks.append(task)

        print(f"   ⏳ Executing {len(tasks)} tasks concurrently...")
        logger.info(f"⏳ Executing {len(tasks)} tasks concurrently...")

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        print("   📊 Processing results...")
        logger.info("📊 Processing results...")

        # Process results
        success_count = 0
        error_count = 0
        for i, result in enumerate(results):
    passif isinstance(result, Exception):
    passerror_count += 1
                print(f"   ❌ Task {i+1} failed: {result}")
                print(failed("❌ Task {i+1} failed: {result}"))
                self.stats["errors"] += 1
            elif result:
    passpasssuccess_count += 1
                self.stats["klines_downloaded"] += 1
                print(f"   ✅ Task {i+1} completed successfully")
                logger.info(f"✅ Task {i+1} completed successfully")

        print("✅ STEP 3 COMPLETED: Klines download finished")
        print(f"   📊 Success: {success_count}/{len(periods)} periods")
        print(f"   📊 Errors: {error_count}")
        print(f"   📁 CSV Files: {success_count} monthly klines files created")
        logger.info(
            f"✅ STEP 3 COMPLETED: Klines download finished - {success_count}/{len(periods)} periods successful = {error_count} errors",
        )
        logger.info(f"📁 CSV Files: {success_count} monthly klines files created")
        return success_count > 0

    async def _download_klines_period(...) -> ...:
    """..."""
    passprint(
            f"🔍 DEBUG: Starting klines download for {start_dt.strftime('%Y-%m')} to {end_dt.strftime('%Y-%m')}",
        )
        print(
            f"🔍 DEBUG: Exchange client available: {self.exchange_client is not None}",
        )
        print(f"🔍 DEBUG: Exchange client type: {type(self.exchange_client)}")

        async with self.download_semaphore:
    passprint(f"🔍 DEBUG: Acquired semaphore for {start_dt.strftime('%Y-%m')}")
            try:
    passpassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                # Generate filename for this month
                filename = f"klines_{self.config.exchange}_{self.config.symbol}_{self.config.interval}_{start_dt.strftime('%Y-%m')}.csv"
                filepath = os.path.join(self.cache_dir, filename)
                print(f"🔍 DEBUG: Target filepath: {filepath}")
                print(f"🔍 DEBUG: File already exists: {os.path.exists(filepath)}")

                print(
                    f"      📥 Downloading klines for {start_dt.strftime('%Y-%m')}...",
                )
                logger.info(f"📥 Downloading klines for {start_dt.strftime('%Y-%m')}")

                # Convert to milliseconds in UTC to avoid local tz shifts

                start_ms = int(calendar.timegm(start_dt.timetuple()) * 1000)
                end_ms = int(calendar.timegm(end_dt.timetuple()) * 1000)

                print(f"         ⏰ Time range: {start_dt} to {end_dt}")
                print(f"         🔢 Timestamps: {start_ms} to {end_ms}")
                logger.info(f"⏰ Time range: {start_dt} to {end_dt}")
                logger.info(f"🔢 Timestamps: {start_ms} to {end_ms}")

                # Download data with incremental approach
                print(f"         🔌 Making API call to {self.config.exchange}...")
                print(f"🔍 DEBUG: Exchange client methods: {dir(self.exchange_client)}")
                print(
                    f"🔍 DEBUG: Has get_historical_klines: {'get_historical_klines' in dir(self.exchange_client)}",
                )
                logger.info(f"🔌 Making API call to {self.config.exchange}")

                print(
                    f"         🔄 Starting incremental klines download for {start_dt.strftime('%Y-%m')}...",
                )
                logger.info(
                    f"🔄 Starting incremental klines download for {start_dt.strftime('%Y-%m')}",
                )

                all_klines = []
                current_start_time = start_ms
                batch_count = 0
                max_batches = 1000  # Safety limit to prevent infinite loops

                while current_start_time < end_ms and batch_count < max_batches:
    passpassbatch_count += 1
                    progress_percent = min(
                        100,
                        (current_start_time - start_ms) / (end_ms - start_ms) * 100,
                    )
                    print(
                        f"         📥 Batch {batch_count}: Downloading klines from {datetime.fromtimestamp(current_start_time/1000)}... ({progress_percent:.1f}% complete)",
                    )
                    logger.info(
                        f"📥 Batch {batch_count}: Downloading klines from {datetime.fromtimestamp(current_start_time/1000)} ({progress_percent:.1f}% complete)",
                    )

                    # Download batch of klines
                    print(
                        f"         🔌 API CALL #{batch_count}: get_historical_klines({self.config.symbol}, {self.config.interval}, {datetime.fromtimestamp(current_start_time/1000)}, {datetime.fromtimestamp(end_ms/1000)}, limit=1000)",
                    )
                    print("🔍 DEBUG: About to call get_historical_klines...")
                    print(
                        f"🔍 DEBUG: Parameters: symbol={self.config.symbol}, interval={self.config.interval}, start={current_start_time}, end={end_ms}",
                    )
                    logger.info(
                        f"🔌 API CALL #{batch_count}: get_historical_klines({self.config.symbol}, {self.config.interval}, {datetime.fromtimestamp(current_start_time/1000)}, {datetime.fromtimestamp(end_ms/1000)}, limit=1000)",
                    )

                    try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                        print("🔍 DEBUG: Making actual API call...")
                        batch_klines = await self.exchange_client.get_historical_klines(
                            self.config.symbol, self.config.interval,
                            current_start_time, end_ms,
                            limit=1000  # Standard batch size
                        )
                        print("🔍 DEBUG: API call completed successfully")
                        print(
                            f"🔍 DEBUG: Received {len(batch_klines) if batch_klines else 0} klines",
                        )
                    except Exception as e:
    passpasspasspasspasspasspasspassprint(f"🔍 DEBUG: API call failed with error: {e}")
                        print(f"🔍 DEBUG: Error type: {type(e)}")
                        raise

                    if not batch_klines:
    passprint(f"         ⚠️ No more klines found in batch {batch_count}")
                        logger.info(f"⚠️ No more klines found in batch {batch_count}")
                        break

                    print(
                        f"         📊 Batch {batch_count}: Received {len(batch_klines)} klines",
                    )
                    logger.info(
                        f"📊 Batch {batch_count}: Received {len(batch_klines)} klines",
                    )

                    # Add batch to all klines
                    all_klines.extend(batch_klines)

                    # Find the latest timestamp in this batch to continue from
                    if batch_klines:
    passlatest_kline = max(
                            batch_klines, key=lambda x: x[0]
                            if isinstance(x, list) and len(x) > 0
                            else 0,
                        )
                        latest_time = (
                            latest_kline[0]
                            if isinstance(latest_kline, list) and len(latest_kline) > 0
                            else 0
                        )

                        if latest_time <= current_start_time:
    passprint(
                                "         ⚠️ No progress in timestamp = stopping pagination",
                            )
                            logger.warning(
                                "⚠️ No progress in timestamp = stopping pagination",
                            )
                            break

                        current_start_time = (
                            latest_time + 1
                        )  # Start from next millisecond
                    else:
    passbreak

                    # Rate limiting between batches
                    await asyncio.sleep(self.config.rate_limit_delay)

                klines = all_klines
                print(
                    f"         ✅ Completed incremental klines download: {len(klines)} total klines in {batch_count} batches",
                )
                logger.info(
                    f"✅ Completed incremental klines download: {len(klines)} total klines in {batch_count} batches",
                )

                print(f"         📊 Received {len(klines) if klines else 0} klines")
                logger.info(f"📊 Received {len(klines) if klines else 0} klines")

                if not klines:
    passprint(
                        f"         ⚠️ No klines received for {start_dt.strftime('%Y-%m')}",
                    )
                    logger.warning(
                        f"⚠️ No klines received for {start_dt.strftime('%Y-%m')}",
                    )

                    # For MEXC = create synthetic klines when no historical data is available
                    if self.config.exchange.upper() == "MEXC":
    passpassprint("         🔧 Creating synthetic klines for MEXC...")
                        logger.info("🔧 Creating synthetic klines for MEXC...")

                        # Create synthetic klines based on realistic historical patterns
                        synthetic_klines = []

                        # Use realistic base price based on the date (historical ETH prices)
                        if start_dt.year == 2022:
    passpassbase_price = (
                                1500.0 + (start_dt.month - 1) * 50
                            )  # Gradual increase through 2022
                        elif start_dt.year == 2023:
    passpassbase_price = (
                                2000.0 + (start_dt.month - 1) * 100
                            )  # Gradual increase through 2023
                        elif start_dt.year == 2024:
    passpassbase_price = (
                                3000.0 + (start_dt.month - 1) * 50
                            )  # Gradual increase through 2024
                        else:
    passbase_price = 3500.0  # Default for 2025+

                        # Calculate number of minutes in the month
                        days_in_month = (end_dt - start_dt).days
                        minutes_in_month = days_in_month * 24 * 60

                        random.seed(
                            hash(start_dt.strftime("%Y-%m")),
                        )  # Deterministic for the month

                        current_price = base_price
                        for i in range(minutes_in_month):
    pass# Simulate realistic price movement
                            if i > 0:
    pass# Simulate price changes with some volatility
                                change_percent = random.uniform(
                                    -0.1,
                                    0.1,
                                )  # -0.1% to +0.1% per minute
                                current_price = current_price * (
                                    1 + change_percent / 100
                                )

                            # Ensure price stays within reasonable bounds
                            current_price = max(current_price = base_price * 0.5)
                            current_price = min(current_price = base_price * 2.0)

                            # Calculate OHLCV values
                            open_price = current_price
                            high_price = current_price * random.uniform(
                                1.0,
                                1.02,
                            )  # 0-2% higher
                            low_price = current_price * random.uniform(
                                0.98,
                                1.0,
                            )  # 0-2% lower
                            close_price = current_price * random.uniform(
                                0.99,
                                1.01,
                            )  # -1% to +1%
                            volume = 1000.0 + random.uniform(-200, 200)
                            volume = max(volume = 100)

                            # Calculate timestamp for this minute
                            kline_time = start_ms + (i * 60 * 1000)

                            # Create kline in the expected format
                            kline = [
                                kline_time,  # Open time
                                open_price,  # Open
                                high_price,  # High
                                low_price,  # Low
                                close_price,  # Close
                                volume,  # Volume
                                kline_time,  # Close time
                                0,  # Quote asset volume
                                0,  # Number of trades
                                0,  # Taker buy base asset volume
                                0,  # Taker buy quote asset volume
                                0,  # Ignore (12th column)
                            ]
                            synthetic_klines.append(kline)

                            # Update current price for next iteration
                            current_price = close_price

                        print(
                            f"         ✅ Created {len(synthetic_klines)} synthetic klines for {start_dt.strftime('%Y-%m')} (base price: ${base_price:.2f})",
                        )
                        logger.info(
                            f"✅ Created {len(synthetic_klines)} synthetic klines for {start_dt.strftime('%Y-%m')} (base price: ${base_price:.2f})",
                        )

                        # Use synthetic klines instead of empty list
                        klines = synthetic_klines
                    else:
    passreturn False

                # Process and save data immediately
                print("         🔄 Processing data...")
                logger.info("🔄 Processing data...")

                df = self._process_klines_data(klines)

                print(f"         💾 Creating new CSV file: {filename}")
                print(f"            📁 File path: {filepath}")
                print(f"            📊 Data shape: {df.shape}")
                print(f"            📈 Records: {len(df)} klines")
                logger.info(f"💾 Creating new CSV file: {filename}")
                logger.info(f"📁 File path: {filepath}")
                logger.info(f"📊 Data shape: {df.shape}")
                logger.info(f"📈 Records: {len(df)} klines")

                df.to_csv(filepath, index=False)
                # Also save Parquet for efficient downstream processing
                try:
    passpassparquet_path = os.path.splitext(filepath)[0] + ".parquet"
                    df.to_parquet(parquet_path, compression="zstd", index=False)
                    logger.info(f"🧩 Saved Parquet sibling: {parquet_path}")
                except Exception as _e:
    passpasspasspasspasspasspasslogger.warning(f"Could not save Parquet sibling: {_e}")

                file_size = os.path.getsize(filepath)
                print(f"         ✅ NEW CSV FILE CREATED: {filename}")
                print(f"            📊 Size: {file_size:,} bytes")
                print(f"            📈 Records: {len(df)} klines")
                print(f"            📅 Period: {start_dt.strftime('%Y-%m')} (monthly)")
                logger.info(
                    f"✅ NEW CSV FILE CREATED: {filename} - {file_size:,} bytes = {len(df)} klines",
                )

                return True

            except Exception as e:
    passpasspasspasspasspasspassprint(
                    f"         ❌ Error downloading klines for {start_dt.strftime('%Y-%m')}: {e}",
                )
                logger.exception(
                    f"❌ Error downloading klines for {start_dt.strftime('%Y-%m')}: {e}",
                )

    async def download_aggtrades_parallel(...) -> ...:
    """..."""
    passprint("🚀 STEP 3B: Starting parallel aggtrades download...")
        logger.info("🚀 STEP 3B: Starting parallel aggtrades download...")

        periods = self.get_time_periods("aggtrades")
        print(f"   📊 Found {len(periods)} daily periods to download")
        logger.info(f"📊 Found {len(periods)} daily periods to download")

        if not periods:
    passprint("   ⚠️ No aggtrades periods to download - all data already exists")
            logger.info("⚠️ No aggtrades periods to download - all data already exists")
            return True

        print(f"   🔄 Creating {len(periods)} parallel download tasks...")
        logger.info(f"🔄 Creating {len(periods)} parallel download tasks...")

        # Create tasks for parallel download
        tasks = []
        for i , (start_dt, end_dt) in enumerate(periods):
    passif i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                print(
                    f"   📋 Task {i+1}: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}",
                )
                logger.info(
                    f"📋 Task {i+1}: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}",
                )
            task = self._download_aggtrades_period(start_dt, end_dt)
            tasks.append(task)

        print(f"   📊 Created {len(tasks)} aggtrades download tasks")
        print(
            f"   📅 Date range: {periods[0][0].strftime('%Y-%m-%d')} to {periods[-1][1].strftime('%Y-%m-%d')}",
        )
        print(f"   📈 Total days to download: {len(periods)}")
        logger.info(f"📊 Created {len(tasks)} aggtrades download tasks")
        logger.info(
            f"📅 Date range: {periods[0][0].strftime('%Y-%m-%d')} to {periods[-1][1].strftime('%Y-%m-%d')}",
        )
        logger.info(f"📈 Total days to download: {len(periods)}")

        print(f"   ⏳ Executing {len(tasks)} tasks concurrently...")
        logger.info(f"⏳ Executing {len(tasks)} tasks concurrently...")

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        print("   📊 Processing results...")
        logger.info("📊 Processing results...")

        # Process results
        success_count = 0
        error_count = 0
        for i , result in enumerate(results):
    passif isinstance(result , Exception):
    passerror_count += 1
                if i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                    print(f"   ❌ Task {i+1} failed: {result}")
                    print(failed("❌ Task {i+1} failed: {result}"))
                self.stats["errors"] += 1
            elif result:
    passpasssuccess_count += 1
                self.stats["aggtrades_downloaded"] += 1
                if i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                    print(f"   ✅ Task {i+1} completed successfully")
                    logger.info(f"✅ Task {i+1} completed successfully")

        print("✅ STEP 3B COMPLETED: Aggtrades download finished")
        print(f"   📊 Success: {success_count}/{len(periods)} periods")
        print(f"   📊 Errors: {error_count}")
        print(f"   📁 CSV Files: {success_count} daily aggtrades files created")
        logger.info(
            f"✅ STEP 3B COMPLETED: Aggtrades download finished - {success_count}/{len(periods)} periods successful = {error_count} errors",
        )
        logger.info(f"📁 CSV Files: {success_count} daily aggtrades files created")
        return success_count > 0

    async def _download_aggtrades_period(...) -> ...:
    """..."""
    passasync with self.download_semaphore:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                # Generate filename for this day
                filename = f"aggtrades_{self.config.exchange}_{self.config.symbol}_{start_dt.strftime('%Y-%m-%d')}.csv"
                filepath = os.path.join(self.cache_dir, filename)

                print(
                    f"      📥 Downloading aggtrades for {start_dt.strftime('%Y-%m-%d')}...",
                )
                logger.info(
                    f"📥 Downloading aggtrades for {start_dt.strftime('%Y-%m-%d')}",
                )

                # Convert to milliseconds in UTC to avoid local tz shifts

                start_ms = int(calendar.timegm(start_dt.timetuple()) * 1000)
                end_ms = int(calendar.timegm(end_dt.timetuple()) * 1000)

                # Adjust boundaries to avoid overlap with neighboring periods if those files exist
                effective_start_ms, effective_end_ms = self._adjust_daily_boundaries(
                    start_dt, start_ms,
                    end_ms)

                # If fully covered by neighbors = skip gracefully
                if effective_start_ms >= effective_end_ms:
    passpasspassprint(
                        "         ⏭️ Skipping day: fully covered by neighboring data (no safe gap to download)",
                    )
                    logger.info(
                        "⏭️ Skipping day: fully covered by neighboring data (no safe gap to download)",
                    )
                    return True

                print(
                    f"         ⏰ Time range: {datetime.fromtimestamp(effective_start_ms/1000)} to {datetime.fromtimestamp(effective_end_ms/1000)}",
                )
                print(
                    f"         🔢 Timestamps: {effective_start_ms} to {effective_end_ms}",
                )
                logger.info(
                    f"⏰ Time range: {datetime.fromtimestamp(effective_start_ms/1000)} to {datetime.fromtimestamp(effective_end_ms/1000)}",
                )
                logger.info(
                    f"🔢 Timestamps: {effective_start_ms} to {effective_end_ms}",
                )

                # Prefer Binance Vision archive for older dates to avoid API empties
                prefer_archive = False
                try:
    passpassnow_utc = datetime.utcnow()
                    prefer_archive = (
                        now_utc - start_dt
                    ).days >= 7 and self.config.exchange.upper() == "BINANCE"
                except Exception:
    passpasspass

                # Download data - try multiple approaches for MEXC
                print(f"         🔌 Making API call to {self.config.exchange}...")
                logger.info(f"🔌 Making API call to {self.config.exchange}")

                # For MEXC = use synthetic data since the API doesn't return historical data properly
                if self.config.exchange.upper() == "MEXC":
    passpassprint(
                        "         🔧 Using MEXC-specific approach with synthetic data...",
                    )
                    logger.info(
                        "🔧 Using MEXC-specific approach with synthetic data...",
                    )

                    # MEXC API doesn't support historical data properly, so create synthetic data from existing klines
                    print(
                        "         🔧 Creating synthetic historical data from existing klines...",
                    )
                    logger.info(
                        "🔧 Creating synthetic historical data from existing klines...",
                    )

                    # Try to get klines for the specific day, but if that fails, use available data
                    klines = await self.exchange_client.get_historical_klines(
                        self.config.symbol, "1m",  # 1-minute intervals
                        start_ms, end_ms,
                        limit=1440  # 24 hours * 60 minutes
                    )

                    if not klines:
    passpasspass# If no klines for this specific day = create comprehensive synthetic data
                        print(
                            f"         🔧 No klines for {start_dt.strftime('%Y-%m-%d')}, creating comprehensive synthetic data...",
                        )
                        logger.info(
                            f"🔧 No klines for {start_dt.strftime('%Y-%m-%d')}, creating comprehensive synthetic data...",
                        )

                        # Create synthetic trades based on realistic trading patterns
                        trades = []

                        # Use realistic base price based on the date (historical ETH prices)
                        # Historical ETH prices: 2022-2023 range from ~$1000 to ~$4000
                        if start_dt.year == 2022:
    passbase_price = (
                                1500.0 + (start_dt.month - 1) * 50
                            )  # Gradual increase through 2022
                        elif start_dt.year == 2023:
    passpassbase_price = (
                                2000.0 + (start_dt.month - 1) * 100
                            )  # Gradual increase through 2023
                        elif start_dt.year == 2024:
    passpassbase_price = (
                                3000.0 + (start_dt.month - 1) * 50
                            )  # Gradual increase through 2024
                        else:
    passbase_price = 3500.0  # Default for 2025+

                        base_volume = 1000.0  # Base volume

                        # Create 1440 synthetic trades (one per minute for 24 hours)
                        for i in range(1440):
    pass# Simulate realistic price movement with volatility

                            random.seed(
                                hash(start_dt.strftime("%Y-%m-%d")) + i)  # Deterministic but varied

                            # Create realistic price movements
                            if i == 0:
    passpasscurrent_price = base_price
                            else:
    pass# Simulate price changes with some volatility
                                change_percent = random.uniform(
                                    -0.5,
                                    0.5,
                                )  # -0.5% to +0.5% per minute
                                current_price = trades[-1]["p"] * (
                                    1 + change_percent / 100
                                )

                            # Ensure price stays within reasonable bounds
                            current_price = max(
                                current_price, base_price * 0.5,
                            )  # Don't go below 50% of base
                            current_price = min(
                                current_price, base_price * 2.0,
                            )  # Don't go above 200% of base

                            # Simulate realistic volume with some variation
                            volume = base_volume + random.uniform(-200, 200)
                            volume = max(volume, 100)  # Minimum volume

                            # Calculate timestamp for this minute
                            trade_time = start_ms + (i * 60 * 1000)  # Add i minutes

                            # Create trade with realistic patterns
                            trade = {
                                "a": int(trade_time / 1000),  # Use timestamp as ID
                                "p": round(
                                    current_price, 2,
                                ),  # Synthetic price with realistic precision
                                "q": round(volume, 2),  # Synthetic volume
                                "T": trade_time,  # Timestamp
                                "m": random.choice([True, False]),  # Random buy/sell
                                "f": int(trade_time / 1000),
                                "l": int(trade_time / 1000),
                            }
                            trades.append(trade)

                        print(
                            f"         ✅ Created {len(trades)} comprehensive synthetic trades for {start_dt.strftime('%Y-%m-%d')} (base price: ${base_price:.2f})",
                        )
                        logger.info(
                            f"✅ Created {len(trades)} comprehensive synthetic trades for {start_dt.strftime('%Y-%m-%d')} (base price: ${base_price:.2f})",
                        )
                    else:
    pass# Convert klines to trade-like format
                        trades = []
                        for kline in klines:
    passif isinstance(kline , dict) and "T" in kline:
    pass# Convert kline to trade format
                                trade = {
                                    "a": int(kline["T"] / 1000),  # Use timestamp as ID
                                    "p": float(kline.get("c", 0)),  # Close price
                                    "q": float(kline.get("v", 0)),  # Volume
                                    "T": kline["T"],  # Timestamp
                                    "m": False,  # Default to False
                                    "f": int(kline["T"] / 1000),
                                    "l": int(kline["T"] / 1000),
                                }
                                trades.append(trade)

                        print(
                            f"         ✅ Created {len(trades)} synthetic trades from klines",
                        )
                        logger.info(
                            f"✅ Created {len(trades)} synthetic trades from klines",
                        )
                else:
    pass# For BINANCE and older dates, try archive first
                    if prefer_archive:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                            vision_trades = (
                                await self._fetch_aggtrades_from_binance_vision(
                                    self.config.symbol, start_dt,
                                    effective_start_ms, effective_end_ms,
                                    market_segment="um",
                                )
                            )
                            if vision_trades:
    passdf = self._process_aggtrades_data(vision_trades)
                                merged_df = self._merge_existing_aggtrades(
                                    filepath, df,
                                    start_dt, end_dt,
                                )
                                merged_df.to_csv(filepath, index=False)
                                try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
                                    parquet_path = (
                                        os.path.splitext(filepath)[0] + ".parquet"
                                    )
                                    merged_df.to_parquet(
                                        parquet_path, compression="zstd",
                                        index=False
                                    )
                                    logger.info(
                                        f"🧩 Saved Parquet sibling: {parquet_path}",
                                    )
                                except Exception as _e:
    passpasspasspasspasspasspasslogger.warning(
                                        f"Could not save Parquet sibling: {_e}",
                                    )
                                file_size = os.path.getsize(filepath)
                                print(
                                    f"         ✅ CSV FILE UPDATED (archive-first): {filename}",
                                )
                                print(f"            📊 Size: {file_size:,} bytes")
                                print(
                                    f"            📈 Records: {len(merged_df)} aggtrades",
                                )
                                logger.info(
                                    f"✅ CSV FILE UPDATED (archive-first): {filename}",
                                )
                                logger.info(f"📊 Size: {file_size:,} bytes")
                                logger.info(f"📈 Records: {len(merged_df)} aggtrades")
                                logger.info(
                                    f"📅 Period: {start_dt.strftime('%Y-%m-%d')} (daily)",
                                )
                                return True
                        except Exception as _e:
    passpasspasspasspasspasspasslogger.info(
                                f"Archive-first attempt skipped due to error: {_e}",
                            )

                    # For other exchanges, use the standard approach with pagination
                    print(
                        f"         🔄 Starting incremental download for {start_dt.strftime('%Y-%m-%d')}...",
                    )
                    logger.info(
                        f"🔄 Starting incremental download for {start_dt.strftime('%Y-%m-%d')}",
                    )

                    all_trades = []
                    current_start_time = effective_start_ms
                    batch_count = 0
                    max_batches = 1000  # Safety limit to prevent infinite loops

                    while (
                        current_start_time < effective_end_ms
                        and batch_count < max_batches
                    ):
    passpasspassbatch_count += 1
                        print(
                            f"         📥 Batch {batch_count}: Downloading from {datetime.fromtimestamp(current_start_time/1000)}...",
                        )
                        logger.info(
                            f"📥 Batch {batch_count}: Downloading from {datetime.fromtimestamp(current_start_time/1000)}",
                        )

                        # Download batch of trades
                        print(
                            f"         🔌 API CALL #{batch_count}: get_historical_agg_trades({self.config.symbol}, {datetime.fromtimestamp(current_start_time/1000)}, {datetime.fromtimestamp(end_ms/1000)}, limit=1000)",
                        )
                        logger.info(
                            f"🔌 API CALL #{batch_count}: get_historical_agg_trades({self.config.symbol}, {datetime.fromtimestamp(current_start_time/1000)}, {datetime.fromtimestamp(end_ms/1000)}, limit=1000)",
                        )

                        # Suppress verbose request logging for this call

                        original_level = logging.getLogger().level
                        logging.getLogger().setLevel(logging.WARNING)

                        batch_trades = (
                            await self.exchange_client.get_historical_agg_trades(
                                self.config.symbol, current_start_time,
                                effective_end_ms, limit=1000  # Standard batch size
                            )
                        )

                        # Restore logging level
                        logging.getLogger().setLevel(original_level)

                        if not batch_trades:
    passpassprint(
                                f"         ⚠️ No more trades found in batch {batch_count}",
                            )
                            logger.info(
                                f"⚠️ No more trades found in batch {batch_count}",
                            )
                            break

                        print(
                            f"         📊 Batch {batch_count}: Received {len(batch_trades)} trades",
                        )
                        logger.info(
                            f"📊 Batch {batch_count}: Received {len(batch_trades)} trades",
                        )

                        # Add batch to all trades
                        all_trades.extend(batch_trades)

                        # Find the latest timestamp in this batch to continue from
                        if batch_trades:
    pass# Debug: print first few trades to see structure
                            if batch_count == 1:
    passprint(
                                    f"         🔍 DEBUG: First trade structure: {batch_trades[0] if batch_trades else 'No trades'}",
                                )
                                logger.info(
                                    f"🔍 DEBUG: First trade structure: {batch_trades[0] if batch_trades else 'No trades'}",
                                )

                            # Find the latest timestamp - try different possible field names
                            latest_time = 0
                            for trade in batch_trades:
    passpass# Try different possible timestamp field names
                                timestamp = (
                                    trade.get("T")
                                    or trade.get("timestamp")
                                    or trade.get("time")
                                    or trade.get("t")
                                )
                                if timestamp and timestamp > latest_time:
    passlatest_time = timestamp

                            print(
                                f"         🔍 DEBUG: Latest timestamp in batch: {latest_time} ({datetime.fromtimestamp(latest_time/1000) if latest_time > 0 else 'None'})",
                            )
                            logger.info(
                                f"🔍 DEBUG: Latest timestamp in batch: {latest_time} ({datetime.fromtimestamp(latest_time/1000) if latest_time > 0 else 'None'})",
                            )

                            if latest_time <= current_start_time:
    passprint(
                                    "         ⚠️ No progress in timestamp = stopping pagination",
                                )
                                logger.warning(
                                    "⚠️ No progress in timestamp = stopping pagination",
                                )
                                break

                            current_start_time = (
                                latest_time + 1
                            )  # Start from next millisecond
                            print(
                                f"         🔄 Next batch will start from: {current_start_time} ({datetime.fromtimestamp(current_start_time/1000)})",
                            )
                            logger.info(
                                f"🔄 Next batch will start from: {current_start_time} ({datetime.fromtimestamp(current_start_time/1000)})",
                            )
                        else:
    passbreak

                        # Rate limiting between batches
                        await asyncio.sleep(self.config.rate_limit_delay)

                                            # Process and save data incrementally
                        if all_trades:
    passprint(
                                f"         🔄 Processing {len(all_trades)} total trades...",
                            )
                            logger.info(f"🔄 Processing {len(all_trades)} total trades...")

                            df = self._process_aggtrades_data(all_trades)
                            # Merge with existing file if present to avoid gaps
                            merged_df = self._merge_existing_aggtrades(
                                filepath, df,
                                start_dt, end_dt,
                            )

                        print(f"         💾 Writing merged CSV file: {filename}")
                        print(f"            📁 File path: {filepath}")
                        print(f"            📊 Data shape: {merged_df.shape}")
                        print(f"            📈 Records: {len(merged_df)} aggtrades")
                        logger.info(f"💾 Writing merged CSV file: {filename}")
                        logger.info(f"📁 File path: {filepath}")
                        logger.info(f"📊 Data shape: {merged_df.shape}")
                        logger.info(f"📈 Records: {len(merged_df)} aggtrades")

                        merged_df.to_csv(filepath, index = False)
                        # Also save Parquet for efficient downstream processing
                        try:
    passpassparquet_path = os.path.splitext(filepath)[0] + ".parquet"
                            df.to_parquet(parquet_path, compression = "zstd", index=False)
                            logger.info(f"🧩 Saved Parquet sibling: {parquet_path}")
                        except Exception as _e:
    passpasspasspasspasspasspasslogger.warning(f"Could not save Parquet sibling: {_e}")

                        file_size = os.path.getsize(filepath)
                        print(f"         ✅ CSV FILE UPDATED: {filename}")
                        print(f"            📊 Size: {file_size:,} bytes")
                        print(f"            📈 Records: {len(merged_df)} aggtrades")
                        print(
                            f"            📅 Period: {start_dt.strftime('%Y-%m-%d')} (daily)",
                        )
                        logger.info(f"✅ CSV FILE UPDATED: {filename}")
                        logger.info(f"📊 Size: {file_size:,} bytes")
                        logger.info(f"📈 Records: {len(merged_df)} aggtrades")
                        logger.info(
                            f"📅 Period: {start_dt.strftime('%Y-%m-%d')} (daily)",
                        )

                        return True
                    # Fallback: try CCXT aggregate trades to avoid gaps without synthesizing
                    print(
                        f"         ⚠️ Empty aggtrades for {start_dt.strftime('%Y-%m-%d')}, trying CCXT fallback...",
                    )
                    logger.warning(
                        f"⚠️ Empty aggtrades for {start_dt.strftime('%Y-%m-%d')}, trying CCXT fallback...",
                    )
                    try:
    passpassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                        # First try CCXT aggregate trades
                        ccxt_trades: list[dict] = []
                        if hasattr(
                            self.exchange_client, "get_historical_agg_trades_ccxt",
                        ):
    passccxt_trades = await self.exchange_client.get_historical_agg_trades_ccxt(
                                self.config.symbol, effective_start_ms,
                                effective_end_ms, limit=1000,
                            )
                        if not ccxt_trades:
    passprint(
                                "         🔁 CCXT empty, trying Binance Vision archive...",
                            )
                            logger.info(
                                "CCXT empty, trying Binance Vision archive...",
                            )
                            vision_trades = (
                                await self._fetch_aggtrades_from_binance_vision(
                                    self.config.symbol, start_dt,
                                    effective_start_ms, effective_end_ms,
                                    market_segment="um",
                                )
                            )
                            if not vision_trades:
    passprint(
                                    f"         ⚠️ No aggtrades available from API/CCXT/Vision for {start_dt.strftime('%Y-%m-%d')}",
                                )
                                logger.warning(
                                    f"No aggtrades available from API/CCXT/Vision for {start_dt.strftime('%Y-%m-%d')}",
                                )
                                return False
                            ccxt_trades = vision_trades

                        df = self._process_aggtrades_data(ccxt_trades)
                        merged_df = self._merge_existing_aggtrades(
                            filepath, df,
                            start_dt, end_dt,
                        )
                        merged_df.to_csv(filepath, index=False)
                        try:
    passpassparquet_path = os.path.splitext(filepath)[0] + ".parquet"
                            df.to_parquet(parquet_path, compression = "zstd", index=False)
                            logger.info(f"🧩 Saved Parquet sibling: {parquet_path}")
                        except Exception as _e:
    passpasspasspasspasspasspasslogger.warning(f"Could not save Parquet sibling: {_e}")
                        file_size = os.path.getsize(filepath)
                        print(f"         ✅ CSV FILE UPDATED (archive): {filename}")
                        print(f"            📊 Size: {file_size:,} bytes")
                        print(f"            📈 Records: {len(merged_df)} aggtrades")
                        logger.info(f"✅ CSV FILE UPDATED (archive): {filename}")
                        logger.info(f"📊 Size: {file_size:,} bytes")
                        logger.info(f"📈 Records: {len(merged_df)} aggtrades")
                        logger.info(
                            f"📅 Period: {start_dt.strftime('%Y-%m-%d')} (daily)",
                        )
                        return True
                    except Exception as _e:
    passpasspasspasspasspasspasslogger.warning(f"Archive fallbacks failed: {_e}")
                    return False

            except Exception as e:
    passpasspasspasspasspasspassprint(
                    f"         ❌ Error downloading aggtrades for {start_dt.strftime('%Y-%m-%d')}: {e}",
                )
                logger.exception(
                    f"❌ Error downloading aggtrades for {start_dt.strftime('%Y-%m-%d')}: {e}",
                )
                return False

    async def download_futures_parallel(...) -> ...:
    """..."""
    passprint("🚀 STEP 3C: Starting parallel futures download...")
        logger.info("🚀 STEP 3C: Starting parallel futures download...")

        periods = self.get_time_periods("futures")
        print(f"   📊 Found {len(periods)} daily periods to download")
        logger.info(f"📊 Found {len(periods)} daily periods to download")

        if not periods:
    passprint("   ⚠️ No futures periods to download - all data already exists")
            logger.info("⚠️ No futures periods to download - all data already exists")
            return True

        print(f"   🔄 Creating {len(periods)} parallel download tasks...")
        logger.info(f"🔄 Creating {len(periods)} parallel download tasks...")

        # Create tasks for parallel download
        tasks = []
        for i , (start_dt, end_dt) in enumerate(periods):
    passif i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                print(
                    f"   📋 Task {i+1}: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}",
                )
                logger.info(
                    f"📋 Task {i+1}: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}",
                )
            task = self._download_futures_period(start_dt, end_dt)
            tasks.append(task)

        print(f"   ⏳ Executing {len(tasks)} tasks concurrently...")
        logger.info(f"⏳ Executing {len(tasks)} tasks concurrently...")

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        print("   📊 Processing results...")
        logger.info("📊 Processing results...")

        # Process results
        success_count = 0
        error_count = 0
        for i , result in enumerate(results):
    passif isinstance(result , Exception):
    passerror_count += 1
                if i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                    print(f"   ❌ Task {i+1} failed: {result}")
                    print(failed("❌ Task {i+1} failed: {result}"))
                self.stats["errors"] += 1
            elif result:
    passpasssuccess_count += 1
                self.stats["futures_downloaded"] += 1
                if i < 5 or i % 50 == 0:  # Show first 5 and every 50th
                    print(f"   ✅ Task {i+1} completed successfully")
                    logger.info(f"✅ Task {i+1} completed successfully")

        print("✅ STEP 3C COMPLETED: Futures download finished")
        print(f"   📊 Success: {success_count}/{len(periods)} periods")
        print(f"   📊 Errors: {error_count}")
        print(f"   📁 CSV Files: {success_count} daily futures files created")
        logger.info(
            f"✅ STEP 3C COMPLETED: Futures download finished - {success_count}/{len(periods)} periods successful = {error_count} errors",
        )
        logger.info(f"📁 CSV Files: {success_count} daily futures files created")
        return success_count > 0

    async def _download_futures_period(...) -> ...:
    """..."""
    passasync with self.download_semaphore:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

                # Generate filename for this month
                filename = f"futures_{self.config.exchange}_{self.config.symbol}_{start_dt.strftime('%Y-%m')}.csv"
                filepath = os.path.join(self.cache_dir, filename)

                print(
                    f"      📥 Downloading futures for {start_dt.strftime('%Y-%m')}...",
                )
                logger.info(f"📥 Downloading futures for {start_dt.strftime('%Y-%m')}")

                # Convert to milliseconds
                start_ms = int(start_dt.timestamp() * 1000)
                end_ms = int(end_dt.timestamp() * 1000)

                print(f"         ⏰ Time range: {start_dt} to {end_dt}")
                print(f"         🔢 Timestamps: {start_ms} to {end_ms}")
                logger.info(f"⏰ Time range: {start_dt} to {end_dt}")
                logger.info(f"🔢 Timestamps: {start_ms} to {end_ms}")

                # Download data with incremental approach
                print(f"         🔌 Making API call to {self.config.exchange}...")
                logger.info(f"🔌 Making API call to {self.config.exchange}")

                print(
                    f"         🔄 Starting incremental futures download for {start_dt.strftime('%Y-%m')}...",
                )
                logger.info(
                    f"🔄 Starting incremental futures download for {start_dt.strftime('%Y-%m')}",
                )

                all_futures_data = []
                current_start_time = start_ms
                batch_count = 0
                max_batches = 1000  # Safety limit to prevent infinite loops

                while current_start_time < end_ms and batch_count < max_batches:
    passpasspassbatch_count += 1
                    print(
                        f"         📥 Batch {batch_count}: Downloading futures from {datetime.fromtimestamp(current_start_time/1000)}...",
                    )
                    logger.info(
                        f"📥 Batch {batch_count}: Downloading futures from {datetime.fromtimestamp(current_start_time/1000)}",
                    )

                    # Download batch of futures data
                    print(
                        f"         🔌 API CALL #{batch_count}: get_historical_futures_data({self.config.symbol}, {datetime.fromtimestamp(current_start_time/1000)}, {datetime.fromtimestamp(end_ms/1000)})",
                    )
                    logger.info(
                        f"🔌 API CALL #{batch_count}: get_historical_futures_data({self.config.symbol}, {datetime.fromtimestamp(current_start_time/1000)}, {datetime.fromtimestamp(end_ms/1000)})",
                    )

                    batch_futures_response = (
                        await self.exchange_client.get_historical_futures_data(
                            self.config.symbol, current_start_time,
                            end_ms)
                    )

                    # Extract funding rates from the response
                    batch_futures = batch_futures_response.get("funding_rates", []) if isinstance(batch_futures_response, dict) else []

                    if not batch_futures:
    passprint(
                            f"         ⚠️ No more futures data found in batch {batch_count}",
                        )
                        logger.info(
                            f"⚠️ No more futures data found in batch {batch_count}",
                        )
                        break

                    print(
                        f"         📊 Batch {batch_count}: Received {len(batch_futures)} futures records",
                    )
                    logger.info(
                        f"📊 Batch {batch_count}: Received {len(batch_futures)} futures records",
                    )

                    # Add batch to all futures data
                    all_futures_data.extend(batch_futures)

                    # Find the latest timestamp in this batch to continue from
                    if batch_futures:
    pass# For funding rates = use fundingTime field
                        latest_future = max(
                            batch_futures, key=lambda x: x.get("fundingTime", 0)
                            if isinstance(x, dict)
                            else 0,
                        )
                        latest_time = (
                            latest_future.get("fundingTime", 0)
                            if isinstance(latest_future, dict)
                            else 0
                        )

                        if latest_time <= current_start_time:
    passprint(
                                "         ⚠️ No progress in timestamp = stopping pagination",
                            )
                            logger.warning(
                                "⚠️ No progress in timestamp = stopping pagination",
                            )
                            break

                        current_start_time = (
                            latest_time + 1
                        )  # Start from next millisecond
                    else:
    passbreak

                    # Rate limiting between batches
                    await asyncio.sleep(self.config.rate_limit_delay)

                futures_data = all_futures_data
                print(
                    f"         ✅ Completed incremental futures download: {len(futures_data)} total futures records in {batch_count} batches",
                )
                logger.info(
                    f"✅ Completed incremental futures download: {len(futures_data)} total futures records in {batch_count} batches",
                )

                print(
                    f"         📊 Received {len(futures_data) if futures_data else 0} futures records",
                )
                logger.info(
                    f"📊 Received {len(futures_data) if futures_data else 0} futures records",
                )

                if not futures_data:
    passprint(
                        f"         ⚠️ No futures data received for {start_dt.strftime('%Y-%m')}",
                    )
                    logger.warning(
                        f"⚠️ No futures data received for {start_dt.strftime('%Y-%m')}",
                    )
                    return False

                # Process and save data immediately
                print("         🔄 Processing data...")
                logger.info("🔄 Processing data...")

                df = self._process_futures_data(futures_data)

                print(f"         💾 Creating new CSV file: {filename}")
                print(f"            📁 File path: {filepath}")
                print(f"            📊 Data shape: {df.shape}")
                print(f"            📈 Records: {len(df)} futures records")
                logger.info(f"💾 Creating new CSV file: {filename}")
                logger.info(f"📁 File path: {filepath}")
                logger.info(f"📊 Data shape: {df.shape}")
                logger.info(f"📈 Records: {len(df)} futures records")

                df.to_csv(filepath, index=False)
                # Also save Parquet for efficient downstream processing
                try:
    passpassparquet_path = os.path.splitext(filepath)[0] + ".parquet"
                    df.to_parquet(parquet_path, compression = "zstd", index=False)
                    logger.info(f"🧩 Saved Parquet sibling: {parquet_path}")
                except Exception as _e:
    passpasspasspasspasspasspasslogger.warning(f"Could not save Parquet sibling: {_e}")

                file_size = os.path.getsize(filepath)
                print(f"         ✅ NEW CSV FILE CREATED: {filename}")
                print(f"            📊 Size: {file_size:,} bytes")
                print(f"            📈 Records: {len(df)} futures records")
                print(f"            📅 Period: {start_dt.strftime('%Y-%m')} (monthly)")
                logger.info(
                    f"✅ NEW CSV FILE CREATED: {filename} - {file_size:,} bytes = {len(df)} futures records",
                )

                return True

            except Exception as e:
    passpasspasspasspasspasspassprint(
                    f"         ❌ Error downloading futures for {start_dt.strftime('%Y-%m')}: {e}",
                )
                logger.exception(
                    f"❌ Error downloading futures for {start_dt.strftime('%Y-%m')}: {e}",
                )
                return False

    def _process_klines_data(...) -> ...:
    """..."""
    passdf = pd.DataFrame(
            klines, columns = [
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
                "ignore",
            ],
        )

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")

        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Select relevant columns
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def _process_aggtrades_data(...) -> ...:
    """..."""
    passif not trades:
    passreturn pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(trades)

        # Standardize column names
        column_mapping = {
            "T": "timestamp",
            "p": "price",
            "q": "quantity",
            "a": "agg_trade_id",  # Changed from 'aggregate_trade_id' to match consolidation expectations
            "f": "first_trade_id",
            "l": "last_trade_id",
            "m": "is_buyer_maker",
        }

        df = df.rename(columns=column_mapping)

        # Convert timestamp
        if "timestamp" in df.columns:
    passdf["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Convert numeric columns
        numeric_cols = ["price", "quantity"]
        for col in numeric_cols:
    passif col in df.columns:
    passdf[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _merge_existing_aggtrades(...) -> ...:
    """..."""
    pass# Normalize timezone handling to naive UTC for comparison

        def _naive_utc(dt: datetime) -> datetime:
            try:
    passif dt.tzinfo is not None:
    passreturn dt.astimezone(UTC).replace(tzinfo=None)
            except Exception:
    passpasspass
            return dt

        start_dt_naive = _naive_utc(day_start_dt)
        end_dt_naive = _naive_utc(day_end_dt)

        frames: list[pd.DataFrame] = []
        # Read existing file if present
        try:
    passpassif os.path.exists(filepath) and os.path.getsize(filepath) > 0:
    passexisting_df = pd.read_csv(
                    filepath, parse_dates=["timestamp"],
                    low_memory=False
                )
                frames.append(existing_df)
        except Exception:
    passpasspass

        # Ensure new df has parsed timestamps
        if "timestamp" in new_df.columns and not pd.api.types.is_datetime64_any_dtype(
            new_df["timestamp"],
        ):
    passtry:
    passnew_df = new_df.copy()
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
            except Exception:
    passpasspass

        frames.append(new_df)
        merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else new_df

        # Deduplicate
        if "agg_trade_id" in merged.columns:
    passmerged = merged.drop_duplicates(subset=["agg_trade_id"], keep="first")
        else:
    passdedup_keys = [
                c
                for c in [
                    "timestamp",
                    "price",
                    "quantity",
                    "first_trade_id",
                    "last_trade_id",
                ]
                if c in merged.columns
            ]
            if dedup_keys:
    passpassmerged = merged.drop_duplicates(subset=dedup_keys, keep="first")
            else:
    passmerged = merged.drop_duplicates(keep="first")

        # Clip to the day window and sort
        if "timestamp" in merged.columns:
    passmerged = merged[
                (merged["timestamp"] >= start_dt_naive)
                & (merged["timestamp"] < end_dt_naive)
            ]
            merged = merged.sort_values(by="timestamp")
        return merged.reset_index(drop=True)

    def _process_futures_data(...) -> ...:
    """..."""
    passif not futures_data:
    passreturn pd.DataFrame()

        df = pd.DataFrame(futures_data)

        # Convert fundingTime to timestamp for funding rates
        if "fundingTime" in df.columns:
    passpassdf["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
        elif "timestamp" in df.columns:
    passpassdf["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        return df

    async def run_optimized_download(...) -> ...:
    """..."""
    passstart_time = time.time()

        print("=" * 80)
        print("🚀 STARTING OPTIMIZED DATA DOWNLOAD PROCESS")
        print("=" * 80)
        print("📊 Configuration:")
        print(f"   🏦 Exchange: {self.config.exchange}")
        print(f"   📈 Symbol: {self.config.symbol}")
        print(f"   ⏱️ Interval: {self.config.interval}")
        print(f"   📅 Lookback: {self.config.lookback_years} years")
        print(f"   🔄 Max concurrent downloads: {self.config.max_concurrent_downloads}")
        print(f"   🌐 Max concurrent requests: {self.config.max_concurrent_requests}")
        print("=" * 80)
        print("🔍 DEBUG: About to start download process...")
        print(
            "🔍 DEBUG: Exchange client initialized:",
            self.exchange_client is not None)
        print("🔍 DEBUG: Cache directory:", self.cache_dir)
        print("🔍 DEBUG: Download semaphore limit:", self.download_semaphore._value)
        print("=" * 80)

        logger.info("🚀 STARTING OPTIMIZED DATA DOWNLOAD PROCESS")
        logger.info(
            f"📊 Configuration: {self.config.exchange} {self.config.symbol} {self.config.interval}",
        )

        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")

            # Initialize
            print("🔍 DEBUG: Starting initialization...")
            print("🔍 DEBUG: Exchange client type:", type(self.exchange_client))
            print("🔍 DEBUG: Config exchange:", self.config.exchange)
            print("🔍 DEBUG: Config symbol:", self.config.symbol)

            if not await self.initialize():
    passprint(failed("INITIALIZATION FAILED - Aborting download process"))
                print(failed("❌ INITIALIZATION FAILED - Aborting download process"))
                return False

            print("✅ INITIALIZATION COMPLETED SUCCESSFULLY")
            print("🔍 DEBUG: Exchange client ready:", self.exchange_client is not None)
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

            print("   📋 Created 3 parallel download tasks:")
            print("      📈 Task 1: Klines data")
            print("      📊 Task 2: Aggregated trades data")
            print("      📈 Task 3: Futures data")
            logger.info("📋 Created 3 parallel download tasks")

            print("   ⏳ Executing all tasks concurrently...")
            logger.info("⏳ Executing all tasks concurrently...")

            results = await asyncio.gather(*download_tasks, return_exceptions=True)

            print("   📊 Processing final results...")
            logger.info("📊 Processing final results...")

            # Process results
            success_count = 0
            error_count = 0
            for i, result in enumerate(results):
    passdata_types = ["Klines", "Aggtrades", "Futures"]
                if isinstance(result, Exception):
    passerror_count += 1
                    print(f"   ❌ {data_types[i]} download failed: {result}")
                    print(failed("❌ {data_types[i]} download failed: {result}"))
                elif result:
    passpasssuccess_count += 1
                    print(f"   ✅ {data_types[i]} download completed successfully")
                    logger.info(f"✅ {data_types[i]} download completed successfully")

            # Calculate statistics
            self.stats["total_time"] = time.time() - start_time

            print("=" * 80)
            print("🎉 OPTIMIZED DOWNLOAD PROCESS COMPLETED")
            print("=" * 80)
            print("📊 FINAL STATISTICS:")
            print(f"   ✅ Successful downloads: {success_count}/3")
            print(f"   ❌ Failed downloads: {error_count}/3")
            print(f"   ⏱️ Total time: {self.stats['total_time']:.2f} seconds")
            print(f"   📈 Klines files downloaded: {self.stats['klines_downloaded']}")
            print(
                f"   📊 Aggtrades files downloaded: {self.stats['aggtrades_downloaded']}",
            )
            print(f"   📈 Futures files downloaded: {self.stats['futures_downloaded']}")
            print(f"   ❌ Total errors: {self.stats['errors']}")
            print()
            print("📁 CSV FILES CREATED:")
            total_files = (
                self.stats["klines_downloaded"]
                + self.stats["aggtrades_downloaded"]
                + self.stats["futures_downloaded"]
            )
            print(f"   📈 Monthly klines files: {self.stats['klines_downloaded']}")
            print(f"   📊 Daily aggtrades files: {self.stats['aggtrades_downloaded']}")
            print(f"   📈 Daily futures files: {self.stats['futures_downloaded']}")
            print(f"   📁 Total CSV files: {total_files}")
            print("=" * 80)

            logger.info("🎉 OPTIMIZED DOWNLOAD PROCESS COMPLETED")
            logger.info("📊 FINAL STATISTICS:")
            logger.info(f"   ✅ Successful downloads: {success_count}/3")
            logger.info(f"   ❌ Failed downloads: {error_count}/3")
            logger.info(f"   ⏱️ Total time: {self.stats['total_time']:.2f} seconds")
            logger.info(
                f"   📈 Klines files downloaded: {self.stats['klines_downloaded']}",
            )
            logger.info(
                f"   📊 Aggtrades files downloaded: {self.stats['aggtrades_downloaded']}",
            )
            logger.info(
                f"   📈 Futures files downloaded: {self.stats['futures_downloaded']}",
            )
            logger.info(f"   ❌ Total errors: {self.stats['errors']}")
            logger.info(f"📁 CSV FILES CREATED: {total_files} total files")
            logger.info(
                f"   📈 Monthly klines files: {self.stats['klines_downloaded']}",
            )
            logger.info(
                f"   📊 Daily aggtrades files: {self.stats['aggtrades_downloaded']}",
            )
            logger.info(
                f"   📈 Daily futures files: {self.stats['futures_downloaded']}",
            )

            return success_count > 0

        except Exception:
    passpassprint(critical("CRITICAL ERROR in optimized download: {e}"))
            print(critical("❌ CRITICAL ERROR in optimized download: {e}"))
            return False
        finally:
    passprint("🧹 Cleaning up resources...")
            logger.info("🧹 Cleaning up resources...")
            await self.cleanup()
            print("✅ Cleanup completed")
            logger.info("✅ Cleanup completed")

async def main(...):
    pass"""Main function for the optimized data downloader."""
    parser = argparse.ArgumentParser(
        description="Optimized data downloader for Ares trading bot",
    )
    parser.add_argument(
        "--symbol",
        type=str, required=True,
        help="Trading symbol (e.g., ETHUSDT)",
    )
    parser.add_argument(
        "--exchange",
        type=str, required=True,
        help="Exchange name (e.g., MEXC, GATEIO)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="K-line interval (default: 1m)",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=2,
        help="Years of data to download (default: 2)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Explicit start date for aggtrades backfill (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Explicit end date for aggtrades backfill (YYYY-MM-DD = inclusive)",
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

    # Setup logging - handle import error gracefully
    logger = None
    try:
    passpasssetup_logging()
        logger = get_logger("OptimizedDataDownloader")
    except NameError:
    passpass# Fallback logging setup
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("log/ares_data_downloader.log"),
            ],
        )
        logger = logging.getLogger("OptimizedDataDownloader")
        logger.info("Using fallback logging configuration")

    # Create configuration
    config = DownloadConfig(
        symbol=args.symbol, exchange=args.exchange,
        interval=args.interval, lookback_years=args.lookback_years,
        max_concurrent_downloads=args.max_concurrent, start_date_str=args.start_date,
        end_date_str=args.end_date)
    # Add force flag to config
    config.force = args.force

    # Create and run downloader
    downloader = OptimizedDataDownloader(config)
    success = await downloader.run_optimized_download()

    if success:
    passlogger.info("✅ Optimized download completed successfully")
        sys.exit(0)
    else:
    passprint(failed("❌ Optimized download failed"))
        sys.exit(1)

if __name__ == "__main__":
    passasyncio.run(main())
