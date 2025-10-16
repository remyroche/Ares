#!/usr/bin/env python3
"""
Monthly Data Downloader

This module provides functionality to download and organize cryptocurrency data
by month, creating separate files for each month's worth of data. This ensures
better organization and easier data management for long-term historical data.

Key Features:
- Downloads data month by month for better organization
- Creates one file per month with clear naming convention
- Handles 3-year data collection (36 monthly files)
- Includes comprehensive duplicate analysis and gap detection
- Integrates with existing quality frameworks
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from calendar import monthrange

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.error_handler import handles_errors
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.data.quality.comprehensive_duplicate_analyzer import ComprehensiveDuplicateAnalyzer
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

logger = system_logger.getChild("MonthlyDataDownloader")


class MonthlyDataDownloader:
    """Monthly data downloader with comprehensive quality checks."""

    def __init__(self, data_cache_path: str = "historical_data", realtime_buffer_hours: int = 2):
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        self.logger = logger.getChild('MonthlyDataDownloader')

        # Real-time data buffer to prevent downloading data too close to current time
        # Binance API typically needs 1-2 hours buffer for data completeness
        self.realtime_buffer_hours = realtime_buffer_hours
        self.logger.info(f"📊 Using {self.realtime_buffer_hours} hour buffer for real-time data")

        # Initialize components
        self.parquet_handler = standardized_parquet_handler
        self.duplicate_analyzer = ComprehensiveDuplicateAnalyzer(self.logger)

        # Download statistics
        self.download_stats = {
            'total_months': 0,
            'successful_months': 0,
            'failed_months': 0,
            'total_rows': 0,
            'total_files': 0,
            'start_time': None,
            'last_download_time': None,
            'monthly_stats': []
        }

        # Initialize exchange instances cache
        self._exchange_instances = {}

        # Lazy initialization of Binance API - only when needed
        self.binance_class = None
        self.binance_config = {}

    def _ensure_binance_api(self) -> bool:
        """Ensure Binance API is available when needed."""
        if self.binance_class is None:
            try:
                from src.exchange.binance import BinanceExchange
                self.binance_class = BinanceExchange
                # Create a default config for BinanceExchange
                self.binance_config = {
                    'binance_exchange': {
                        'use_testnet': True,
                        'timeout': 30,
                        'max_retries': 3
                    }
                }
                self.logger.info("✅ Binance API available")
                return True
            except ImportError:
                self.binance_class = None
                self.binance_config = {}
                self.logger.warning("⚠️ Binance API not available")
                return False
        return True

    def generate_monthly_date_ranges(self, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Generate date ranges for each month within the specified period.
        Applies real-time buffer to prevent downloading data too close to current time.

        Args:
            start_date: Start of the period
            end_date: End of the period

        Returns:
            List of (start_date, end_date) tuples for each month
        """
        monthly_ranges = []
        current_date = start_date.replace(day=1)  # Start from beginning of month

        # Calculate the safe end date (current time minus buffer)
        now = datetime.now()
        safe_end_date = now - timedelta(hours=self.realtime_buffer_hours)
        self.logger.info(f"🛡️ Safe end date: {safe_end_date} (current time minus {self.realtime_buffer_hours}h buffer)")
        self.logger.info(f"📅 Input date range: {start_date} to {end_date}")
        self.logger.info(f"📅 Current time: {now}")

        while current_date <= end_date:
            # Get the last day of current month
            _, last_day = monthrange(current_date.year, current_date.month)
            month_end = current_date.replace(day=last_day, hour=23, minute=59, second=59)

            # If we're in the last month, use the actual end_date
            if month_end > end_date:
                month_end = end_date

            # Apply real-time buffer: if month_end is too close to current time, adjust it
            if month_end > safe_end_date:
                month_end = safe_end_date
                self.logger.info(f"📊 Adjusted month end for {current_date.year}-{current_date.month:02d} to: {month_end}")

            # Only add month if it has valid data range AND is not in the future
            if current_date < month_end and month_end <= safe_end_date:
                monthly_ranges.append((current_date, month_end))
                self.logger.info(f"✅ Added month {current_date.year}-{current_date.month:02d}: {current_date.date()} to {month_end.date()}")
            else:
                self.logger.info(f"⚠️ Skipping month {current_date.year}-{current_date.month:02d}: no valid data range or too close to real-time")

            # Safety check: if current_date itself is too close to real-time, stop processing
            if current_date >= safe_end_date:
                self.logger.warning(f"🛡️ Current date {current_date} is too close to real-time, stopping month generation")
                break

            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1, day=1)

        self.logger.info(f"📅 Generated {len(monthly_ranges)} monthly ranges with {self.realtime_buffer_hours}h buffer")
        return monthly_ranges

    def get_monthly_filename(self, symbol: str, exchange: str, data_type: str,
                           timeframe: str, year: int, month: int) -> str:
        """
        Generate standardized filename for monthly data.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Data type (klines, aggtrades, etc.)
            timeframe: Timeframe (1m, 5m, etc.)
            year: Year
            month: Month

        Returns:
            Standardized filename
        """
        month_str = f"{month:02d}"
        filename = f"{data_type}_{exchange.lower()}_{symbol.lower()}_{timeframe}_{year}_{month_str}.parquet"
        return filename

    def check_monthly_data_exists(self, symbol: str, exchange: str, data_type: str,
                                 timeframe: str, year: int, month: int) -> bool:
        """
        Check if monthly data file already exists.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Data type
            timeframe: Timeframe
            year: Year
            month: Month

        Returns:
            True if file exists, False otherwise
        """
        filename = self.get_monthly_filename(symbol, exchange, data_type, timeframe, year, month)
        filepath = self.data_cache_path / exchange.lower() / symbol.lower() / data_type / filename
        return filepath.exists()

    def get_available_months(self, symbol: str, exchange: str, data_type: str, timeframe: str) -> List[Tuple[int, int]]:
        """
        Get list of months that already have data available.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Data type
            timeframe: Timeframe

        Returns:
            List of (year, month) tuples for available data
        """
        available_months = []
        base_path = self.data_cache_path / exchange.lower() / symbol.lower() / data_type

        if base_path.exists():
            pattern = f"{data_type}_{exchange.lower()}_{symbol.lower()}_{timeframe}_*.parquet"
            for file_path in base_path.glob(pattern):
                filename = file_path.name
                # Extract year and month from filename
                # Format: klines_binance_ethusdt_1m_2025_09.parquet
                parts = filename.split('_')
                if len(parts) >= 6:
                    try:
                        year = int(parts[-2])
                        month = int(parts[-1].split('.')[0])
                        available_months.append((year, month))
                    except (ValueError, IndexError):
                        continue

        return sorted(available_months)

    @handles_errors(context="download_monthly_data")
    @log_important_calls
    async def download_monthly_data(
        self,
        symbol: str,
        exchange: str,
        data_type: str = "klines",
        timeframe: str = "1m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        years: int = 3
    ) -> Dict[str, Any]:
        """
        Download data month by month and create separate files for each month.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'binance')
            data_type: Type of data ('klines', 'aggtrades', 'futures')
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            start_date: Start date (defaults to 3 years ago)
            end_date: End date (defaults to now)
            years: Number of years to download (default: 3)

        Returns:
            Comprehensive download statistics and results
        """
        self.logger.info("🚀 Starting monthly data download")
        self.logger.info(f"📊 Parameters: {symbol} {exchange} {data_type} {timeframe}")
        self.logger.info(f"📅 Years to download: {years}")

        # Set default date range
        if end_date is None:
            # Add 1 hour buffer to avoid downloading data too close to real-time
            end_date = datetime.now() - timedelta(hours=1)
        if start_date is None:
            start_date = end_date - timedelta(days=365 * years)

        self.logger.info(f"📅 Date range: {start_date.date()} to {end_date.date()}")

        # Generate monthly date ranges
        monthly_ranges = self.generate_monthly_date_ranges(start_date, end_date)
        self.logger.info(f"📅 Generated {len(monthly_ranges)} monthly ranges")

        # Initialize download tracking
        self.download_stats['start_time'] = datetime.now()
        self.download_stats['total_months'] = len(monthly_ranges)

        results = {
            'symbol': symbol,
            'exchange': exchange,
            'data_type': data_type,
            'timeframe': timeframe,
            'date_range': {'start': start_date, 'end': end_date},
            'monthly_files': [],
            'quality_summary': {
                'total_records': 0,
                'duplicate_analysis': {},
                'quality_issues': [],
                'recommendations': []
            },
            'download_stats': self.download_stats.copy(),
            'errors': []
        }

        # Get list of already available months
        available_months = self.get_available_months(symbol, exchange, data_type, timeframe)
        self.logger.info(f"📊 Found {len(available_months)} existing monthly files")

        # Download data for each month
        for month_start, month_end in monthly_ranges:
            year = month_start.year
            month = month_start.month

            # Check if data already exists
            if (year, month) in available_months:
                self.logger.info(f"⏭️ Skipping {year}-{month:02d}: data already exists")
                self.download_stats['skipped_months'] = self.download_stats.get('skipped_months', 0) + 1
                continue

            # Skip months that are too old (more than 2 years back from current date)
            # Binance typically has data for popular pairs like ETHUSDT for ~2 years
            current_date = datetime.now()
            month_date = datetime(year, month, 1)
            months_diff = (current_date.year - month_date.year) * 12 + (current_date.month - month_date.month)

            if months_diff > 24:  # More than 2 years old
                self.logger.info(f"⏭️ Skipping {year}-{month:02d}: too old ({months_diff} months ago), likely no data available")
                results['errors'].append({
                    'month': f"{year}-{month:02d}",
                    'error': f"Month too old ({months_diff} months ago), skipping"
                })
                self.download_stats['skipped_old_months'] = self.download_stats.get('skipped_old_months', 0) + 1
                continue

            try:
                month_result = await self._download_single_month(
                    symbol, exchange, data_type, timeframe, month_start, month_end
                )

                if month_result['success']:
                    results['monthly_files'].append(month_result)
                    results['quality_summary']['total_records'] += month_result['record_count']
                    self.download_stats['successful_months'] += 1
                    self.download_stats['total_rows'] += month_result['record_count']
                else:
                    results['errors'].append({
                        'month': f"{month_start.year}-{month_start.month:02d}",
                        'error': month_result.get('error', 'Unknown error')
                    })
                    self.download_stats['failed_months'] += 1

            except Exception as e:
                self.logger.error(f"❌ Failed to download {month_start.year}-{month_start.month:02d}: {e}")
                results['errors'].append({
                    'month': f"{month_start.year}-{month_start.month:02d}",
                    'error': str(e)
                })
                self.download_stats['failed_months'] += 1

        # Generate comprehensive quality summary
        results['quality_summary'].update(self._generate_quality_summary(results))

        # Update final statistics
        self.download_stats['last_download_time'] = datetime.now()
        results['download_stats'] = self.download_stats.copy()

        # Log final results
        self._log_final_results(results)

        return results

    async def _download_single_month(
        self,
        symbol: str,
        exchange: str,
        data_type: str,
        timeframe: str,
        month_start: datetime,
        month_end: datetime
    ) -> Dict[str, Any]:
        """
        Download data for a single month using multiple batches.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Data type
            timeframe: Timeframe
            month_start: Start of month
            month_end: End of month

        Returns:
            Download result for the month
        """
        year = month_start.year
        month = month_start.month

        self.logger.info(f"📥 Downloading {year}-{month:02d}: {month_start.date()} to {month_end.date()}")

        # Get exchange instance
        exchange_instance = await self._get_exchange_instance(exchange)
        if not exchange_instance:
            return {
                'success': False,
                'year': year,
                'month': month,
                'error': f"Failed to initialize {exchange} exchange"
            }

        try:
            if data_type == "klines":
                # Calculate expected records for the month
                days_in_month = (month_end - month_start).days + 1
                if timeframe == "1m":
                    expected_records = days_in_month * 24 * 60  # days * hours * minutes
                elif timeframe == "5m":
                    expected_records = days_in_month * 24 * 12  # days * hours * 5min intervals
                elif timeframe == "15m":
                    expected_records = days_in_month * 24 * 4   # days * hours * 15min intervals
                elif timeframe == "1h":
                    expected_records = days_in_month * 24       # days * hours
                else:
                    expected_records = days_in_month * 24 * 60  # default to 1m

                self.logger.info(f"📊 Expected records for {year}-{month:02d}: ~{expected_records:,} ({days_in_month} days)")

                # Download data in batches using proper historical data approach
                all_data = []
                batch_size = 1000  # CCXT typical limit

                # Calculate total minutes in the month
                total_minutes = int((month_end - month_start).total_seconds() / 60)
                total_batches = (total_minutes // batch_size) + 1

                self.logger.info(f"📦 Will download in {total_batches} batches of {batch_size} records each")

                max_retries = 3  # Maximum retries per batch
                consecutive_failures = 0
                max_consecutive_failures = 3  # Stop after 3 consecutive failures

                # Calculate safe timestamp threshold (current time minus buffer)
                now = datetime.now()
                safe_timestamp = int((now - timedelta(hours=self.realtime_buffer_hours)).timestamp() * 1000)
                self.logger.info(f"🛡️ Safe timestamp threshold: {safe_timestamp} ({now - timedelta(hours=self.realtime_buffer_hours)})")

                # Start from the beginning of the month
                current_timestamp = int(month_start.timestamp() * 1000)
                end_timestamp = int(month_end.timestamp() * 1000)

                for batch_num in range(total_batches):
                    # Check if current timestamp is too close to real-time
                    if current_timestamp >= safe_timestamp:
                        self.logger.warning(f"⚠️ Timestamp {current_timestamp} is too close to real-time (threshold: {safe_timestamp})")
                        self.logger.warning(f"🛡️ Stopping download to prevent API issues with real-time data")
                        break

                    # Check if we've exceeded the month end
                    if current_timestamp > end_timestamp:
                        self.logger.info(f"✅ Reached end of month period")
                        break

                    batch_success = False
                    batch_retries = 0

                    while not batch_success and batch_retries < max_retries:
                        try:
                            # Convert timestamp to human-readable format
                            timestamp_dt = datetime.fromtimestamp(current_timestamp / 1000)
                            readable_timestamp = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
                            self.logger.info(f"📥 Batch {batch_num + 1}/{total_batches}: {readable_timestamp} ({current_timestamp}) - attempt {batch_retries + 1}/{max_retries}")

                            # Download batch - use limit only, not since (CCXT has issues with since parameter)
                            batch_data = await exchange_instance._ccxt_fallback_request(
                                'fetch_ohlcv',
                                symbol,
                                timeframe=timeframe,
                                limit=batch_size
                            )

                            if batch_data and len(batch_data) > 0:
                                # Filter data to only include records from our target month
                                filtered_batch = []
                                for record in batch_data:
                                    record_timestamp = record[0]
                                    if month_start.timestamp() * 1000 <= record_timestamp <= month_end.timestamp() * 1000:
                                        filtered_batch.append(record)

                                if filtered_batch:
                                    all_data.extend(filtered_batch)
                                    self.logger.info(f"✅ Batch {batch_num + 1}: got {len(filtered_batch)} records in target month, total so far: {len(all_data)}")
                                    batch_success = True
                                    consecutive_failures = 0  # Reset consecutive failure counter

                                    # Update timestamp for next batch (move forward by batch_size minutes)
                                    current_timestamp += (batch_size * 60 * 1000)  # batch_size minutes in milliseconds
                                else:
                                    # No data in target month, try next batch
                                    current_timestamp += (batch_size * 60 * 1000)  # Move forward anyway
                                    self.logger.info(f"⚠️ Batch {batch_num + 1}: no data in target month, moving to next batch")
                                    batch_success = True  # Consider this successful since we're making progress
                                    consecutive_failures = 0
                            else:
                                timestamp_dt = datetime.fromtimestamp(current_timestamp / 1000)
                                readable_timestamp = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
                                self.logger.warning(f"⚠️ Batch {batch_num + 1}: no data received for {readable_timestamp} (attempt {batch_retries + 1})")
                                if batch_retries < max_retries - 1:
                                    batch_retries += 1
                                    await asyncio.sleep(1)  # Wait 1 second before retry
                                else:
                                    consecutive_failures += 1
                                    timestamp_dt = datetime.fromtimestamp(current_timestamp / 1000)
                                    readable_timestamp = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
                                    self.logger.warning(f"❌ Batch {batch_num + 1}: failed after {max_retries} attempts for {readable_timestamp}")
                                    # Move forward anyway to avoid infinite loop
                                    current_timestamp += (batch_size * 60 * 1000)
                                    break

                        except Exception as e:
                            self.logger.warning(f"❌ Error in batch {batch_num + 1} (attempt {batch_retries + 1}): {e}")
                            if batch_retries < max_retries - 1:
                                batch_retries += 1
                                await asyncio.sleep(2)  # Wait 2 seconds before retry
                            else:
                                consecutive_failures += 1
                                timestamp_dt = datetime.fromtimestamp(current_timestamp / 1000)
                                readable_timestamp = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
                                self.logger.error(f"❌ Batch {batch_num + 1}: failed after {max_retries} attempts for {readable_timestamp} - {e}")
                                # Move forward anyway to avoid infinite loop
                                current_timestamp += (batch_size * 60 * 1000)

                    # Check if we've had too many consecutive failures
                    if consecutive_failures >= max_consecutive_failures:
                        timestamp_dt = datetime.fromtimestamp(current_timestamp / 1000)
                        readable_timestamp = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
                        self.logger.warning(f"⚠️ Too many consecutive failures ({consecutive_failures}) at {readable_timestamp}, stopping download")
                        break

                    # Small delay between batches to be respectful to the API
                    await asyncio.sleep(0.1)

                if not all_data:
                    return {
                        'success': False,
                        'year': year,
                        'month': month,
                        'error': "No data received from any batch"
                    }

                # Log summary of batch download results
                successful_batches = sum(1 for batch_num in range(total_batches)
                                       if batch_num < len(all_data) // batch_size + 1)
                self.logger.info(f"📊 Batch summary: {successful_batches}/{total_batches} batches successful")
                self.logger.info(f"📊 Data collected: {len(all_data)} records from {successful_batches} batches")

                self.logger.info(f"📊 Total records collected: {len(all_data)}")

                # Remove duplicates from overlapping batches
                unique_data = []
                seen_timestamps = set()

                for record in all_data:
                    timestamp = record[0]
                    if timestamp not in seen_timestamps:
                        unique_data.append(record)
                        seen_timestamps.add(timestamp)

                self.logger.info(f"📊 After deduplication: {len(unique_data)} unique records")

                if not unique_data:
                    return {
                        'success': False,
                        'year': year,
                        'month': month,
                        'error': "No unique data after deduplication"
                    }

                # Convert to DataFrame
                df = pd.DataFrame(unique_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            else:
                return {
                    'success': False,
                    'year': year,
                    'month': month,
                    'error': f"Data type {data_type} not supported yet"
                }

            # Convert timestamp to datetime if needed
            if df['timestamp'].dtype != 'datetime64[ns]':
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Add metadata columns
            df['exchange'] = exchange.lower()
            df['symbol'] = symbol.lower()
            df['timeframe'] = timeframe
            df['data_type'] = data_type

            # Convert timestamp to datetime if needed (CCXT returns milliseconds)
            if df['timestamp'].dtype != 'datetime64[ns]':
                if df['timestamp'].max() > 1e12:  # Check if timestamps are in milliseconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            # Data is already filtered during batch processing
            self.logger.info(f"📊 Final dataset: {len(df)} records from {month_start.date()} to {month_end.date()}")

            if len(df) == 0:
                # If we get no data, it might be because:
                # 1. The month is too old and has no data
                # 2. The API returned no data for that time period
                # 3. There was an issue with the data collection
                self.logger.warning(f"⚠️ No data collected for {year}-{month:02d}. This could mean:")
                self.logger.warning(f"   • The month is too old and has no available data")
                self.logger.warning(f"   • The exchange API doesn't have data for this time period")
                self.logger.warning(f"   • There was an issue with the data collection process")

                return {
                    'success': False,
                    'year': year,
                    'month': month,
                    'error': f"No data collected for {year}-{month:02d}. Month may be too old or have no available data."
                }

            # Validate data quality and time range
            validation_result = self._validate_monthly_data(df, month_start, month_end, year, month)
            if not validation_result['valid']:
                self.logger.warning(f"⚠️ Data validation failed: {validation_result['issues']}")
                # Continue anyway but log the issues

            # Analyze duplicates in the monthly data
            duplicate_analysis = self.duplicate_analyzer.analyze_duplicates(df)

            # Generate filename and save
            filename = self.get_monthly_filename(symbol, exchange, data_type, timeframe, year, month)
            filepath = self.data_cache_path / exchange.lower() / symbol.lower() / data_type / filename

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save data
            self.parquet_handler.write_parquet_standardized(df, filepath)

            # Create metadata
            metadata = {
                'filename': filename,
                'filepath': str(filepath),
                'year': year,
                'month': month,
                'record_count': len(df),
                'date_range': {
                    'start': month_start.isoformat(),
                    'end': month_end.isoformat()
                },
                'data_range': {
                    'start': df['timestamp'].min().isoformat() if len(df) > 0 else None,
                    'end': df['timestamp'].max().isoformat() if len(df) > 0 else None
                },
                'duplicate_analysis': {
                    'total_duplicates': duplicate_analysis.total_duplicates,
                    'true_duplicates': duplicate_analysis.true_duplicate_groups,
                    'false_duplicates': duplicate_analysis.false_duplicate_groups,
                    'recommendations': duplicate_analysis.recommendations
                },
                'validation': validation_result,
                'download_timestamp': datetime.now().isoformat(),
                'quality_score': self._calculate_month_quality_score(df, duplicate_analysis)
            }

            # Save metadata
            metadata_filepath = filepath.with_suffix('.metadata.json')
            import json
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            self.logger.info(f"✅ Saved {len(df)} records to {filename}")
            if duplicate_analysis.total_duplicates > 0:
                self.logger.warning(f"⚠️ Found {duplicate_analysis.total_duplicates} duplicates in {filename}")

            return {
                'success': True,
                'year': year,
                'month': month,
                'filename': filename,
                'filepath': str(filepath),
                'record_count': len(df),
                'duplicate_analysis': {
                    'total_duplicates': duplicate_analysis.total_duplicates,
                    'true_duplicates': duplicate_analysis.true_duplicate_groups,
                    'false_duplicates': duplicate_analysis.false_duplicate_groups
                },
                'date_range': {
                    'start': month_start.isoformat(),
                    'end': month_end.isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"❌ Error downloading {year}-{month:02d}: {e}")
            return {
                'success': False,
                'year': year,
                'month': month,
                'error': str(e)
            }

    async def _get_exchange_instance(self, exchange: str):
        """Get or create exchange instance."""
        if exchange.lower() not in self._exchange_instances:
            try:
                if exchange.lower() == "binance":
                    # Ensure Binance API is available before using it
                    if not self._ensure_binance_api():
                        self.logger.error(f"❌ Binance API not available")
                        return None
                    self._exchange_instances[exchange.lower()] = self.binance_class(self.binance_config)
                else:
                    self.logger.error(f"❌ Exchange {exchange} not supported")
                    return None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {exchange} exchange: {e}")
                return None

        return self._exchange_instances[exchange.lower()]

    def _validate_monthly_data(self, df: pd.DataFrame, month_start: datetime, month_end: datetime, year: int, month: int) -> Dict[str, Any]:
        """
        Validate the quality and correctness of monthly data.

        Args:
            df: DataFrame with the monthly data
            month_start: Start of the target month
            month_end: End of the target month
            year: Target year
            month: Target month

        Returns:
            Dict with validation results
        """
        issues = []
        is_valid = True

        # Check 1: Time range validation
        data_start = df['timestamp'].min()
        data_end = df['timestamp'].max()

        if data_start < month_start:
            issues.append(f"Data starts too early: {data_start} < {month_start}")
            is_valid = False

        if data_end > month_end:
            issues.append(f"Data ends too late: {data_end} > {month_end}")
            is_valid = False

        # Check 2: Expected data volume (rough estimate)
        expected_records = self._estimate_expected_records(month_start, month_end, '1m')
        actual_records = len(df)

        if actual_records == 0:
            issues.append("No data records found")
            is_valid = False
        elif actual_records < expected_records * 0.1:  # Less than 10% of expected
            issues.append(f"Very low data volume: {actual_records}/{expected_records} expected records")
            is_valid = False
        elif actual_records > expected_records * 1.5:  # More than 150% of expected
            issues.append(f"Unexpectedly high data volume: {actual_records}/{expected_records} expected records")

        # Check 3: Data continuity (gaps)
        if len(df) > 1:
            time_diffs = df['timestamp'].diff().dropna()
            expected_diff = pd.Timedelta(minutes=1)  # For 1m timeframe

            # Count gaps larger than expected
            large_gaps = (time_diffs > expected_diff * 2).sum()
            if large_gaps > 0:
                gap_percentage = large_gaps / len(time_diffs)
                if gap_percentage > 0.1:  # More than 10% gaps
                    issues.append(f"High number of data gaps: {large_gaps}/{len(time_diffs)} ({gap_percentage:.1%})")

        # Check 4: Required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            is_valid = False

        # Check 5: Data types
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            issues.append("Timestamp column is not datetime type")
            is_valid = False

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column {col} is not numeric type")
                is_valid = False

        # Check 6: Price/volume sanity
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Check for negative prices
            negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
            if negative_prices > 0:
                issues.append(f"Found {negative_prices} records with negative or zero prices")

            # Check OHLC relationships
            invalid_ohlc = ((df['high'] < df['low']) | (df['open'] < 0) | (df['close'] < 0)).sum()
            if invalid_ohlc > 0:
                issues.append(f"Found {invalid_ohlc} records with invalid OHLC relationships")

        return {
            'valid': is_valid,
            'issues': issues,
            'data_range': {'start': data_start, 'end': data_end},
            'record_count': actual_records,
            'expected_records': expected_records
        }

    def _estimate_expected_records(self, start_date: datetime, end_date: datetime, timeframe: str) -> int:
        """Estimate the expected number of records for a date range."""
        duration_minutes = int((end_date - start_date).total_seconds() / 60)

        if timeframe == '1m':
            return duration_minutes
        elif timeframe == '5m':
            return duration_minutes // 5
        elif timeframe == '15m':
            return duration_minutes // 15
        elif timeframe == '1h':
            return duration_minutes // 60
        else:
            return duration_minutes  # Default to 1m

    def _calculate_month_quality_score(self, df: pd.DataFrame, duplicate_analysis) -> float:
        """Calculate quality score for monthly data."""
        score = 100.0

        # Penalize duplicates
        if duplicate_analysis.total_duplicates > 0:
            duplicate_ratio = duplicate_analysis.total_duplicates / len(df)
            score -= duplicate_ratio * 50  # 50% penalty for duplicates

        # Penalize missing data
        null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score -= null_ratio * 30  # 30% penalty for null values

        return max(0.0, score)

    def _generate_quality_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive quality summary."""
        total_duplicates = sum(
            file['duplicate_analysis']['total_duplicates']
            for file in results['monthly_files']
            if file['success']
        )

        false_duplicates = sum(
            file['duplicate_analysis']['false_duplicates']
            for file in results['monthly_files']
            if file['success']
        )

        quality_summary = {
            'total_files': len(results['monthly_files']),
            'total_duplicates': total_duplicates,
            'false_duplicates': false_duplicates,
            'avg_records_per_month': results['quality_summary']['total_records'] / max(1, len(results['monthly_files'])),
            'quality_issues': [],
            'recommendations': []
        }

        # Add recommendations based on analysis
        if false_duplicates > 0:
            quality_summary['recommendations'].append(
                f"INVESTIGATE: Found {false_duplicates} false duplicates across {len(results['monthly_files'])} files. "
                "Review data sources and collection methods."
            )

        if total_duplicates == 0:
            quality_summary['recommendations'].append(
                "EXCELLENT: No duplicate timestamps found across all monthly files."
            )

        return quality_summary

    def _log_final_results(self, results: Dict[str, Any]):
        """Log comprehensive final results."""
        self.logger.info("="*80)
        self.logger.info("📊 MONTHLY DATA DOWNLOAD - FINAL RESULTS")
        self.logger.info("="*80)

        self.logger.info(f"📁 Symbol: {results['symbol']}")
        self.logger.info(f"🏢 Exchange: {results['exchange']}")
        self.logger.info(f"📊 Data Type: {results['data_type']}")
        self.logger.info(f"⏱️ Timeframe: {results['timeframe']}")

        self.logger.info(f"📅 Date Range: {results['date_range']['start'].date()} to {results['date_range']['end'].date()}")

        self.logger.info(f"📁 Files Created: {len(results['monthly_files'])}")
        self.logger.info(f"📊 Total Records: {results['quality_summary']['total_records']:,}")
        self.logger.info(f"📊 Avg Records/Month: {results['quality_summary']['avg_records_per_month']:.0f}")

        # Show skipped statistics
        skipped_existing = self.download_stats.get('skipped_months', 0)
        skipped_old = self.download_stats.get('skipped_old_months', 0)

        if skipped_existing > 0:
            self.logger.info(f"⏭️ Skipped (already exists): {skipped_existing}")
        if skipped_old > 0:
            self.logger.info(f"⏭️ Skipped (too old): {skipped_old}")

        if results['quality_summary']['total_duplicates'] > 0:
            self.logger.warning(f"⚠️ Total Duplicates: {results['quality_summary']['total_duplicates']}")
            self.logger.warning(f"⚠️ False Duplicates: {results['quality_summary']['false_duplicates']}")

        if results['errors']:
            self.logger.error(f"❌ Failed Months: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                self.logger.error(f"   • {error['month']}: {error['error']}")

        if results['quality_summary']['recommendations']:
            self.logger.info("💡 RECOMMENDATIONS:")
            for rec in results['quality_summary']['recommendations']:
                self.logger.info(f"   • {rec}")

        self.logger.info("="*80)


# Convenience functions for easy usage

async def download_monthly_ethusdt_data(
    years: int = 3,
    data_type: str = "klines",
    timeframe: str = "1m",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    realtime_buffer_hours: int = 2
) -> Dict[str, Any]:
    """
    Convenience function to download monthly ETHUSDT data.

    Args:
        years: Number of years to download (default: 3)
        data_type: Type of data ('klines', 'aggtrades', 'futures')
        timeframe: Timeframe ('1m', '5m', '1h', etc.)
        start_date: Custom start date
        end_date: Custom end date
        realtime_buffer_hours: Hours to buffer from current time (default: 2)

    Returns:
        Download results and statistics
    """
    downloader = MonthlyDataDownloader(realtime_buffer_hours=realtime_buffer_hours)
    return await downloader.download_monthly_data(
        symbol="ETHUSDT",
        exchange="binance",
        data_type=data_type,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        years=years
    )


def get_monthly_file_list(symbol: str = "ETHUSDT", exchange: str = "binance",
                         data_type: str = "klines", timeframe: str = "1m") -> List[Path]:
    """
    Get list of monthly files for a symbol.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_type: Data type
        timeframe: Timeframe

    Returns:
        List of monthly file paths
    """
    base_path = Path("historical_data") / exchange.lower() / symbol.lower() / data_type
    pattern = f"{data_type}_{exchange.lower()}_{symbol.lower()}_{timeframe}_*.parquet"

    if base_path.exists():
        return list(base_path.glob(pattern))
    else:
        return []


if __name__ == "__main__":
    # Example usage
    print("Monthly ETHUSDT Data Downloader")
    print("Usage: python -c \"import asyncio; from src.utils.data.monthly_data_downloader import download_monthly_ethusdt_data; asyncio.run(download_monthly_ethusdt_data())\"")
