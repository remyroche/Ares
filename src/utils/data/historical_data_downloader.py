"""
Historical Data Downloader for Exchange Klines

This module provides tools to download historical klines data from any supported exchange
and save it as optimized monthly parquet files with intelligent batching
and duplicate handling.
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
# Removed direct Binance import - using ExchangeInterface instead
from src.utils.logger import system_logger
from src.utils.parquet_utils import ParquetUtils
from src.utils.data.processing.data_processing import DataProcessor
from src.utils.hardware.memory_optimization import MemoryMonitor, optimize_dataframe_dtypes
from src.trading.execution.exchange_interface import ExchangeInterface, create_exchange_interface

class HistoricalDataDownloader:
    """Download and manage historical klines data from any supported exchange."""

    def __init__(self, data_dir: str = "historical_data", exchange: str = "binance"):
        """Initialize the historical data downloader.

        Args:
            data_dir: Base directory for storing historical data
            exchange: Exchange name for data organization
        """
        self.data_dir = Path(data_dir)
        self.exchange = exchange.lower()
        self.raw_data_dir = self.data_dir / self.exchange
        self.logger = system_logger.getChild("HistoricalDataDownloader")
        self.parquet_utils = ParquetUtils()
        self.data_processor = DataProcessor()
        self.memory_monitor = MemoryMonitor()

        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Batch tracking
        self.batch_count = 0
        self.total_downloaded = 0

    async def _get_historical_klines_unified(
        self,
        exchange,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get historical klines data using unified interface for both exchange types.

        Args:
            exchange: Exchange instance (ExchangeInterface or BinanceExchange)
            symbol: Trading symbol
            interval: Kline interval
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of records

        Returns:
            List of kline data dictionaries
        """
        try:
            # Check if it's ExchangeInterface
            if hasattr(exchange, 'get_klines') and hasattr(exchange, 'exchange_type'):
                # Use ExchangeInterface
                from datetime import datetime
                start_time = datetime.fromtimestamp(start_time_ms / 1000)
                end_time = datetime.fromtimestamp(end_time_ms / 1000)

                klines_data = await exchange.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )

                # Convert KlineData objects to dict format
                result = []
                for kline in klines_data:
                    result.append({
                        "timestamp": int(kline.timestamp.timestamp() * 1000),
                        "open_time": int(kline.timestamp.timestamp() * 1000),
                        "open": kline.open_price,
                        "high": kline.high_price,
                        "low": kline.low_price,
                        "close": kline.close_price,
                        "volume": kline.volume,
                        "close_time": int(kline.close_time.timestamp() * 1000),
                        "quote_volume": kline.quote_asset_volume,
                        "trades": kline.number_of_trades,
                        "taker_buy_base": kline.taker_buy_base_asset_volume,
                        "taker_buy_quote": kline.taker_buy_quote_asset_volume
                    })
                return result
            else:
                # Use direct BinanceExchange method
                return await exchange._get_historical_klines_raw(
                    symbol, interval, start_time_ms, end_time_ms, limit
                )
        except Exception as e:
            self.logger.error(f"Error getting historical klines: {e}")
            return []

    async def download_historical_klines(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        years: int = 3,
        exchange_interface: Optional[ExchangeInterface] = None,
        api_key: str = "",
        api_secret: str = ""
    ) -> bool:
        """Download historical klines data for the specified symbol with intelligent batching.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            years: Number of years of historical data to download
            exchange_interface: ExchangeInterface instance (preferred over api_key/api_secret)
            api_key: Exchange API key (fallback if exchange_interface not provided)
            api_secret: Exchange API secret (fallback if exchange_interface not provided)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"🚀 Starting historical data download for {symbol}")
            self.logger.info(f"📅 Downloading {years} years of {interval} data")

            # Initialize exchange (prefer ExchangeInterface if provided)
            if exchange_interface is not None:
                exchange = exchange_interface
                await exchange.connect()
            else:
                # Create ExchangeInterface for the specified exchange
                config = {
                    'exchange_type': self.exchange,
                    'api_key': api_key,
                    'api_secret': api_secret,
                    'trade_symbol': symbol
                }
                exchange = create_exchange_interface(config)
                await exchange.connect()

            # Calculate date range
            end_date = datetime.now().replace(tzinfo=timezone.utc)
            start_date = end_date - timedelta(days=years * 365)

            # Create symbol directory
            symbol_dir = self.raw_data_dir / symbol.lower() / "raw"
            symbol_dir.mkdir(parents=True, exist_ok=True)

            # Reset batch tracking
            self.batch_count = 0
            self.total_downloaded = 0

            # Download data with intelligent batching
            success = await self._download_with_intelligent_batching(
                exchange, symbol, interval, start_date, end_date, symbol_dir
            )

            # Close exchange connection
            if hasattr(exchange, 'close'):
                await exchange.close()
            elif hasattr(exchange, 'disconnect'):
                await exchange.disconnect()

            self.logger.info(f"🎉 Historical data download completed: {self.total_downloaded} total records")
            return success

        except Exception as e:
            self.logger.exception(f"❌ Historical data download failed: {e}")
            return False

    async def _download_with_intelligent_batching(
        self,
        exchange,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        symbol_dir: Path
    ) -> bool:
        """Download data with intelligent batching to minimize duplicates and gaps.

        Args:
            exchange: Binance exchange instance
            symbol: Trading symbol
            interval: Kline interval
            start_date: Start date for download
            end_date: End date for download
            symbol_dir: Directory to save data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate batch size based on interval
            batch_size_days = self._calculate_batch_size(interval)

            # Get existing data to determine where to start
            existing_data = self._get_existing_data_summary(symbol, interval, symbol_dir)

            # Adjust start date if we have existing data
            if existing_data and existing_data.get('latest_timestamp'):
                latest_existing = existing_data['latest_timestamp']
                # Start from the next interval after latest existing data
                next_start = latest_existing + self._get_interval_timedelta(interval)
                if next_start < end_date:
                    start_date = next_start
                    self.logger.info(f"📊 Resuming from existing data: {start_date}")

            # Download in batches
            current_start = start_date
            all_data = []

            while current_start < end_date:
                current_end = min(current_start + timedelta(days=batch_size_days), end_date)

                # Download batch
                batch_data = await self._download_batch(
                    exchange, symbol, interval, current_start, current_end
                )

                if batch_data is not None and not len(batch_data) == 0:
                    all_data.append(batch_data)
                    self.total_downloaded += len(batch_data)

                    # Print sample every 10 batches
                    self.batch_count += 1
                    if self.batch_count % 10 == 0:
                        self._print_batch_sample(batch_data, self.batch_count)

                # Move to next batch
                current_start = current_end

                # Memory management
                if len(all_data) >= 5:  # Process every 5 batches
                    await self._process_and_save_batches(all_data, symbol, interval, symbol_dir)
                    all_data = []
                    self.memory_monitor.trigger_gc()

            # Process remaining data
            if all_data:
                await self._process_and_save_batches(all_data, symbol, interval, symbol_dir)

            return True

        except Exception as e:
            self.logger.exception(f"❌ Intelligent batching failed: {e}")
            return False

    def _calculate_batch_size(self, interval: str) -> int:
        """Calculate optimal batch size in days based on interval.

        Args:
            interval: Kline interval

        Returns:
            Batch size in days
        """
        # Binance API limit is 1000 records per request
        # Calculate days needed for 1000 records based on interval
        interval_minutes = self._get_interval_minutes(interval)
        records_per_day = 24 * 60 // interval_minutes
        batch_size_days = max(1, 1000 // records_per_day)

        # Cap at reasonable limits
        batch_size_days = min(batch_size_days, 7)  # Max 1 week per batch

        self.logger.debug(f"📏 Calculated batch size: {batch_size_days} days for {interval}")
        return batch_size_days

    def _get_interval_minutes(self, interval: str) -> int:
        """Get interval in minutes.

        Args:
            interval: Interval string (e.g., '1m', '5m', '1h')

        Returns:
            Interval in minutes
        """
        interval_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return interval_map.get(interval, 1)

    def _get_interval_timedelta(self, interval: str) -> timedelta:
        """Get interval as timedelta.

        Args:
            interval: Interval string

        Returns:
            Timedelta object
        """
        minutes = self._get_interval_minutes(interval)
        return timedelta(minutes=minutes)

    def _get_existing_data_summary(self, symbol: str, interval: str, symbol_dir: Path) -> Optional[Dict[str, Any]]:
        """Get summary of existing data to determine where to resume.

        Args:
            symbol: Trading symbol
            interval: Kline interval
            symbol_dir: Data directory

        Returns:
            Dictionary with existing data summary or None
        """
        try:
            # Find existing files
            pattern = f"{symbol.lower()}_{interval}_*.parquet"
            files = list(symbol_dir.glob(pattern))

            if not files:
                return None

            # Get the latest file
            latest_file = max(files, key=lambda f: f.stat().st_mtime)

            # Read a sample to get timestamp info
            df = self.parquet_utils.safe_read_parquet(str(latest_file))
            if df is None or len(df) == 0:
                return None

            return {
                'latest_timestamp': df.index.max(),
                'earliest_timestamp': df.index.min(),
                'record_count': len(df),
                'file_count': len(files)
            }

        except Exception as e:
            self.logger.warning(f"Could not get existing data summary: {e}")
            return None

    async def _download_batch(
        self,
        exchange,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Download a single batch of data.

        Args:
            exchange: Binance exchange instance
            symbol: Trading symbol
            interval: Kline interval
            start_date: Batch start date
            end_date: Batch end date

        Returns:
            DataFrame with batch data or None if failed
        """
        try:
            # Convert to milliseconds
            start_time_ms = int(start_date.timestamp() * 1000)
            end_time_ms = int(end_date.timestamp() * 1000)

            # Download raw klines data
            raw_data = await self._get_historical_klines_unified(
                exchange, symbol, interval, start_time_ms, end_time_ms, 1000
            )

            if not raw_data:
                self.logger.warning(f"No data received for batch {start_date} to {end_date}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(raw_data)

            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp')

            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add metadata
            df['symbol'] = symbol
            df['interval'] = interval
            df['year'] = df.index.year.astype('int32')  # Explicitly set as int32
            df['month'] = df.index.month.astype('int32')  # Explicitly set as int32
            df['day'] = df.index.day.astype('int32')  # Explicitly set as int32

            # Optimize data types
            df = optimize_dataframe_dtypes(df)

            # Ensure metadata columns remain properly typed for PyArrow compatibility
            if 'year' in df.columns:
                df['year'] = df['year'].astype('int32')
            if 'month' in df.columns:
                df['month'] = df['month'].astype('int32')
            if 'day' in df.columns:
                df['day'] = df['day'].astype('int32')

            return df

        except Exception as e:
            self.logger.exception(f"❌ Failed to download batch {start_date} to {end_date}: {e}")
            return None

    def _print_batch_sample(self, df: pd.DataFrame, batch_number: int) -> None:
        """Print sample of batch data every 10 batches.

        Args:
            df: DataFrame with batch data
            batch_number: Current batch number
        """
        try:
            self.logger.info(f"📊 Batch {batch_number} sample (top 10 rows):")
            sample_df = df.head(10)

            # Print key columns
            key_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in key_columns if col in sample_df.columns]

            if available_columns:
                sample_data = sample_df[available_columns].round(4)
                for idx, row in sample_data.iterrows():
                    self.logger.info(f"  {idx}: {dict(row)}")

            self.logger.info(f"  Total records in batch: {len(df)}")
            self.logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")

        except Exception as e:
            self.logger.warning(f"Could not print batch sample: {e}")

    async def _process_and_save_batches(
        self,
        batches: List[pd.DataFrame],
        symbol: str,
        interval: str,
        symbol_dir: Path
    ) -> None:
        """Process and save multiple batches to monthly files.

        Args:
            batches: List of DataFrames to process
            symbol: Trading symbol
            interval: Kline interval
            symbol_dir: Directory to save data
        """
        try:
            if not batches:
                return

            # Combine all batches
            combined_df = pd.concat(batches, ignore_index=False)
            combined_df = combined_df.sort_index()

            # Remove duplicates (keep last occurrence)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

            # Group by month and save
            for (year, month), month_data in combined_df.groupby([combined_df.index.year, combined_df.index.month]):
                filename = f"{symbol.lower()}_{interval}_{year}_{month:02d}.parquet"
                filepath = symbol_dir / filename

                # Load existing data if file exists
                if filepath.exists():
                    existing_df = self.parquet_utils.safe_read_parquet(str(filepath))
                    if existing_df is not None:
                        # Combine with existing data
                        month_data = pd.concat([existing_df, month_data], ignore_index=False)
                        month_data = month_data.sort_index()
                        # Remove duplicates
                        month_data = month_data[~month_data.index.duplicated(keep='last')]

                # Ensure metadata columns are properly typed before saving
                if 'year' in month_data.columns:
                    month_data['year'] = month_data['year'].astype('int32')
                if 'month' in month_data.columns:
                    month_data['month'] = month_data['month'].astype('int32')
                if 'day' in month_data.columns:
                    month_data['day'] = month_data['day'].astype('int32')

                # Save the combined data
                month_data.to_parquet(filepath, index=True, compression='snappy')
                self.logger.info(f"💾 Saved {len(month_data)} records to {filename}")

        except Exception as e:
            self.logger.exception(f"❌ Failed to process and save batches: {e}")

    async def _download_month_data(
        self,
        exchange,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        symbol_dir: Path
    ) -> bool:
        """Download data for a specific month.

        Args:
            exchange: Binance exchange instance
            symbol: Trading symbol
            interval: Kline interval
            start_date: Start date for the month
            end_date: End date for the month
            symbol_dir: Directory to save the data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to milliseconds
            start_time_ms = int(start_date.timestamp() * 1000)
            end_time_ms = int(end_date.timestamp() * 1000)

            # Download raw klines data
            raw_data = await self._get_historical_klines_unified(
                exchange, symbol, interval, start_time_ms, end_time_ms, 1000
            )

            if not raw_data:
                self.logger.warning(f"No data received for {start_date.strftime('%Y-%m')}")
                return False

            # Convert to DataFrame
            df = pd.DataFrame(raw_data)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp')

            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add metadata
            df['symbol'] = symbol
            df['interval'] = interval
            df['year'] = df.index.year.astype('int32')  # Explicitly set as int32
            df['month'] = df.index.month.astype('int32')  # Explicitly set as int32
            df['day'] = df.index.day.astype('int32')  # Explicitly set as int32

            # Optimize data types
            df = self.data_processor.optimize_dataframe_dtypes(df)

            # Ensure metadata columns remain properly typed for PyArrow compatibility
            if 'year' in df.columns:
                df['year'] = df['year'].astype('int32')
            if 'month' in df.columns:
                df['month'] = df['month'].astype('int32')
            if 'day' in df.columns:
                df['day'] = df['day'].astype('int32')

            # Save as parquet
            filename = f"{symbol.lower()}_{interval}_{start_date.strftime('%Y_%m')}.parquet"
            filepath = symbol_dir / filename

            df.to_parquet(filepath, index=True, compression='snappy')

            self.logger.info(f"💾 Saved {len(df)} records to {filename}")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to download month data for {start_date.strftime('%Y-%m')}: {e}")
            return False

    def get_downloaded_files(self, symbol: str) -> List[Path]:
        """Get list of downloaded files for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of file paths
        """
        symbol_dir = self.raw_data_dir / symbol.lower() / "raw"
        if not symbol_dir.exists():
            return []

        return list(symbol_dir.glob("*.parquet"))

    def get_data_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of downloaded data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with data summary
        """
        files = self.get_downloaded_files(symbol)

        if not files:
            return {"files_count": 0, "total_records": 0, "date_range": None}

        total_records = 0
        date_ranges = []

        for file_path in files:
            try:
                df = self.parquet_utils.safe_read_parquet(str(file_path))
                if df is not None and not len(df) == 0:
                    total_records += len(df)
                    date_ranges.append((df.index.min(), df.index.max()))
            except Exception as e:
                self.logger.warning(f"Could not read {file_path}: {e}")

        summary = {
            "files_count": len(files),
            "total_records": total_records,
            "date_range": None
        }

        if date_ranges:
            min_date = min(dt[0] for dt in date_ranges)
            max_date = max(dt[1] for dt in date_ranges)
            summary["date_range"] = (min_date, max_date)

        return summary

# Convenience functions
async def download_ethusdt_historical_data(
    years: int = 3,
    data_dir: str = "historical_data",
    api_key: str = "",
    api_secret: str = ""
) -> bool:
    """Download 3 years of ETHUSDT historical data.

    Args:
        years: Number of years to download
        data_dir: Base directory for data storage
        api_key: Binance API key
        api_secret: Binance API secret

    Returns:
        True if successful, False otherwise
    """
    downloader = HistoricalDataDownloader(data_dir)
    return await downloader.download_historical_klines(
        symbol="ETHUSDT",
        interval="1m",
        years=years,
        api_key=api_key,
        api_secret=api_secret
    )

if __name__ == "__main__":
    # Example usage
    async def main():
        downloader = HistoricalDataDownloader()
        success = await downloader.download_historical_klines(
            symbol="ETHUSDT",
            interval="1m",
            years=3
        )

        if success:
            summary = downloader.get_data_summary("ETHUSDT")
            print(f"Download completed: {summary}")
        else:
            print("Download failed")

    asyncio.run(main())
