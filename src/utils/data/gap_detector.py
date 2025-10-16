"""
Gap Detection and Filling Tools for Historical Data

This module provides tools to detect gaps in historical klines data
and fill them by downloading missing data from any supported exchange.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
# Removed direct Binance import - using ExchangeInterface instead
from src.utils.logger import system_logger
from src.utils.parquet_utils import ParquetUtils
from src.trading.execution.exchange_interface import ExchangeInterface

class GapDetector:
    """Detect and fill gaps in historical klines data."""

    def __init__(self, data_dir: str = "historical_data", exchange: str = "binance"):
        """Initialize the gap detector.

        Args:
            data_dir: Base directory for historical data
            exchange: Exchange name for data organization
        """
        self.data_dir = Path(data_dir)
        self.exchange = exchange.lower()
        self.raw_data_dir = self.data_dir / self.exchange
        self.logger = system_logger.getChild("GapDetector")
        self.parquet_utils = ParquetUtils()

    def detect_gaps(
        self,
        symbol: str,
        interval: str = "1m",
        max_gap_minutes: int = 1
    ) -> List[Dict[str, Any]]:
        """Detect gaps in historical data.

        Args:
            symbol: Trading symbol
            interval: Kline interval
            max_gap_minutes: Maximum allowed gap in minutes

        Returns:
            List of gap information dictionaries
        """
        try:
            self.logger.info(f"🔍 Detecting gaps in {symbol} {interval} data")

            # Load all data for the symbol
            all_data = self._load_all_data(symbol, interval)

            if all_data is None or all_data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return []

            # Sort by timestamp
            all_data = all_data.sort_index()

            # Calculate expected interval
            expected_interval = self._get_expected_interval(interval)
            max_gap = timedelta(minutes=max_gap_minutes)

            # Find gaps with intelligent filtering for klines data
            gaps = []
            time_diffs = all_data.index.to_series().diff().dropna()

            # For klines data, filter out very small gaps that are likely artifacts
            if expected_interval >= timedelta(minutes=1):
                # For 1m+ intervals, ignore gaps smaller than 30 seconds
                min_significant_gap = timedelta(seconds=30)
                significant_diffs = time_diffs[time_diffs >= min_significant_gap]
            else:
                # For sub-minute intervals, use the original logic
                significant_diffs = time_diffs

            for i, (timestamp, diff) in enumerate(significant_diffs.items()):
                # For klines, be more lenient with gap detection
                # Only flag gaps that are significantly larger than expected
                if expected_interval >= timedelta(minutes=1):
                    # For klines, require gap to be at least 1.5x expected interval + max_gap
                    gap_threshold = expected_interval * 1.5 + max_gap
                else:
                    # For tick data, use original logic
                    gap_threshold = expected_interval + max_gap

                if diff > gap_threshold:
                    # Find the correct indices in the original data
                    original_idx = time_diffs.index.get_loc(timestamp)
                    gap_start = all_data.index[original_idx-1] if original_idx > 0 else all_data.index[0]
                    gap_end = timestamp
                    gap_duration = gap_end - gap_start

                    gap_info = {
                        "gap_start": gap_start,
                        "gap_end": gap_end,
                        "gap_duration": gap_duration,
                        "gap_seconds": gap_duration.total_seconds(),
                        "gap_minutes": gap_duration.total_seconds() / 60,
                        "expected_records": int(gap_duration.total_seconds() / expected_interval.total_seconds()),
                        "symbol": symbol,
                        "interval": interval,
                        "gap_type": "large" if gap_duration > timedelta(hours=1) else "medium"
                    }
                    gaps.append(gap_info)

                    # Log with appropriate level based on gap size
                    if gap_duration > timedelta(hours=1):
                        self.logger.warning(
                            f"🚨 Large gap detected: {gap_start} to {gap_end} "
                            f"({gap_duration.total_seconds()/3600:.1f} hours, {gap_info['expected_records']} missing records)"
                        )
                    elif gap_duration > timedelta(minutes=5):
                        self.logger.info(
                            f"⚠️ Medium gap detected: {gap_start} to {gap_end} "
                            f"({gap_duration.total_seconds()/60:.1f} minutes)"
                        )
                    else:
                        self.logger.debug(
                            f"ℹ️ Small gap detected: {gap_start} to {gap_end} "
                            f"({gap_duration.total_seconds():.1f} seconds)"
                        )

            self.logger.info(f"🔍 Gap detection completed: {len(gaps)} gaps found")
            return gaps

        except Exception as e:
            self.logger.exception(f"❌ Gap detection failed: {e}")
            return []

    def _load_all_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load all historical data for a symbol.

        Args:
            symbol: Trading symbol
            interval: Kline interval

        Returns:
            Combined DataFrame or None if no data
        """
        try:
            symbol_dir = self.raw_data_dir / symbol.lower() / "raw"
            if not symbol_dir.exists():
                return None

            # Find all parquet files for this symbol and interval
            pattern = f"{symbol.lower()}_{interval}_*.parquet"
            files = list(symbol_dir.glob(pattern))

            if not files:
                return None

            # Load and combine all files
            dataframes = []
            for file_path in sorted(files):
                try:
                    df = self.parquet_utils.safe_read_parquet(str(file_path))
                    if df is not None and not df.empty:
                        dataframes.append(df)
                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")

            if not dataframes:
                return None

            # Combine all dataframes
            combined_df = pd.concat(dataframes, ignore_index=False)
            combined_df = combined_df.sort_index()

            # Remove duplicates
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

            self.logger.info(f"📊 Loaded {len(combined_df)} records from {len(files)} files")
            return combined_df

        except Exception as e:
            self.logger.exception(f"❌ Failed to load data for {symbol}: {e}")
            return None

    def _get_expected_interval(self, interval: str) -> timedelta:
        """Get expected time interval from interval string.

        Args:
            interval: Interval string (e.g., '1m', '5m', '1h')

        Returns:
            Timedelta object
        """
        interval_map = {
            '1m': timedelta(minutes=1),
            '3m': timedelta(minutes=3),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '2h': timedelta(hours=2),
            '4h': timedelta(hours=4),
            '6h': timedelta(hours=6),
            '8h': timedelta(hours=8),
            '12h': timedelta(hours=12),
            '1d': timedelta(days=1),
            '3d': timedelta(days=3),
            '1w': timedelta(weeks=1),
            '1M': timedelta(days=30)  # Approximate month
        }

        return interval_map.get(interval, timedelta(minutes=1))

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
            exchange: Exchange instance (ExchangeInterface)
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
                # Use ExchangeInterface method
                return await exchange.get_klines(
                    symbol, interval, start_time, end_time, limit
                )
        except Exception as e:
            self.logger.error(f"Error getting historical klines: {e}")
            return []

    async def fill_gaps(
        self,
        gaps: List[Dict[str, Any]],
        exchange_interface: Optional[ExchangeInterface] = None,
        api_key: str = "",
        api_secret: str = ""
    ) -> Dict[str, Any]:
        """Fill detected gaps by downloading missing data.

        Args:
            gaps: List of gap information from detect_gaps
            exchange_interface: ExchangeInterface instance (preferred over api_key/api_secret)
            api_key: Exchange API key (fallback if exchange_interface not provided)
            api_secret: Exchange API secret (fallback if exchange_interface not provided)

        Returns:
            Dictionary with filling results
        """
        if not gaps:
            return {"filled_gaps": 0, "total_records_added": 0, "errors": []}

        try:
            self.logger.info(f"🔧 Filling {len(gaps)} gaps")

            # Initialize exchange (prefer ExchangeInterface if provided)
            if exchange_interface is not None:
                exchange = exchange_interface
                await exchange.connect()
            else:
                # Create ExchangeInterface for the specified exchange
                from src.trading.execution.exchange_interface import create_exchange_interface
                config = {
                    'exchange_type': 'binance',  # Default to binance for backward compatibility
                    'api_key': api_key,
                    'api_secret': api_secret,
                    'trade_symbol': gaps[0]["symbol"]
                }
                exchange = create_exchange_interface(config)
                await exchange.connect()

            filled_gaps = 0
            total_records_added = 0
            errors = []

            for i, gap in enumerate(gaps):
                try:
                    self.logger.info(f"🔧 Filling gap {i+1}/{len(gaps)}: {gap['gap_start']} to {gap['gap_end']}")

                    # Download data for the gap period
                    success, records_added = await self._fill_single_gap(exchange, gap)

                    if success:
                        filled_gaps += 1
                        total_records_added += records_added
                        self.logger.info(f"✅ Gap {i+1} filled: {records_added} records added")
                    else:
                        errors.append(f"Failed to fill gap {i+1}: {gap['gap_start']} to {gap['gap_end']}")

                except Exception as e:
                    error_msg = f"Error filling gap {i+1}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)

            # Close exchange connection
            if hasattr(exchange, 'close'):
                await exchange.close()
            elif hasattr(exchange, 'disconnect'):
                await exchange.disconnect()

            result = {
                "filled_gaps": filled_gaps,
                "total_records_added": total_records_added,
                "errors": errors
            }

            self.logger.info(f"🎉 Gap filling completed: {filled_gaps}/{len(gaps)} gaps filled, {total_records_added} records added")
            return result

        except Exception as e:
            self.logger.exception(f"❌ Gap filling failed: {e}")
            return {"filled_gaps": 0, "total_records_added": 0, "errors": [str(e)]}

    async def _fill_single_gap(
        self,
        exchange,
        gap: Dict[str, Any]
    ) -> Tuple[bool, int]:
        """Fill a single gap by downloading missing data.

        Args:
            exchange: Exchange instance
            gap: Gap information dictionary

        Returns:
            Tuple of (success, records_added)
        """
        try:
            symbol = gap["symbol"]
            interval = gap["interval"]
            gap_start = gap["gap_start"]
            gap_end = gap["gap_end"]

            # Convert to milliseconds
            start_time_ms = int(gap_start.timestamp() * 1000)
            end_time_ms = int(gap_end.timestamp() * 1000)

            # Download data for the gap period
            raw_data = await self._get_historical_klines_unified(
                exchange, symbol, interval, start_time_ms, end_time_ms, 1000
            )

            if not raw_data:
                return False, 0

            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
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
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['day'] = df.index.day

            # Save the gap data to appropriate monthly file
            success = await self._save_gap_data(df, symbol, interval)

            return success, len(df)

        except Exception as e:
            self.logger.exception(f"❌ Failed to fill single gap: {e}")
            return False, 0

    async def _save_gap_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> bool:
        """Save gap data to the appropriate monthly file.

        Args:
            df: DataFrame with gap data
            symbol: Trading symbol
            interval: Kline interval

        Returns:
            True if successful, False otherwise
        """
        try:
            symbol_dir = self.raw_data_dir / symbol.lower() / "raw"

            # Group by month and save each month's data
            for (year, month), month_data in df.groupby([df.index.year, df.index.month]):
                filename = f"{symbol.lower()}_{interval}_{year}_{month:02d}.parquet"
                filepath = symbol_dir / filename

                # Load existing data if file exists
                if filepath.exists():
                    existing_df = self.parquet_utils.safe_read_parquet(str(filepath))
                    if existing_df is not None:
                        # Ensure consistent data types before concatenation
                        metadata_cols = ['year', 'month', 'day']
                        for col in metadata_cols:
                            if col in existing_df.columns:
                                existing_df[col] = pd.to_numeric(existing_df[col], errors='coerce').astype('int32')
                            if col in month_data.columns:
                                month_data[col] = pd.to_numeric(month_data[col], errors='coerce').astype('int32')

                        # Combine with existing data
                        combined_df = pd.concat([existing_df, month_data], ignore_index=False)
                        combined_df = combined_df.sort_index()
                        # Remove duplicates
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    else:
                        combined_df = month_data
                else:
                    combined_df = month_data

                # Ensure consistent data types for metadata columns before saving
                metadata_cols = ['year', 'month', 'day']
                for col in metadata_cols:
                    if col in combined_df.columns:
                        combined_df[col] = combined_df[col].astype('int32')

                # Save the combined data
                combined_df.to_parquet(filepath, index=True, compression='snappy')
                self.logger.info(f"💾 Updated {filename} with {len(month_data)} gap records")

            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to save gap data: {e}")
            return False

    def log_gaps(self, gaps: List[Dict[str, Any]]) -> None:
        """Log detailed information about detected gaps.

        Args:
            gaps: List of gap information from detect_gaps
        """
        if not gaps:
            self.logger.info("✅ No gaps detected in the data")
            return

        self.logger.warning(f"⚠️ {len(gaps)} gaps detected:")

        for i, gap in enumerate(gaps, 1):
            self.logger.warning(
                f"  Gap {i}: {gap['gap_start']} to {gap['gap_end']} "
                f"({gap['gap_minutes']:.1f} minutes, {gap['expected_records']} expected records)"
            )

# Convenience functions
async def detect_and_fill_gaps(
    symbol: str = "ETHUSDT",
    interval: str = "1m",
    max_gap_minutes: int = 1,
    data_dir: str = "historical_data",
    api_key: str = "",
    api_secret: str = ""
) -> Dict[str, Any]:
    """Detect and fill gaps in historical data.

    Args:
        symbol: Trading symbol
        interval: Kline interval
        max_gap_minutes: Maximum allowed gap in minutes
        data_dir: Base directory for data storage
        api_key: Exchange API key
        api_secret: Exchange API secret

    Returns:
        Dictionary with gap detection and filling results
    """
    detector = GapDetector(data_dir)

    # Detect gaps
    gaps = detector.detect_gaps(symbol, interval, max_gap_minutes)
    detector.log_gaps(gaps)

    if not gaps:
        return {"gaps_detected": 0, "filled_gaps": 0, "total_records_added": 0, "errors": []}

    # Fill gaps
    fill_results = await detector.fill_gaps(gaps, api_key, api_secret)

    return {
        "gaps_detected": len(gaps),
        **fill_results
    }

if __name__ == "__main__":
    # Example usage
    async def main():
        detector = GapDetector()

        # Detect gaps
        gaps = detector.detect_gaps("ETHUSDT", "1m", max_gap_minutes=1)
        detector.log_gaps(gaps)

        # Fill gaps
        if gaps:
            results = await detector.fill_gaps(gaps)
            print(f"Gap filling results: {results}")

    asyncio.run(main())
