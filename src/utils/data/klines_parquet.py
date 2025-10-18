"""
Unified Klines Parquet Data Management

This module provides a unified interface for creating, updating, and accessing
historical klines data stored in optimized parquet format.
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import logging
from src.utils.parquet_utils import ParquetUtils
from src.utils.data.processing.data_processing import DataProcessor
from src.utils.tprint import tprint

class KlinesParquetManager:
    """Unified manager for klines parquet data operations."""

    def __init__(self, data_dir: str = "historical_data", exchange: str = "binance"):
        """Initialize the klines parquet manager.

        Args:
            data_dir: Base directory for data storage
            exchange: Exchange name (binance, bingx, mexc, etc.)
        """
        self.data_dir = Path(data_dir)
        self.exchange = exchange.lower()
        self.raw_data_dir = self.data_dir / self.exchange
        self.processed_data_dir = self.data_dir / self.exchange
        self.logger = logging.getLogger("KlinesParquetManager")
        self.parquet_utils = ParquetUtils()
        self.data_processor = DataProcessor()

        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def _get_last_x_days_fallback(self, df: pd.DataFrame, x_days: int = 20) -> Optional[pd.DataFrame]:
        """Get the last x days of available data as a fallback.

        Args:
            df: DataFrame to filter
            x_days: Number of days to go back from the maximum available date

        Returns:
            Filtered DataFrame with last x days of data or None if no data
        """
        if df is None or df.empty:
            return None

        try:
            # Find the maximum available date
            timestamp_col = None
            if 'timestamp' in df.columns:
                timestamp_series = df['timestamp'].copy()
                valid_mask = pd.notna(timestamp_series) & np.isfinite(timestamp_series)
                if valid_mask.any():
                    timestamp_series = timestamp_series[valid_mask]
                    df = df[valid_mask]
                    timestamp_col = pd.to_datetime(timestamp_series, unit='s')
            elif 'open_time' in df.columns:
                open_time_series = df['open_time'].copy()
                valid_mask = pd.notna(open_time_series) & np.isfinite(open_time_series)
                if valid_mask.any():
                    open_time_series = open_time_series[valid_mask]
                    df = df[valid_mask]
                    timestamp_col = pd.to_datetime(open_time_series, unit='ms')

            if timestamp_col is not None and len(timestamp_col) > 0:
                max_date = timestamp_col.max()
                if not pd.isna(max_date):
                    # Calculate start date for last x days
                    start_date = max_date - timedelta(days=x_days)

                    # Apply the filter
                    mask = timestamp_col >= start_date
                    filtered_df = df[mask]

                    # Only log successful fallback data retrieval
                    max_date_str = max_date.date() if hasattr(max_date, 'date') else str(max_date)
                    self.logger.info(f"✅ Fallback data loaded: last {x_days} days from {max_date_str} -> {len(filtered_df)} records")

                    return filtered_df

        except Exception as e:
            self.logger.warning(f"Could not apply last {x_days} days fallback: {e}")

        return None

    def _apply_date_filter_to_dataframe(self, df: pd.DataFrame, start_date: Optional[datetime], end_date: Optional[datetime]) -> Optional[pd.DataFrame]:
        """Apply date filtering to a dataframe.

        Args:
            df: DataFrame to filter
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Filtered DataFrame or None if no data matches the filter
        """
        if df is None or df.empty:
            return None

        # Convert string dates to datetime objects if needed
        if start_date and isinstance(start_date, str):
            try:
                start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except:
                start_date = None
        
        if end_date and isinstance(end_date, str):
            try:
                end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except:
                end_date = None

        # Apply date filtering if specified
        if start_date is not None or end_date is not None:
            # Only log the date filter application at debug level to reduce verbosity
            self.logger.debug(f"📅 Applying date filter to dataframe: {start_date} to {end_date}")

            # Convert timestamps to datetime for proper filtering
            timestamp_col = None
            if 'timestamp' in df.columns:
                # Use timestamp column for filtering
                try:
                    # First, clean the timestamp column by removing NaN and infinite values
                    timestamp_series = df['timestamp'].copy()

                    # Remove NaN and infinite values
                    valid_mask = pd.notna(timestamp_series) & np.isfinite(timestamp_series)
                    if not valid_mask.all():
                        invalid_count = len(timestamp_series) - valid_mask.sum()
                        self.logger.warning(f"⚠️ Found {invalid_count} invalid timestamps in dataframe, removing them")
                        timestamp_series = timestamp_series[valid_mask]
                        df = df[valid_mask]

                    # Try to convert timestamps to datetime using seconds first
                    timestamp_col = pd.to_datetime(timestamp_series, unit='s')
                except (OverflowError, FloatingPointError):
                    # If overflow occurs, try with milliseconds
                    try:
                        timestamp_col = pd.to_datetime(timestamp_series, unit='ms')
                    except (OverflowError, FloatingPointError):
                        # If still failing, convert to int64 and try again
                        timestamp_int = timestamp_series.astype('int64')
                        timestamp_col = pd.to_datetime(timestamp_int, unit='s')
            elif 'open_time' in df.columns:
                # Handle case where data has open_time but no timestamp column
                try:
                    # First, clean the open_time column by removing NaN and infinite values
                    open_time_series = df['open_time'].copy()

                    # Remove NaN and infinite values
                    valid_mask = pd.notna(open_time_series) & np.isfinite(open_time_series)
                    if not valid_mask.all():
                        invalid_count = len(open_time_series) - valid_mask.sum()
                        self.logger.warning(f"⚠️ Found {invalid_count} invalid open_time values in dataframe, removing them")
                        open_time_series = open_time_series[valid_mask]
                        df = df[valid_mask]

                    # open_time is typically in milliseconds for exchange data
                    timestamp_col = pd.to_datetime(open_time_series, unit='ms')
                except (OverflowError, FloatingPointError):
                    # If overflow occurs, try with seconds
                    try:
                        timestamp_col = pd.to_datetime(open_time_series, unit='s')
                    except (OverflowError, FloatingPointError):
                        # If still failing, convert to int64 and try again
                        timestamp_int = open_time_series.astype('int64')
                        timestamp_col = pd.to_datetime(timestamp_int, unit='ms')

            if timestamp_col is not None and len(timestamp_col) > 0:
                # Apply the date filtering
                mask = (timestamp_col >= start_date) & (timestamp_col <= end_date)
                df = df[mask]

                if len(df) == 0:
                    # Only log at debug level to reduce verbosity
                    self.logger.debug(f"⚠️ No data found in date range {start_date} to {end_date}")
                    return None

                # Only log successful filtering results
                start_date_str = start_date.date() if hasattr(start_date, 'date') else str(start_date)
                end_date_str = end_date.date() if hasattr(end_date, 'date') else str(end_date)
                self.logger.info(f"📅 Data filtered: {len(df)} records from {start_date_str} to {end_date_str}")
            else:
                self.logger.warning("⚠️ Could not find timestamp column for date filtering or timestamp column is empty")
                return None

        return df

    def get_data_info(self, symbol: str, interval: str, data_type: str = "raw") -> Dict[str, Any]:
        """Get information about available data.

        Args:
            symbol: Trading symbol
            interval: Data interval
            data_type: 'raw' or 'processed'

        Returns:
            Dictionary with data information
        """
        try:
            if data_type == "raw":
                data_dir = self.raw_data_dir / symbol.lower() / "raw"
            else:
                data_dir = self.processed_data_dir / symbol.lower() / "processed"

            if not data_dir.exists():
                return {
                    "available": False,
                    "files_count": 0,
                    "total_records": 0,
                    "date_range": None,
                    "file_size_mb": 0
                }

            # Find matching files
            if data_type == "raw":
                pattern = f"{symbol.lower()}_{interval}_*.parquet"
            else:
                pattern = f"{symbol.lower()}_{interval}"

            files = list(data_dir.glob(f"{pattern}*"))

            if not files:
                return {
                    "available": False,
                    "files_count": 0,
                    "total_records": 0,
                    "date_range": None,
                    "file_size_mb": 0
                }

            # Calculate total size - handle both files and partitioned directories
            total_size = 0
            for f in files:
                if f.is_file():
                    total_size += f.stat().st_size
                elif f.is_dir() and data_type == "processed":
                    # For partitioned processed data, calculate size recursively
                    for root, dirs, files_in_dir in os.walk(f):
                        for file in files_in_dir:
                            if file.endswith('.parquet'):
                                file_path = os.path.join(root, file)
                                try:
                                    total_size += os.path.getsize(file_path)
                                except OSError:
                                    pass  # Skip files that can't be accessed

            # Get date range and record count
            total_records = 0
            date_ranges = []

            for file_path in files:
                try:
                    if data_type == "processed" and file_path.is_dir():
                        # For processed data, it might be partitioned - recursively find all parquet files
                        all_parquet_files = []
                        for root, dirs, files_in_dir in os.walk(file_path):
                            for file in files_in_dir:
                                if file.endswith('.parquet'):
                                    all_parquet_files.append(os.path.join(root, file))

                        # Sample a subset of files to avoid reading everything (for performance)
                        sample_size = min(10, len(all_parquet_files))  # Sample up to 10 files
                        sampled_files = all_parquet_files[:sample_size] if sample_size > 0 else all_parquet_files

                        for pf in sampled_files:
                            df = self.parquet_utils.safe_read_parquet(pf)
                            if df is not None and not df.empty:
                                total_records += len(df)
                                date_ranges.append((df.index.min(), df.index.max()))

                        # Estimate total records based on sample
                        if len(all_parquet_files) > sample_size and sample_size > 0:
                            avg_records_per_file = total_records / sample_size
                            estimated_total = int(avg_records_per_file * len(all_parquet_files))
                            total_records = estimated_total
                            self.logger.info(f"📊 Estimated {estimated_total:,} total records from {len(all_parquet_files)} files (sampled {sample_size})")

                    else:
                        # For raw data or single files
                        df = self.parquet_utils.safe_read_parquet(str(file_path))
                        if df is not None and not df.empty:
                            total_records += len(df)
                            date_ranges.append((df.index.min(), df.index.max()))
                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")

            # Ensure total_size is numeric to prevent string division errors
            try:
                file_size_mb = float(total_size) / (1024 * 1024)
            except (TypeError, ValueError):
                self.logger.warning(f"⚠️ Could not calculate file size, total_size type: {type(total_size)}, value: {total_size}")
                file_size_mb = 0.0

            info = {
                "available": True,
                "files_count": len(files),
                "total_records": total_records,
                "date_range": None,
                "file_size_mb": file_size_mb
            }

            if date_ranges:
                # Normalize timezone-aware and timezone-naive timestamps to prevent comparison errors
                normalized_ranges = []
                for dt in date_ranges:
                    try:
                        # Convert to timezone-naive if needed
                        start_dt = dt[0]
                        end_dt = dt[1]
                        
                        # Handle timezone-aware timestamps by converting to UTC and removing timezone info
                        if hasattr(start_dt, 'tz') and start_dt.tz is not None:
                            start_dt = start_dt.tz_convert('UTC').tz_localize(None)
                        if hasattr(end_dt, 'tz') and end_dt.tz is not None:
                            end_dt = end_dt.tz_convert('UTC').tz_localize(None)
                            
                        normalized_ranges.append((start_dt, end_dt))
                    except Exception as e:
                        self.logger.warning(f"Could not normalize date range {dt}: {e}")
                        # Skip problematic date ranges
                        continue
                
                if normalized_ranges:
                    min_date = min(dt[0] for dt in normalized_ranges)
                    max_date = max(dt[1] for dt in normalized_ranges)
                    info["date_range"] = (min_date, max_date)

            return info

        except Exception as e:
            self.logger.exception(f"❌ Failed to get data info: {e}")
            return {
                "available": False,
                "files_count": 0,
                "total_records": 0,
                "date_range": None,
                "file_size_mb": 0,
                "error": str(e)
            }

    def read_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        data_type: str = "raw",
        columns: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Read klines data for a symbol and interval.

        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date for filtering
            end_date: End date for filtering
            data_type: 'raw' or 'processed'
            columns: List of columns to read

        Returns:
            DataFrame with klines data or None if not found
        """
        try:
            # Convert string dates to datetime objects if needed
            if start_date and isinstance(start_date, str):
                try:
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                except:
                    start_date = None
            
            if end_date and isinstance(end_date, str):
                try:
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                except:
                    end_date = None
            # Auto-detect data type: use processed data for timeframes > 1m
            if data_type == "raw" and interval in ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"]:
                self.logger.info(f"🔄 Auto-switching to processed data for {interval} timeframe")
                data_type = "processed"

            if data_type == "raw":
                data_dir = self.raw_data_dir / symbol.lower() / "raw"
                pattern = f"{symbol.lower()}_{interval}_*.parquet"
            else:
                data_dir = self.processed_data_dir / symbol.lower() / "processed"
                pattern = f"{symbol.lower()}_{interval}"

            if not data_dir.exists():
                self.logger.warning(f"No data directory found for {symbol} {interval}")
                self.logger.warning(f"🔍 DEBUG: Looking for directory: {data_dir}")
                return None

            # Find matching files - be more flexible with pattern matching
            files = list(data_dir.glob(f"*{symbol.lower()}_{interval}*"))

            if not files:
                self.logger.warning(f"⚠️ No files found for {symbol} {interval}")
                return None

            # Load and combine data
            dataframes = []

            # Prioritize consolidated files over partitioned directories
            consolidated_files = [f for f in files if f.is_file() and 'consolidated' in f.name.lower()]
            partitioned_dirs = [f for f in files if f.is_dir()]

            # Also check for consolidated files inside directories
            for dir_path in partitioned_dirs:
                consolidated_in_dir = list(dir_path.glob('*consolidated*.parquet'))
                if consolidated_in_dir:
                    consolidated_files.extend(consolidated_in_dir)

            # Determine data loading strategy
            use_consolidated_only = False

            # First try consolidated files - if found, use ONLY consolidated files
            if consolidated_files:
                for file_path in sorted(consolidated_files):
                    try:
                        df = self.parquet_utils.safe_read_parquet(str(file_path), columns=columns)
                        if df is not None and not df.empty:
                            # Apply date filtering to consolidated file if dates are specified
                            if start_date is not None or end_date is not None:
                                df = self._apply_date_filter_to_dataframe(df, start_date, end_date)
                                if df is not None and not df.empty:
                                    # Only log successful data retrieval with timeframe and period info
                                    start_date_str = start_date.date() if hasattr(start_date, 'date') else str(start_date)
                                    end_date_str = end_date.date() if hasattr(end_date, 'date') else str(end_date)
                                    period_info = f"from {start_date_str} to {end_date_str}" if start_date and end_date else "full period"
                                    self.logger.info(f"✅ Data loaded: {symbol} {interval} {period_info} -> {len(df)} records")
                                    dataframes.append(df)
                                    use_consolidated_only = True  # Mark to use only consolidated data
                                    break  # Use only the first consolidated file
                            else:
                                # Only log successful data retrieval with timeframe info
                                self.logger.info(f"✅ Data loaded: {symbol} {interval} full period -> {len(df)} records")
                                dataframes.append(df)
                                use_consolidated_only = True  # Mark to use only consolidated data
                                break  # Use only the first consolidated file
                    except Exception as e:
                        self.logger.warning(f"Could not read consolidated file {file_path}: {e}")

            # Load partitioned data only if we haven't loaded consolidated data
            if not use_consolidated_only and partitioned_dirs:
                # Filter partitioned directories by date range if dates are specified
                filtered_partitioned_dirs = []
                if start_date is not None or end_date is not None:
                    for file_path in sorted(partitioned_dirs):
                        # Extract year and month from path (format: year=YYYY/month=MM)
                        path_str = str(file_path)
                        year_match = re.search(r'year=(\d{4})', path_str)
                        month_match = re.search(r'month=(\d{1,2})', path_str)

                        if year_match and month_match:
                            year = int(year_match.group(1))
                            month = int(month_match.group(1))

                            # Check if this partition overlaps with the requested date range
                            partition_start = datetime(year, month, 1)
                            if month == 12:
                                partition_end = datetime(year + 1, 1, 1) - timedelta(days=1)
                            else:
                                partition_end = datetime(year, month + 1, 1) - timedelta(days=1)

                            # Check if partition overlaps with requested range
                            # Convert string dates to datetime objects if needed
                            if start_date and isinstance(start_date, str):
                                try:
                                    range_start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                                except:
                                    range_start = datetime.min
                            else:
                                range_start = start_date if start_date else datetime.min
                            
                            if end_date and isinstance(end_date, str):
                                try:
                                    range_end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                                except:
                                    range_end = datetime.max
                            else:
                                range_end = end_date if end_date else datetime.max

                            if partition_end >= range_start and partition_start <= range_end:
                                filtered_partitioned_dirs.append(file_path)
                        else:
                            # If we can't parse the date, include it (fallback)
                            filtered_partitioned_dirs.append(file_path)
                else:
                    filtered_partitioned_dirs = partitioned_dirs

                for file_path in sorted(filtered_partitioned_dirs):
                    try:
                        # For processed data, it might be partitioned - recursively find all parquet files
                        all_parquet_files = []
                        for root, dirs, files_in_dir in os.walk(file_path):
                            for file in files_in_dir:
                                if file.endswith('.parquet'):
                                    all_parquet_files.append(os.path.join(root, file))

                        # Sort files for consistent ordering
                        all_parquet_files.sort()

                        for pf in all_parquet_files:
                            df = self.parquet_utils.safe_read_parquet(pf, columns=columns)
                            if df is not None and not df.empty:
                                # Apply date filtering to each partitioned file
                                if start_date is not None or end_date is not None:
                                    df = self._apply_date_filter_to_dataframe(df, start_date, end_date)
                                    if df is not None and not df.empty:
                                        dataframes.append(df)
                                else:
                                    dataframes.append(df)
                    except Exception as e:
                        self.logger.warning(f"Could not read partitioned directory {file_path}: {e}")

            # Handle other files (non-consolidated, non-partitioned)
            other_files = [f for f in files if f.is_file() and 'consolidated' not in f.name.lower()]
            if not dataframes:  # Only use other files if no consolidated or partitioned data was found
                for file_path in sorted(other_files):
                    try:
                        df = self.parquet_utils.safe_read_parquet(str(file_path), columns=columns)
                        if df is not None and not df.empty:
                            dataframes.append(df)
                    except Exception as e:
                        self.logger.warning(f"Could not read file {file_path}: {e}")

            if not dataframes:
                self.logger.warning(f"No valid data found for {symbol} {interval}")
                return None

            # Combine all dataframes
            if dataframes:
                # Only log successful data combination with summary info
                total_records_before_combine = sum(len(df) for df in dataframes)
                start_date_str = start_date.date() if hasattr(start_date, 'date') else str(start_date)
                end_date_str = end_date.date() if hasattr(end_date, 'date') else str(end_date)
                period_info = f"from {start_date_str} to {end_date_str}" if start_date and end_date else "full period"
                self.logger.info(f"✅ Combining {len(dataframes)} dataframes for {symbol} {interval} {period_info} -> {total_records_before_combine} records")
            combined_df = pd.concat(dataframes, ignore_index=False)

            # Normalize index types before sorting to prevent Timestamp vs int comparison errors
            if not combined_df.empty and len(combined_df.index) > 0:
                # Convert all index values to pandas Timestamp if they aren't already
                try:
                    # Handle mixed timezone values more robustly
                    if hasattr(combined_df.index, 'dtype') and 'datetime' in str(combined_df.index.dtype):
                        # If already datetime, normalize timezone info
                        if hasattr(combined_df.index, 'tz') and combined_df.index.tz is not None:
                            combined_df.index = combined_df.index.tz_convert('UTC').tz_localize(None)
                    else:
                        # Convert non-datetime index, forcing timezone-naive
                        combined_df.index = pd.to_datetime(combined_df.index, utc=True).tz_localize(None)
                except Exception as e:
                    self.logger.warning(f"Could not convert index to datetime: {e}")
                    # If conversion fails, try to sort by converting to numeric
                    try:
                        combined_df.index = pd.to_numeric(combined_df.index, errors='coerce')
                    except Exception as e2:
                        self.logger.warning(f"Could not convert index to numeric either: {e2}")

            combined_df = combined_df.sort_index()

            # Remove true duplicates - with detailed logging
            records_before_dedup = len(combined_df)
            self.logger.info(f"🔍 DEBUG: About to check for duplicates in {records_before_dedup} records")
            self.logger.info(f"🔍 DEBUG: Combined DF date range: {combined_df.index.min()} to {combined_df.index.max()}")

            # Check for duplicate indices first
            index_duplicates = combined_df.index.duplicated(keep=False)  # keep=False marks all duplicates

            if index_duplicates.sum() > 0:
                self.logger.info(f"🔍 Found {index_duplicates.sum():,} records with duplicate timestamps")

                # DEBUG: Print some statistics about the duplicates
                self.logger.info(f"🔍 DEBUG: Total unique duplicate timestamps: {len(combined_df.index[index_duplicates].unique())}")
                self.logger.info(f"🔍 DEBUG: Sample duplicate timestamps: {combined_df.index[index_duplicates].unique()[:5].tolist()}")

                # For records with duplicate timestamps, check if they have identical data
                # Define key columns that should be identical for true duplicates
                key_columns = ['open', 'high', 'low', 'close', 'volume', 'open_time', 'close_time']

                # Create a subset with only the key columns for comparison
                subset_df = combined_df[key_columns].copy()

                # For each group of duplicate timestamps, check if data values are identical
                true_duplicates = []
                data_duplicates = 0

                duplicate_timestamps = combined_df.index[index_duplicates].unique()

                for timestamp in duplicate_timestamps:
                    # Get all records for this timestamp
                    timestamp_records = combined_df[combined_df.index == timestamp]

                    if len(timestamp_records) > 1:
                        # DEBUG: Print info about this timestamp group
                        self.logger.info(f"🔍 DEBUG: Timestamp {timestamp} has {len(timestamp_records)} records")

                        # Check if all records for this timestamp are identical across key columns
                        first_record = timestamp_records.iloc[0]
                        all_identical = True

                        for i in range(1, len(timestamp_records)):
                            current_record = timestamp_records.iloc[i]
                            # Check if all key columns match
                            if not all(first_record[col] == current_record[col] for col in key_columns):
                                all_identical = False
                                # DEBUG: Show what differs
                                differing_cols = [col for col in key_columns if first_record[col] != current_record[col]]
                                self.logger.info(f"🔍 DEBUG: Records differ in columns: {differing_cols}")
                                break

                        if all_identical:
                            # All records are identical - keep only the last one, mark others as duplicates
                            true_duplicates.extend(timestamp_records.index[:-1].tolist())
                            data_duplicates += len(timestamp_records) - 1
                            self.logger.info(f"🔍 DEBUG: Found {len(timestamp_records)} identical records for {timestamp}")
                        else:
                            self.logger.info(f"🔍 DEBUG: Records for {timestamp} have different data - keeping all")
                        # If not all identical, keep all records (they represent different data at same timestamp)

                # Create mask for true duplicates only
                if true_duplicates:
                    duplicate_mask = combined_df.index.isin(true_duplicates)
                    num_duplicates = duplicate_mask.sum()

                    # Get sample of duplicate timestamps for debugging
                    duplicate_indices = combined_df.index[duplicate_mask]
                    sample_duplicates = duplicate_indices[:10].tolist() if len(duplicate_indices) > 0 else []

                    # Count how many times each timestamp appears
                    index_counts = combined_df.index.value_counts()
                    most_duplicated = index_counts[index_counts > 1].head(5)

                    self.logger.warning(
                        f"⚠️ 🔍 Found {num_duplicates:,} true duplicate records ({num_duplicates/records_before_dedup*100:.2f}% of data)\n"
                        f"   📊 Records before dedup: {records_before_dedup:,}\n"
                        f"   🔢 Unique timestamps: {len(index_counts):,}\n"
                        f"   🔝 Most duplicated timestamps:\n{most_duplicated.to_string()}\n"
                        f"   🧪 Sample duplicate timestamps: {sample_duplicates}"
                    )
                else:
                    self.logger.info(f"✅ No true duplicates found - all duplicate timestamps have different data values")
                    duplicate_mask = pd.Series(False, index=combined_df.index)
                    num_duplicates = 0

                # DEBUG: Final summary
                self.logger.info(f"🔍 DEBUG: Final duplicate summary - True duplicates: {num_duplicates}, Records with same timestamp but different data: {index_duplicates.sum() - num_duplicates}")
            else:
                self.logger.info(f"✅ No duplicate timestamps found")
                duplicate_mask = pd.Series(False, index=combined_df.index)
                num_duplicates = 0

            combined_df = combined_df[~duplicate_mask]

            # Apply date filtering if specified
            if start_date is not None or end_date is not None:

                # Convert timestamps to datetime for proper filtering
                timestamp_col = None
                if 'timestamp' in combined_df.columns:
                    # Use timestamp column for filtering
                    try:
                        # First, clean the timestamp column by removing NaN and infinite values
                        timestamp_series = combined_df['timestamp'].copy()

                        # Remove NaN and infinite values
                        valid_mask = pd.notna(timestamp_series) & np.isfinite(timestamp_series)
                        if not valid_mask.all():
                            invalid_count = len(timestamp_series) - valid_mask.sum()
                            invalid_indices = timestamp_series.index[~valid_mask].tolist()

                            # Get sample of invalid values for debugging
                            invalid_sample = timestamp_series[~valid_mask].head(5).tolist()

                            # Analyze types of invalid values
                            nan_count = timestamp_series.isna().sum()
                            inf_count = np.isinf(timestamp_series.replace([np.inf, -np.inf], np.nan).dropna()).sum()

                            # Check if there are valid timestamps to compare
                            valid_timestamps = timestamp_series[valid_mask]
                            timestamp_range = f"{valid_timestamps.min()} to {valid_timestamps.max()}" if len(valid_timestamps) > 0 else "No valid timestamps"

                            self.logger.warning(
                                f"⚠️ Found {invalid_count:,} invalid timestamps (NaN or inf) in 'timestamp' column, removing them\n"
                                f"   📊 Dataframe info: {len(combined_df):,} rows × {len(combined_df.columns)} columns\n"
                                f"   📋 Columns: {list(combined_df.columns)}\n"
                                f"   🔢 Invalid breakdown: {nan_count:,} NaN, {inf_count:,} Inf\n"
                                f"   📈 Valid timestamp range: {timestamp_range}\n"
                                f"   💯 Invalid percentage: {invalid_count/len(timestamp_series)*100:.2f}%\n"
                                f"   🔍 Invalid indices (first 10): {invalid_indices[:10]}\n"
                                f"   🧪 Sample invalid values: {invalid_sample}"
                            )
                            timestamp_series = timestamp_series[valid_mask]
                            combined_df = combined_df[valid_mask]
                            self.logger.info(f"🔧 After cleaning invalid timestamps: {len(combined_df):,} records (removed {invalid_count:,})")

                        # Try to convert timestamps to datetime using seconds first
                        timestamp_col = pd.to_datetime(timestamp_series, unit='s')
                    except (OverflowError, FloatingPointError):
                        # If overflow occurs, try with milliseconds
                        try:
                            timestamp_col = pd.to_datetime(timestamp_series, unit='ms')
                        except (OverflowError, FloatingPointError):
                            # If still failing, convert to int64 and try again
                            timestamp_int = timestamp_series.astype('int64')
                            timestamp_col = pd.to_datetime(timestamp_int, unit='s')
                elif 'open_time' in combined_df.columns:
                    # Handle case where data has open_time but no timestamp column
                    try:
                        # First, clean the open_time column by removing NaN and infinite values
                        open_time_series = combined_df['open_time'].copy()

                        # Remove NaN and infinite values
                        valid_mask = pd.notna(open_time_series) & np.isfinite(open_time_series)
                        if not valid_mask.all():
                            invalid_count = len(open_time_series) - valid_mask.sum()
                            invalid_indices = open_time_series.index[~valid_mask].tolist()

                            # Get sample of invalid values for debugging
                            invalid_sample = open_time_series[~valid_mask].head(5).tolist()

                            # Analyze types of invalid values
                            nan_count = open_time_series.isna().sum()
                            inf_count = np.isinf(open_time_series.replace([np.inf, -np.inf], np.nan).dropna()).sum()

                            # Check if there are valid timestamps to compare
                            valid_timestamps = open_time_series[valid_mask]
                            timestamp_range = f"{valid_timestamps.min()} to {valid_timestamps.max()}" if len(valid_timestamps) > 0 else "No valid timestamps"

                            self.logger.warning(
                                f"⚠️ Found {invalid_count:,} invalid timestamps (NaN or inf) in 'open_time' column, removing them\n"
                                f"   📊 Dataframe info: {len(combined_df):,} rows × {len(combined_df.columns)} columns\n"
                                f"   📋 Columns: {list(combined_df.columns)}\n"
                                f"   🔢 Invalid breakdown: {nan_count:,} NaN, {inf_count:,} Inf\n"
                                f"   📈 Valid timestamp range: {timestamp_range}\n"
                                f"   💯 Invalid percentage: {invalid_count/len(open_time_series)*100:.2f}%\n"
                                f"   🔍 Invalid indices (first 10): {invalid_indices[:10]}\n"
                                f"   🧪 Sample invalid values: {invalid_sample}"
                            )
                            open_time_series = open_time_series[valid_mask]
                            combined_df = combined_df[valid_mask]
                            self.logger.info(f"🔧 After cleaning invalid open_time values: {len(combined_df):,} records (removed {invalid_count:,})")

                        # open_time is typically in milliseconds for exchange data
                        timestamp_col = pd.to_datetime(open_time_series, unit='ms')
                        self.logger.info(f"🔧 Using 'open_time' column for timestamp filtering")
                    except (OverflowError, FloatingPointError):
                        # If overflow occurs, try with seconds
                        try:
                            timestamp_col = pd.to_datetime(open_time_series, unit='s')
                        except (OverflowError, FloatingPointError):
                            # If still failing, convert to int64 and try again
                            timestamp_int = open_time_series.astype('int64')
                            timestamp_col = pd.to_datetime(timestamp_int, unit='ms')

                    max_date = timestamp_col.max()
                    min_date = timestamp_col.min()

                    # Check if we have valid dates (not NaT)
                    if pd.isna(max_date) or pd.isna(min_date):
                        self.logger.warning(f"⚠️ No valid timestamps found in data")
                        return None

                    min_date_str = min_date.date() if hasattr(min_date, 'date') else str(min_date)
                    max_date_str = max_date.date() if hasattr(max_date, 'date') else str(max_date)
                    tprint(f"📅 Available data range: {min_date_str} to {max_date_str}")

                    # If the requested date range doesn't match available data, use last 20 days (light mode default)
                    if start_date is not None:
                        # Convert string to datetime if needed
                        if isinstance(start_date, str):
                            # Handle numpy array inputs
                            if isinstance(start_date, np.ndarray):
                                if start_date.size == 1:
                                    start_date = start_date.item()
                                else:
                                    self.logger.warning(f"Invalid start_date format: numpy array with {start_date.size} elements")
                                    return None

                            start_date = pd.to_datetime(start_date)
                        start_date_str = start_date.date() if hasattr(start_date, 'date') else str(start_date)
                        max_date_str = max_date.date() if hasattr(max_date, 'date') else str(max_date)
                        min_date_str = min_date.date() if hasattr(min_date, 'date') else str(min_date)
                        if start_date.date() > max_date.date():
                            tprint(f"⚠️ Requested start_date {start_date_str} is beyond available data (max: {max_date_str})")
                            tprint(f"📅 Using all available data from {min_date_str} to {max_date_str}")
                            start_date = min_date
                            end_date = max_date

                    if end_date is not None:
                        # Convert string to datetime if needed
                        if isinstance(end_date, str):
                            # Handle numpy array inputs
                            if isinstance(end_date, np.ndarray):
                                if end_date.size == 1:
                                    end_date = end_date.item()
                                else:
                                    self.logger.warning(f"Invalid end_date format: numpy array with {end_date.size} elements")
                                    return None

                            end_date = pd.to_datetime(end_date)
                        end_date_str = end_date.date() if hasattr(end_date, 'date') else str(end_date)
                        if end_date.date() > max_date.date():
                            tprint(f"⚠️ Requested end_date {end_date_str} is beyond available data (max: {max_date_str})")
                            tprint(f"📅 Using all available data from {min_date_str} to {max_date_str}")
                            start_date = min_date
                            end_date = max_date

                    if start_date is not None and end_date is not None and start_date.date() > end_date.date():
                        tprint(f"⚠️ Invalid date range: start_date {start_date_str} > end_date {end_date_str}")
                        tprint(f"📅 Using all available data from {min_date_str} to {max_date_str}")
                        start_date = min_date
                        end_date = max_date

                    final_start_str = start_date.date() if hasattr(start_date, 'date') else str(start_date)
                    final_end_str = end_date.date() if hasattr(end_date, 'date') else str(end_date)
                    tprint(f"📅 Final date range: {final_start_str} to {final_end_str}")

                    # Apply the date filtering
                    mask = (timestamp_col >= start_date) & (timestamp_col <= end_date)
                    combined_df = combined_df[mask]

                    if len(combined_df) == 0:
                        tprint(f"⚠️ No data found in date range {final_start_str} to {final_end_str}")
                        tprint("🔄 Applying fallback: Using last 20 days of available data")
                        # Use the new fallback method to get last x days of available data
                        combined_df = self._get_last_x_days_fallback(combined_df, x_days=20)
                else:
                    # Fallback to index-based filtering
                    try:
                        # Convert index to datetime if it's not already
                        if not isinstance(combined_df.index, pd.DatetimeIndex):
                            # Try to use open_time as index if available
                            if 'open_time' in combined_df.columns:
                                combined_df.index = pd.to_datetime(combined_df['open_time'], unit='ms')
                                self.logger.info(f"🔧 Using 'open_time' column as DataFrame index")
                            else:
                                combined_df.index = pd.to_datetime(combined_df.index, unit='s')

                        if start_date is not None:
                            combined_df = combined_df[combined_df.index >= start_date]
                        if end_date is not None:
                            combined_df = combined_df[combined_df.index <= end_date]

                        # Check if we have data after filtering, if not use fallback
                        if len(combined_df) == 0:
                            tprint(f"⚠️ No data found in date range after index-based filtering")
                            tprint("🔄 Applying fallback: Using last 20 days of available data")
                            # Use the new fallback method to get last x days of available data
                            combined_df = self._get_last_x_days_fallback(combined_df, x_days=20)

                    except Exception as e:
                        self.logger.warning(f"Could not apply date filtering: {e}")
                        # Continue without filtering if conversion fails

            # Only log final successful data retrieval with timeframe and period info
            final_start_str = start_date.date() if hasattr(start_date, 'date') else str(start_date) if start_date else None
            final_end_str = end_date.date() if hasattr(end_date, 'date') else str(end_date) if end_date else None
            period_info = f"from {final_start_str} to {final_end_str}" if start_date and end_date else "full period"
            self.logger.info(f"✅ Data loaded: {symbol} {interval} {period_info} -> {len(combined_df)} records")
            return combined_df

        except Exception as e:
            self.logger.exception(f"❌ Failed to read data: {e}")
            return None

    def write_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        data_type: str = "raw",
        overwrite: bool = False
    ) -> bool:
        """Write klines data to parquet files.

        Args:
            df: DataFrame to write
            symbol: Trading symbol
            interval: Data interval
            data_type: 'raw' or 'processed'
            overwrite: Whether to overwrite existing files

        Returns:
            True if successful, False otherwise
        """
        try:
            if df is None or df.empty:
                self.logger.warning("Cannot write empty DataFrame")
                return False

            if data_type == "raw":
                data_dir = self.raw_data_dir / symbol.lower() / "raw"
            else:
                data_dir = self.processed_data_dir / symbol.lower() / "processed"

            data_dir.mkdir(parents=True, exist_ok=True)

            # Add metadata if not present
            if 'symbol' not in df.columns:
                df = df.copy()
                df['symbol'] = symbol
            if 'interval' not in df.columns:
                df = df.copy()
                df['interval'] = interval

            # Add time-based columns for partitioning
            df_with_partitions = df.copy()
            df_with_partitions['year'] = df_with_partitions.index.year
            df_with_partitions['month'] = df_with_partitions.index.month
            df_with_partitions['day'] = df_with_partitions.index.day

            if data_type == "raw":
                # For raw data, save as monthly files
                for (year, month), month_data in df_with_partitions.groupby([df_with_partitions.index.year, df_with_partitions.index.month]):
                    filename = f"{symbol.lower()}_{interval}_{year}_{month:02d}.parquet"
                    filepath = data_dir / filename

                    if filepath.exists() and not overwrite:
                        # Merge with existing data
                        existing_df = self.parquet_utils.safe_read_parquet(str(filepath))
                        if existing_df is not None:
                            combined_df = pd.concat([existing_df, month_data], ignore_index=False)

                            # Normalize index types before sorting to prevent Timestamp vs int comparison errors
                            if not combined_df.empty and len(combined_df.index) > 0:
                                try:
                                    # Handle mixed timezone values more robustly
                                    if hasattr(combined_df.index, 'dtype') and 'datetime' in str(combined_df.index.dtype):
                                        # If already datetime, normalize timezone info
                                        if hasattr(combined_df.index, 'tz') and combined_df.index.tz is not None:
                                            combined_df.index = combined_df.index.tz_convert('UTC').tz_localize(None)
                                    else:
                                        # Convert non-datetime index, forcing timezone-naive
                                        combined_df.index = pd.to_datetime(combined_df.index, utc=True).tz_localize(None)
                                except Exception as e:
                                    self.logger.warning(f"Could not convert index to datetime: {e}")
                                    try:
                                        combined_df.index = pd.to_numeric(combined_df.index, errors='coerce')
                                    except Exception as e2:
                                        self.logger.warning(f"Could not convert index to numeric either: {e2}")

                            combined_df = combined_df.sort_index()
                            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        else:
                            combined_df = month_data
                    else:
                        combined_df = month_data

                    # Optimize data types
                    combined_df = self.data_processor.optimize_dataframe_dtypes(combined_df)

                    # Save file
                    combined_df.to_parquet(filepath, index=True, compression='snappy')
                    self.logger.info(f"💾 Saved {len(combined_df)} records to {filename}")

            else:
                # For processed data, save as partitioned parquet
                output_path = data_dir / f"{symbol.lower()}_{interval}"

                if output_path.exists() and not overwrite:
                    self.logger.warning(f"Processed data already exists for {symbol} {interval}")
                    return False

                # Optimize data types
                df_with_partitions = self.data_processor.optimize_feature_engineering_pipeline(
                    df_with_partitions, stage="output"
                )

                # Save as partitioned parquet
                df_with_partitions.to_parquet(
                    output_path,
                    partition_cols=['year', 'month'],
                    index=True,
                    compression='snappy',
                    engine='pyarrow'
                )

                self.logger.info(f"💾 Saved processed data: {symbol} {interval} ({len(df)} records)")

            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to write data: {e}")
            return False

    def update_data(
        self,
        new_data: pd.DataFrame,
        symbol: str,
        interval: str,
        data_type: str = "raw"
    ) -> bool:
        """Update existing data with new data.

        Args:
            new_data: New data to add
            symbol: Trading symbol
            interval: Data interval
            data_type: 'raw' or 'processed'

        Returns:
            True if successful, False otherwise
        """
        try:
            if new_data is None or len(new_data) == 0:
                return True

            # Read existing data
            existing_data = self.read_data(symbol, interval, data_type=data_type)

            if existing_data is None or len(existing_data) == 0:
                # No existing data, just write new data
                return self.write_data(new_data, symbol, interval, data_type, overwrite=True)

            # Combine with existing data
            combined_data = pd.concat([existing_data, new_data], ignore_index=False)
            combined_data = combined_data.sort_index()

            # Remove duplicates (keep last occurrence)
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]

            # Write updated data
            return self.write_data(combined_data, symbol, interval, data_type, overwrite=True)

        except Exception as e:
            self.logger.exception(f"❌ Failed to update data: {e}")
            return False

    def delete_data(
        self,
        symbol: str,
        interval: str,
        data_type: str = "raw",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """Delete data for a symbol and interval.

        Args:
            symbol: Trading symbol
            interval: Data interval
            data_type: 'raw' or 'processed'
            start_date: Start date for deletion (optional)
            end_date: End date for deletion (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            if data_type == "raw":
                data_dir = self.raw_data_dir / symbol.lower() / "raw"
                pattern = f"{symbol.lower()}_{interval}_*.parquet"
            else:
                data_dir = self.processed_data_dir / symbol.lower() / "processed"
                pattern = f"{symbol.lower()}_{interval}"

            if not data_dir.exists():
                return True  # Nothing to delete

            files = list(data_dir.glob(f"{pattern}*"))

            if not files:
                return True  # Nothing to delete

            if start_date is None and end_date is None:
                # Delete all data
                for file_path in files:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        import shutil
                        shutil.rmtree(file_path)

                self.logger.info(f"🗑️ Deleted all data for {symbol} {interval}")
                return True

            # Delete specific date range
            deleted_files = 0
            for file_path in files:
                try:
                    if file_path.is_file():
                        # Check if file contains data in the specified range
                        df = self.parquet_utils.safe_read_parquet(str(file_path))
                        if df is not None and not df.empty:
                            file_start = df.index.min()
                            file_end = df.index.max()

                            # Check if file overlaps with deletion range
                            if (start_date is None or file_end >= start_date) and \
                               (end_date is None or file_start <= end_date):

                                if start_date is not None and end_date is not None:
                                    # Partial deletion - need to filter and rewrite
                                    filtered_df = df[(df.index < start_date) | (df.index > end_date)]
                                    if filtered_df.empty:
                                        file_path.unlink()
                                    else:
                                        filtered_df.to_parquet(file_path, index=True, compression='snappy')
                                else:
                                    file_path.unlink()

                                deleted_files += 1

                except Exception as e:
                    self.logger.warning(f"Could not process {file_path}: {e}")

            self.logger.info(f"🗑️ Deleted {deleted_files} files for {symbol} {interval}")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to delete data: {e}")
            return False

    def list_available_data(self) -> Dict[str, List[str]]:
        """List all available data.

        Returns:
            Dictionary mapping symbols to available intervals
        """
        try:
            available_data = {}

            # Check raw data
            for symbol_dir in self.raw_data_dir.iterdir():
                if symbol_dir.is_dir():
                    symbol = symbol_dir.name.upper()
                    raw_dir = symbol_dir / "raw"
                    if raw_dir.exists():
                        intervals = set()
                        for file_path in raw_dir.glob("*.parquet"):
                            # Extract interval from filename
                            parts = file_path.stem.split('_')
                            if len(parts) >= 2:
                                interval = parts[1]
                                intervals.add(interval)

                        if intervals:
                            available_data[symbol] = list(intervals)

            # Check processed data
            for symbol_dir in self.processed_data_dir.iterdir():
                if symbol_dir.is_dir():
                    symbol = symbol_dir.name.upper()
                    processed_dir = symbol_dir / "processed"
                    if processed_dir.exists():
                        intervals = set()
                        for item in processed_dir.iterdir():
                            if item.is_dir():
                                # Extract interval from directory name
                                parts = item.name.split('_')
                                if len(parts) >= 2:
                                    interval = parts[1]
                                    intervals.add(interval)

                        if intervals:
                            if symbol in available_data:
                                available_data[symbol].extend(list(intervals))
                            else:
                                available_data[symbol] = list(intervals)

            return available_data

        except Exception as e:
            self.logger.exception(f"❌ Failed to list available data: {e}")
            return {}

    def read_last_x_days_data(
        self,
        symbol: str,
        interval: str,
        x_days: int = 20,
        data_type: str = "raw",
        columns: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Read the last x days of available data for a symbol and interval.

        Args:
            symbol: Trading symbol
            interval: Data interval
            x_days: Number of days to go back from the maximum available date
            data_type: 'raw' or 'processed'
            columns: List of columns to read

        Returns:
            DataFrame with the last x days of klines data or None if not found
        """
        try:
            # First, get all available data without date filtering
            all_data = self.read_data(symbol, interval, data_type=data_type, columns=columns)

            if all_data is None or len(all_data) == 0:
                self.logger.warning(f"No data available for {symbol} {interval}")
                return None

            # Apply the last x days fallback
            last_x_days_data = self._get_last_x_days_fallback(all_data, x_days)

            if last_x_days_data is not None and not len(last_x_days_data) == 0:
                # Only log successful data retrieval with timeframe and period info
                self.logger.info(f"✅ Data loaded: {symbol} {interval} last {x_days} days -> {len(last_x_days_data)} records")
                return last_x_days_data
            else:
                self.logger.warning(f"❌ Could not get last {x_days} days of data for {symbol} {interval}")
                return None

        except Exception as e:
            self.logger.exception(f"❌ Failed to read last {x_days} days data: {e}")
            return None

    def get_data_statistics(self, symbol: str, interval: str, data_type: str = "raw") -> Dict[str, Any]:
        """Get detailed statistics for data.

        Args:
            symbol: Trading symbol
            interval: Data interval
            data_type: 'raw' or 'processed'

        Returns:
            Dictionary with detailed statistics
        """
        try:
            # Get basic info
            info = self.get_data_info(symbol, interval, data_type)

            if not info["available"]:
                return info

            # Read a sample of data for detailed statistics
            sample_data = self.read_data(symbol, interval, data_type=data_type)

            if sample_data is None or len(sample_data) == 0:
                return info

            # Calculate additional statistics
            stats = info.copy()
            stats.update({
                "columns": list(sample_data.columns),
                "dtypes": {col: str(dtype) for col, dtype in sample_data.dtypes.items()},
                "memory_usage_mb": sample_data.memory_usage(deep=True).sum() / (1024 * 1024),
                "null_counts": sample_data.isnull().sum().to_dict(),
                "price_range": {
                    "min": sample_data['close'].min() if 'close' in sample_data.columns else None,
                    "max": sample_data['close'].max() if 'close' in sample_data.columns else None,
                    "mean": sample_data['close'].mean() if 'close' in sample_data.columns else None
                } if 'close' in sample_data.columns else None,
                "volume_stats": {
                    "min": sample_data['volume'].min() if 'volume' in sample_data.columns else None,
                    "max": sample_data['volume'].max() if 'volume' in sample_data.columns else None,
                    "mean": sample_data['volume'].mean() if 'volume' in sample_data.columns else None
                } if 'volume' in sample_data.columns else None
            })

            return stats

        except Exception as e:
            self.logger.exception(f"❌ Failed to get data statistics: {e}")
            return {"error": str(e)}

# Convenience functions
def get_klines_manager(data_dir: str = "historical_data", exchange: str = "binance") -> KlinesParquetManager:
    """Get a klines parquet manager instance.

    Args:
        data_dir: Base directory for data storage
        exchange: Exchange name (binance, bingx, mexc, etc.)

    Returns:
        KlinesParquetManager instance
    """
    return KlinesParquetManager(data_dir, exchange)

def read_ethusdt_data(
    interval: str = "1m",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    data_type: str = "raw",
    data_dir: str = "historical_data"
) -> Optional[pd.DataFrame]:
    """Read ETHUSDT data.

    Args:
        interval: Data interval
        start_date: Start date for filtering
        end_date: End date for filtering
        data_type: 'raw' or 'processed'
        data_dir: Base directory for data storage

    Returns:
        DataFrame with ETHUSDT data or None if not found
    """
    manager = get_klines_manager(data_dir)
    return manager.read_data("ETHUSDT", interval, start_date, end_date, data_type)

def read_ethusdt_last_x_days(
    interval: str = "1m",
    x_days: int = 20,
    data_type: str = "raw",
    data_dir: str = "historical_data"
) -> Optional[pd.DataFrame]:
    """Read the last x days of ETHUSDT data.

    Args:
        interval: Data interval
        x_days: Number of days to go back from the maximum available date
        data_type: 'raw' or 'processed'
        data_dir: Base directory for data storage

    Returns:
        DataFrame with the last x days of ETHUSDT data or None if not found
    """
    manager = get_klines_manager(data_dir)
    return manager.read_last_x_days_data("ETHUSDT", interval, x_days, data_type)

# Backward compatibility functions
def save_klines_to_parquet(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    data_type: str = "raw",
    overwrite: bool = False,
    data_dir: str = "historical_data",
    exchange: str = "binance"
) -> bool:
    """Save klines data to parquet files (backward compatibility function).

    Args:
        df: DataFrame to write
        symbol: Trading symbol
        interval: Data interval
        data_type: 'raw' or 'processed'
        overwrite: Whether to overwrite existing files
        data_dir: Base directory for data storage
        exchange: Exchange name (binance, bingx, mexc, etc.)

    Returns:
        True if successful, False otherwise
    """
    manager = get_klines_manager(data_dir, exchange)
    return manager.write_data(df, symbol, interval, data_type, overwrite)

def load_klines_from_parquet(
    symbol: str,
    interval: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    data_type: str = "raw",
    columns: Optional[List[str]] = None,
    data_dir: str = "historical_data",
    exchange: str = "binance"
) -> Optional[pd.DataFrame]:
    """Load klines data from parquet files (backward compatibility function).

    Args:
        symbol: Trading symbol
        interval: Data interval
        start_date: Start date for filtering
        end_date: End date for filtering
        data_type: 'raw' or 'processed'
        columns: List of columns to read
        data_dir: Base directory for data storage
        exchange: Exchange name (binance, bingx, mexc, etc.)

    Returns:
        DataFrame with klines data or None if not found
    """
    manager = get_klines_manager(data_dir, exchange)
    return manager.read_data(symbol, interval, start_date, end_date, data_type, columns)

def validate_klines_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate klines data (backward compatibility function).

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    if df is None or df.empty:
        return {
            "valid": False,
            "errors": ["DataFrame is empty or None"],
            "warnings": [],
            "info": {}
        }

    errors = []
    warnings = []
    info = {}

    # Basic structure validation
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    # Data type validation
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                warnings.append(f"Column '{col}' is not numeric")

    # Price validation (high >= low >= 0, etc.)
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        price_issues = df[(df['high'] < df['low']) | (df['low'] < 0)].index
        if len(price_issues) > 0:
            errors.append(f"Invalid price relationships found in {len(price_issues)} rows")

    # Volume validation
    if 'volume' in df.columns:
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            warnings.append(f"Found {negative_volume} rows with negative volume")

    # Timestamp validation
    if hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index):
        if not df.index.is_monotonic_increasing:
            warnings.append("Timestamp index is not monotonic increasing")

        # Check for duplicates
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate timestamps")

    # Calculate basic info
    info = {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "null_counts": df.isnull().sum().to_dict()
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "info": info
    }

def process_klines_data(
    df: pd.DataFrame,
    symbol: str = None,
    interval: str = None,
    data_type: str = "raw"
) -> pd.DataFrame:
    """Process klines data for enhanced analysis.

    Args:
        df: DataFrame with klines data
        symbol: Trading symbol (optional, for metadata)
        interval: Data interval (optional, for metadata)
        data_type: 'raw' or 'processed'

    Returns:
        Processed DataFrame with enhanced features
    """
    if df is None or df.empty:
        return df

    try:
        # Create a copy to avoid modifying original
        processed_df = df.copy()

        # Add metadata if provided
        if symbol is not None:
            processed_df['symbol'] = symbol
        if interval is not None:
            processed_df['interval'] = interval

        # Ensure proper datetime index
        if not isinstance(processed_df.index, pd.DatetimeIndex):
            if 'timestamp' in processed_df.columns:
                processed_df.index = pd.to_datetime(processed_df['timestamp'], unit='s')
            elif 'open_time' in processed_df.columns:
                processed_df.index = pd.to_datetime(processed_df['open_time'], unit='ms')

        # Sort by index
        processed_df = processed_df.sort_index()

        # Remove duplicates
        processed_df = processed_df[~processed_df.index.duplicated(keep='last')]

        # Basic data validation
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col in processed_df.columns:
                # Remove any non-finite values
                processed_df = processed_df[np.isfinite(processed_df[col])]

        return processed_df

    except Exception as e:
        logger = logging.getLogger("process_klines_data")
        logger.warning(f"Error processing klines data: {e}")
        return df

if __name__ == "__main__":
    # Example usage
    manager = get_klines_manager()

    # List available data
    available = manager.list_available_data()
    print(f"Available data: {available}")

    # Get data info
    info = manager.get_data_info("ETHUSDT", "1m", "raw")
    print(f"ETHUSDT 1m raw data info: {info}")

    # Read data with date range (will fallback to last 20 days if range not available)
    data = manager.read_data("ETHUSDT", "1m", data_type="raw")
    if data is not None:
        print(f"Loaded {len(data)} records")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")

    # Read last 30 days of data explicitly
    last_30_days = manager.read_last_x_days_data("ETHUSDT", "1m", x_days=30, data_type="raw")
    if last_30_days is not None:
        print(f"Last 30 days: {len(last_30_days)} records")
        print(f"Date range: {last_30_days.index.min()} to {last_30_days.index.max()}")

    # Test convenience function
    ethusdt_last_7_days = read_ethusdt_last_x_days("1m", x_days=7)
    if ethusdt_last_7_days is not None:
        print(f"ETHUSDT last 7 days: {len(ethusdt_last_7_days)} records")

        # Test backward compatibility functions
        validation_result = validate_klines_data(ethusdt_last_7_days)
        print(f"Validation result: {validation_result['valid']}")
        if validation_result['errors']:
            print(f"Errors: {validation_result['errors']}")
        if validation_result['warnings']:
            print(f"Warnings: {validation_result['warnings']}")
