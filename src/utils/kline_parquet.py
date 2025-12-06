"""
KlinesParquetManager - Specialized Parquet Storage for OHLCV/Time-Series Data

This module provides specialized storage for historical market data (OHLCV/klines)
using Parquet format with optimizations for time-series data.

RESPONSIBILITIES:
----------------
1. Historical OHLCV Data Storage:
   - Candlestick/klines data (open, high, low, close, volume)
   - Raw market data from exchanges
   - Time-series data with temporal ordering

2. Parquet Optimization:
   - Efficient compression (ZSTD, LZ4, Snappy)
   - Batch management for incremental updates
   - Memory-efficient chunking
   - Column-oriented storage for fast queries

3. Data Integrity:
   - Validation of OHLCV relationships
   - Gap detection and handling
   - Duplicate removal
   - Metadata tracking

4. Storage Organization:
   - Stores in historical_data/ directory
   - Organized by exchange/symbol/interval
   - Batch versioning for incremental updates

WHEN TO USE:
-----------
- Use this for: Historical OHLCV/klines data, raw market data
- Use serialization_utils for: Configs, models, metadata
- Use versioned_artifacts/ for: Feature DataFrames, training data

Example:
    from src.utils.kline_parquet import KlinesParquetManager

    manager = KlinesParquetManager()
    manager.store_klines(df, symbol="BTCUSDT", exchange="binance", interval="15m")
    loaded_data = manager.load_klines("BTCUSDT", "binance", "15m")

Features:
- Efficient parquet storage with compression
- Batch management for incremental data updates
- Data integrity validation
- Automatic directory structure management
- Exchange-agnostic data format
- Memory-efficient operations
- Comprehensive error handling
"""

import os
import gc
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.parquet_utils import ParquetUtils
from src.core.decorators import handles_errors, traced, log_execution_time

@dataclass
class KlinesMetadata:
    """Metadata for klines data batches."""
    symbol: str
    exchange: str
    interval: str
    batch_id: str
    start_time: datetime
    end_time: datetime
    record_count: int
    file_size_bytes: int
    compression_ratio: float
    created_at: datetime
    data_quality_score: float = 0.0
    gaps_detected: int = 0
    gaps_filled: int = 0
    resampled_intervals: List[str] = field(default_factory=list)
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StorageConfig:
    """Enhanced configuration for klines storage with optimization options."""
    base_dir: str = "historical_data"
    compression: str = "zstd"  # Better compression than snappy
    compression_level: int = 3  # ZSTD compression level
    index: bool = False  # Don't store index as separate column
    partition_by: List[str] = field(default_factory=lambda: ["exchange", "symbol", "interval"])
    max_file_size_mb: int = 100
    enable_metadata: bool = True
    enable_validation: bool = True
    row_group_size: int = 50000  # Optimized row group size
    use_dictionary_encoding: bool = True  # Enable dictionary encoding for categorical data
    enable_schema_optimization: bool = True  # Enable schema optimization
    enable_compression_analysis: bool = True  # Enable compression analysis

class KlinesParquetManager:
    """
    Manager for efficient klines data storage and retrieval using parquet format.

    Provides:
    - Efficient storage with compression
    - Batch management for incremental updates
    - Data integrity validation
    - Automatic directory structure management
    - Exchange-agnostic data format
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize the KlinesParquetManager.

        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self.base_dir = Path(self.config.base_dir)
        self.parquet_utils = ParquetUtils()
        self.logger = system_logger.getChild("KlinesParquetManager")

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Storage tracking
        self._metadata_cache: Dict[str, KlinesMetadata] = {}
        self._batch_counter: Dict[str, int] = {}
        self._compression_stats: Dict[str, Any] = {}

        # Column optimization mapping
        self.column_optimizations = {
            'timestamp': {'dtype': 'datetime64[ns]', 'nullable': False},
            'open': {'dtype': 'float32', 'nullable': False},
            'high': {'dtype': 'float32', 'nullable': False},
            'low': {'dtype': 'float32', 'nullable': False},
            'close': {'dtype': 'float32', 'nullable': False},
            'volume': {'dtype': 'float32', 'nullable': False},
            'symbol': {'dtype': 'category', 'nullable': False},
            'exchange': {'dtype': 'category', 'nullable': False},
            'interval': {'dtype': 'category', 'nullable': False},
        }

        self.logger.info(f"✅ KlinesParquetManager initialized with base_dir: {self.base_dir}")

    @handles_errors(default_return=False, context="KlinesParquetManager.store_klines")
    @traced()
    @log_execution_time()
    def store_klines(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str,
        batch_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store klines data in parquet format.

        Args:
            df: DataFrame containing klines data
            symbol: Trading symbol (e.g., "ETHUSDT")
            exchange: Exchange name (e.g., "binance")
            interval: Data interval (e.g., "1m")
            batch_id: Optional batch identifier
            metadata: Additional metadata to store

        Returns:
            True if storage was successful, False otherwise
        """
        if df is None or df.empty:
            tprint_error("❌ Cannot store empty DataFrame")
            return False

        try:
            # Generate batch ID if not provided
            if batch_id is None:
                batch_id = self._generate_batch_id(symbol, exchange, interval)

            # Apply comprehensive optimizations
            storage_df = self._apply_comprehensive_optimizations(df, symbol, exchange, interval)

            # Determine storage path
            storage_path = self._get_storage_path(symbol, exchange, interval, batch_id)

            # Ensure directory exists
            storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Get optimal parquet write parameters
            parquet_kwargs = self._get_optimal_parquet_kwargs(storage_df)

            # Store data with optimizations
            success = self._store_dataframe_optimized(storage_df, storage_path, parquet_kwargs)
            if not success:
                return False

            # Calculate compression statistics
            compression_stats = self._calculate_compression_stats(df, storage_df, storage_path)

            # Create and store metadata with compression stats
            klines_metadata = self._create_enhanced_metadata(
                storage_df, symbol, exchange, interval, batch_id,
                storage_path, metadata, compression_stats
            )

            # Store metadata
            self._store_metadata(klines_metadata, storage_path)

            # Update cache
            self._metadata_cache[f"{symbol}_{exchange}_{interval}_{batch_id}"] = klines_metadata

            tprint_success(f"✅ Stored {len(storage_df)} klines records for {symbol} {interval}")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to store klines data: {e}")
            return False

    @handles_errors(default_return=pd.DataFrame(), context="KlinesParquetManager.load_klines")
    @traced()
    @log_execution_time()
    def load_klines(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_id: Optional[str] = None,
        last_n_days: Optional[int] = None
    ) -> pd.DataFrame:
        """Load klines data from parquet files.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            start_time: Optional start time filter
            end_time: Optional end time filter
            batch_id: Optional specific batch to load
            last_n_days: Optional number of days to load from the latest available data
                        (overrides start_time/end_time if provided)

        Returns:
            DataFrame containing klines data
        """
        try:
            tprint(f"🐛 DEBUG [load_klines]: last_n_days={last_n_days}, start_time={start_time}, end_time={end_time}", "INFO")
            
            # If last_n_days is specified, load all data first to find the latest date
            if last_n_days is not None:
                tprint_info(f"📅 Loading last {last_n_days} days from latest available data")
                
                # Load all files without time filtering to find the latest date
                files = self._find_klines_files(symbol, exchange, interval, batch_id, None, None)
                if not files:
                    tprint_warning(f"⚠️ No klines files found for {symbol} {exchange} {interval}")
                    return pd.DataFrame()
                
                # Load and combine all data
                combined_df = self._load_and_combine_files(files, None, None)
                tprint_info(
                    f"📊 Combined dataset before time filter: shape={combined_df.shape if hasattr(combined_df, 'shape') else 'NA'}; "
                    f"index_type={type(combined_df.index).__name__ if hasattr(combined_df, 'index') else 'NA'}"
                )
                
                if combined_df.empty:
                    tprint_warning(f"⚠️ No data found for {symbol} {exchange} {interval}")
                    return pd.DataFrame()
                
                # Find the latest timestamp in the data. Support multiple layouts:
                # 1) DatetimeIndex
                # 2) 'timestamp' column
                # 3) 'open_time' / 'close_time' columns (Binance-style epochs)
                time_series: Optional[pd.Series]
                time_series = None
                time_source = "unknown"

                if isinstance(combined_df.index, pd.DatetimeIndex):
                    time_series = pd.Series(combined_df.index, index=combined_df.index)
                    time_source = "index"
                elif 'timestamp' in combined_df.columns:
                    # Generic timestamp column – let pandas infer details.
                    try:
                        ts = pd.to_datetime(combined_df['timestamp'], errors='coerce')
                        if ts.notna().any():
                            time_series = ts
                            time_source = "timestamp"
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to parse 'timestamp' column as datetime: {e}")

                # If we still don't have a time_series, try Binance-style
                # open_time/close_time columns which may be stored as seconds,
                # milliseconds, microseconds, or nanoseconds since epoch, or
                # already as datetime.
                if time_series is None:
                    def _infer_epoch_series(col_name: str) -> Optional[pd.Series]:
                        if col_name not in combined_df.columns:
                            return None

                        col = combined_df[col_name]

                        try:
                            # If already datetime-like, use directly.
                            if pd.api.types.is_datetime64_any_dtype(col):
                                ts_local = pd.to_datetime(col, errors='coerce')
                            else:
                                # Try numeric epoch-based parsing with multiple
                                # candidate units and a plausibility check on
                                # the resulting dates to avoid 1970-era
                                # artifacts from overflow or mis-scaled epochs.
                                with np.errstate(all='ignore'):
                                    col_numeric = pd.to_numeric(col, errors='coerce')
                                    finite = col_numeric[np.isfinite(col_numeric)]

                                    if finite.empty:
                                        ts_local = pd.to_datetime(col, errors='coerce')
                                    else:
                                        ts_local = None
                                        for unit in [
                                            'ms',  # most common for exchange klines
                                            's',   # sometimes stored as seconds
                                            'ns',  # high-precision storage
                                        ]:
                                            try:
                                                cand = pd.to_datetime(
                                                    col_numeric,
                                                    unit=unit,
                                                    errors='coerce',
                                                )
                                            except (OverflowError, ValueError, TypeError, FloatingPointError) as unit_exc:
                                                tprint_warning(
                                                    f"⚠️ Failed to parse '{col_name}' with unit={unit}: {unit_exc}; trying next candidate unit",
                                                )
                                                continue

                                            if not isinstance(cand, pd.Series) or not cand.notna().any():
                                                continue

                                            # Plausibility check: require
                                            # timestamps to be in a reasonable
                                            # trading window rather than near
                                            # the Unix epoch.
                                            max_ts = cand.max()
                                            min_ts = cand.min()
                                            if (
                                                isinstance(max_ts, pd.Timestamp)
                                                and isinstance(min_ts, pd.Timestamp)
                                                and max_ts.year >= 2000
                                                and max_ts.year <= 2100
                                            ):
                                                ts_local = cand
                                                break

                                        # Final fallback if no unit produced a
                                        # plausible range.
                                        if ts_local is None:
                                            ts_local = pd.to_datetime(col, errors='coerce')

                            if ts_local is None or not isinstance(ts_local, pd.Series):
                                return None

                            if ts_local.notna().any():
                                return ts_local
                        except Exception as parse_exc:
                            tprint_warning(
                                f"⚠️ Failed to interpret '{col_name}' as datetime: {parse_exc}",
                            )
                        return None

                    # Prefer open_time, then close_time
                    ts_open = _infer_epoch_series('open_time')
                    ts_close = _infer_epoch_series('close_time') if ts_open is None else None

                    if ts_open is not None and ts_open.notna().any():
                        time_series = ts_open
                        time_source = "open_time"
                    elif ts_close is not None and ts_close.notna().any():
                        time_series = ts_close
                        time_source = "close_time"

                if time_series is None or len(time_series) == 0:
                    tprint_error("❌ Cannot determine latest date - no usable time column or index found")
                    return pd.DataFrame()

                latest_date = pd.Timestamp(time_series.max())  # type: ignore[arg-type]

                # Calculate start date as latest_date - last_n_days
                start_date = latest_date - pd.Timedelta(days=last_n_days)  # type: ignore[arg-type]

                tprint_info(f"📊 Latest available data ({time_source}): {latest_date}")
                tprint_info(f"📊 Filtering to last {last_n_days} days: {start_date} to {latest_date}")

                # Normalize timezone information for consistent comparison
                if hasattr(time_series, 'dt') and getattr(time_series.dt, 'tz', None) is not None:
                    time_series = time_series.dt.tz_convert('UTC').dt.tz_localize(None)
                if hasattr(start_date, 'tz') and start_date.tz is not None:
                    start_date = start_date.tz_convert('UTC').tz_localize(None)

                # Filter the data to the last N days
                mask = time_series >= start_date
                combined_df = combined_df.loc[mask].copy()

                # If the index is not already a DatetimeIndex, set it from the
                # time series so downstream steps see a proper DateTimeIndex.
                if not isinstance(combined_df.index, pd.DatetimeIndex):
                    combined_df.index = pd.to_datetime(time_series[mask])

                tprint_success(f"✅ Loaded {len(combined_df)} klines records for {symbol} {interval} (last {last_n_days} days)")
                return combined_df
            
            # Original behavior: use start_time/end_time filters
            # Find relevant files (with pre-filtering by date range if provided)
            files = self._find_klines_files(symbol, exchange, interval, batch_id, start_time, end_time)
            if not files:
                tprint_warning(f"⚠️ No klines files found for {symbol} {exchange} {interval}")
                return pd.DataFrame()

            # Load and combine data
            combined_df = self._load_and_combine_files(files, start_time, end_time)

            if combined_df.empty:
                tprint_warning(f"⚠️ No data found for {symbol} {exchange} {interval}")
                return pd.DataFrame()

            tprint_success(f"✅ Loaded {len(combined_df)} klines records for {symbol} {interval}")
            return combined_df

        except Exception as e:
            tprint_error(f"❌ Failed to load klines data: {e}")
            return pd.DataFrame()

    @handles_errors(default_return=List[str], context="KlinesParquetManager.list_available_data")
    def list_available_data(self) -> List[Dict[str, Any]]:
        """List all available klines data.

        Returns:
            List of dictionaries containing available data information
        """
        try:
            available_data = []

            # Scan base directory for klines data
            for exchange_dir in self.base_dir.iterdir():
                if not exchange_dir.is_dir():
                    continue

                for symbol_dir in exchange_dir.iterdir():
                    if not symbol_dir.is_dir():
                        continue

                    klines_dir = symbol_dir / "klines"
                    if not klines_dir.exists():
                        continue

                    # Find parquet files
                    for file_path in klines_dir.glob("*.parquet"):
                        metadata = self._load_file_metadata(file_path)
                        if metadata:
                            available_data.append({
                                "symbol": metadata.symbol,
                                "exchange": metadata.exchange,
                                "interval": metadata.interval,
                                "batch_id": metadata.batch_id,
                                "start_time": metadata.start_time,
                                "end_time": metadata.end_time,
                                "record_count": metadata.record_count,
                                "file_size_mb": metadata.file_size_bytes / (1024 * 1024),
                                "created_at": metadata.created_at
                            })

            return available_data

        except Exception as e:
            tprint_error(f"❌ Failed to list available data: {e}")
            return []

    @handles_errors(default_return=False, context="KlinesParquetManager.update_klines")
    def update_klines(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str,
        append_mode: bool = True
    ) -> bool:
        """Update existing klines data.

        Args:
            df: New klines data
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            append_mode: If True, append to existing data; if False, replace

        Returns:
            True if update was successful, False otherwise
        """
        try:
            if append_mode:
                # Load existing data
                existing_df = self.load_klines(symbol, exchange, interval)

                if not existing_df.empty:
                    # Combine with existing data
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                    combined_df = combined_df.sort_values('timestamp')
                else:
                    combined_df = df
            else:
                combined_df = df

            # Store updated data
            return self.store_klines(combined_df, symbol, exchange, interval)

        except Exception as e:
            tprint_error(f"❌ Failed to update klines data: {e}")
            return False

    @handles_errors(default_return=False, context="KlinesParquetManager.delete_klines")
    def delete_klines(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        batch_id: Optional[str] = None
    ) -> bool:
        """Delete klines data.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            batch_id: Optional specific batch to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            files = self._find_klines_files(symbol, exchange, interval, batch_id)

            for file_path in files:
                # Delete parquet file
                if file_path.exists():
                    file_path.unlink()
                    tprint_info(f"🗑️ Deleted {file_path}")

                # Delete metadata file
                metadata_path = file_path.with_suffix('.metadata.json')
                if metadata_path.exists():
                    metadata_path.unlink()
                    tprint_info(f"🗑️ Deleted {metadata_path}")

            tprint_success(f"✅ Deleted klines data for {symbol} {exchange} {interval}")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to delete klines data: {e}")
            return False

    def _generate_batch_id(self, symbol: str, exchange: str, interval: str) -> str:
        """Generate a unique batch ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = self._batch_counter.get(f"{symbol}_{exchange}_{interval}", 0) + 1
        self._batch_counter[f"{symbol}_{exchange}_{interval}"] = counter
        return f"batch_{counter:03d}_{timestamp}"

    def _apply_comprehensive_optimizations(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str
    ) -> pd.DataFrame:
        """Apply comprehensive optimizations to DataFrame."""
        optimized_df = df.copy()

        # 1. Optimize data types
        optimized_df = self._optimize_dtypes(optimized_df)

        # 2. Sort data for better compression
        optimized_df = self._sort_for_compression(optimized_df)

        # 3. Remove unnecessary columns
        optimized_df = self._remove_unnecessary_columns(optimized_df)

        # 4. Optimize categorical data
        optimized_df = self._optimize_categorical_data(optimized_df)

        # 5. Handle missing values efficiently
        optimized_df = self._handle_missing_values(optimized_df)

        # 6. Add required columns if missing
        optimized_df = self._ensure_required_columns(optimized_df, symbol, exchange, interval)

        return optimized_df

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for parquet storage."""
        optimized_df = df.copy()

        for col, config in self.column_optimizations.items():
            if col in optimized_df.columns:
                target_dtype = config['dtype']

                try:
                    if target_dtype == 'category':
                        # Use category for high-cardinality string columns
                        optimized_df[col] = optimized_df[col].astype('category')
                    elif target_dtype == 'datetime64[ns]':
                        # Ensure proper datetime format
                        if not pd.api.types.is_datetime64_any_dtype(optimized_df[col]):
                            optimized_df[col] = pd.to_datetime(optimized_df[col], utc=True)
                    else:
                        # Convert to target numeric type
                        optimized_df[col] = optimized_df[col].astype(target_dtype)

                except Exception as e:
                    self.logger.warning(f"Could not optimize {col} to {target_dtype}: {e}")

        return optimized_df

    def _sort_for_compression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort data for better compression efficiency."""
        if 'timestamp' in df.columns:
            return df.sort_values('timestamp').reset_index(drop=True)
        return df

    def _remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that don't add value for storage."""
        # Keep only essential columns
        essential_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'symbol', 'exchange', 'interval'
        ]

        # Add any additional columns that exist
        existing_columns = [col for col in essential_columns if col in df.columns]
        additional_columns = [col for col in df.columns if col not in essential_columns]

        return df[existing_columns + additional_columns]

    def _optimize_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize categorical columns for better compression."""
        optimized_df = df.copy()

        categorical_columns = ['symbol', 'exchange', 'interval']
        for col in categorical_columns:
            if col in optimized_df.columns:
                # Use category type for better compression
                optimized_df[col] = optimized_df[col].astype('category')

        return optimized_df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values efficiently."""
        optimized_df = df.copy()

        # For OHLCV data, forward fill missing values
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_columns:
            if col in optimized_df.columns:
                if optimized_df[col].isnull().any():
                    optimized_df[col] = optimized_df[col].fillna(method='ffill')

        return optimized_df

    def _ensure_required_columns(self, df: pd.DataFrame, symbol: str, exchange: str, interval: str) -> pd.DataFrame:
        """Ensure required columns exist."""
        optimized_df = df.copy()

        if 'exchange' not in optimized_df.columns:
            optimized_df['exchange'] = exchange
        if 'symbol' not in optimized_df.columns:
            optimized_df['symbol'] = symbol
        if 'interval' not in optimized_df.columns:
            optimized_df['interval'] = interval

        return optimized_df

    def _get_optimal_parquet_kwargs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get optimal parquet write parameters."""
        kwargs = {
            'engine': 'pyarrow',
            'index': self.config.index,
            'compression': self.config.compression,
        }

        # Add compression level for zstd
        if self.config.compression == 'zstd':
            kwargs['compression_level'] = self.config.compression_level

        # Row group size optimization
        if len(df) > 0:
            optimal_row_group_size = min(
                self.config.row_group_size,
                max(1000, len(df) // 10)  # At least 10 row groups
            )
            kwargs['row_group_size'] = optimal_row_group_size

        # Dictionary encoding for categorical columns
        if self.config.use_dictionary_encoding:
            categorical_columns = df.select_dtypes(include=['category']).columns.tolist()
            if categorical_columns:
                kwargs['use_dictionary'] = True

        return kwargs

    def _store_dataframe_optimized(self, df: pd.DataFrame, path: Path, kwargs: Dict[str, Any]) -> bool:
        """Store DataFrame with optimizations."""
        try:
            df.to_parquet(path, **kwargs)
            return True
        except Exception as e:
            tprint_error(f"❌ Failed to store optimized DataFrame: {e}")
            return False

    def _calculate_compression_stats(
        self,
        original_df: pd.DataFrame,
        optimized_df: pd.DataFrame,
        file_path: Path
    ) -> Dict[str, Any]:
        """Calculate compression statistics."""
        if not file_path.exists():
            return {}

        original_size = original_df.memory_usage(deep=True).sum()
        file_size = file_path.stat().st_size
        compression_ratio = (1 - file_size / original_size) * 100 if original_size > 0 else 0

        return {
            'original_size_bytes': original_size,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'optimization_applied': True
        }

    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Legacy method - now handled by _optimize_dtypes."""
        return self._optimize_dtypes(df)

    def _get_storage_path(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        batch_id: str
    ) -> Path:
        """Get the storage path for klines data."""
        return (
            self.base_dir /
            exchange.lower() /
            symbol.lower() /
            "klines" /
            f"klines_{exchange}_{symbol}_{interval}_{batch_id}.parquet"
        )

    def _create_enhanced_metadata(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str,
        batch_id: str,
        file_path: Path,
        additional_metadata: Optional[Dict[str, Any]] = None,
        compression_stats: Optional[Dict[str, Any]] = None
    ) -> KlinesMetadata:
        """Create enhanced metadata with optimization details."""
        file_size = file_path.stat().st_size if file_path.exists() else 0

        # Calculate compression ratio
        compression_ratio = 0.0
        if compression_stats and 'compression_ratio' in compression_stats:
            compression_ratio = compression_stats['compression_ratio']
        elif file_size > 0 and len(df) > 0:
            # Fallback calculation
            estimated_size = len(df) * len(df.columns) * 8  # Rough estimate
            compression_ratio = (1 - file_size / estimated_size) * 100 if estimated_size > 0 else 0

        return KlinesMetadata(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            batch_id=batch_id,
            start_time=df['timestamp'].min() if 'timestamp' in df.columns else df.index.min(),
            end_time=df['timestamp'].max() if 'timestamp' in df.columns else df.index.max(),
            record_count=len(df),
            file_size_bytes=file_size,
            compression_ratio=compression_ratio,
            created_at=datetime.now(),
            additional_metadata={
                **(additional_metadata or {}),
                'optimization_applied': True,
                'compression_used': self.config.compression,
                'row_group_size': self.config.row_group_size,
                'dictionary_encoding': self.config.use_dictionary_encoding,
                'compression_stats': compression_stats or {}
            }
        )

    def _create_metadata(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str,
        batch_id: str,
        file_path: Path,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> KlinesMetadata:
        """Create metadata for klines data."""
        file_size = file_path.stat().st_size if file_path.exists() else 0

        return KlinesMetadata(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            batch_id=batch_id,
            start_time=df['timestamp'].min() if 'timestamp' in df.columns else df.index.min(),
            end_time=df['timestamp'].max() if 'timestamp' in df.columns else df.index.max(),
            record_count=len(df),
            file_size_bytes=file_size,
            compression_ratio=file_size / (len(df) * len(df.columns) * 8) if len(df) > 0 else 0,
            created_at=datetime.now(),
            additional_metadata=additional_metadata or {}
        )

    def _store_metadata(self, metadata: KlinesMetadata, file_path: Path) -> None:
        """Store metadata to JSON file."""
        if not self.config.enable_metadata:
            return

        metadata_path = file_path.with_suffix('.metadata.json')

        metadata_dict = {
            "symbol": metadata.symbol,
            "exchange": metadata.exchange,
            "interval": metadata.interval,
            "batch_id": metadata.batch_id,
            "start_time": metadata.start_time.isoformat(),
            "end_time": metadata.end_time.isoformat(),
            "record_count": metadata.record_count,
            "file_size_bytes": metadata.file_size_bytes,
            "compression_ratio": metadata.compression_ratio,
            "created_at": metadata.created_at.isoformat(),
            "data_quality_score": metadata.data_quality_score,
            "gaps_detected": metadata.gaps_detected,
            "gaps_filled": metadata.gaps_filled,
            "resampled_intervals": metadata.resampled_intervals,
            "additional_metadata": metadata.additional_metadata
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

    def _load_file_metadata(self, file_path: Path) -> Optional[KlinesMetadata]:
        """Load metadata from JSON file."""
        metadata_path = file_path.with_suffix('.metadata.json')

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)

            return KlinesMetadata(
                symbol=data['symbol'],
                exchange=data['exchange'],
                interval=data['interval'],
                batch_id=data['batch_id'],
                start_time=datetime.fromisoformat(data['start_time']),
                end_time=datetime.fromisoformat(data['end_time']),
                record_count=data['record_count'],
                file_size_bytes=data['file_size_bytes'],
                compression_ratio=data['compression_ratio'],
                created_at=datetime.fromisoformat(data['created_at']),
                data_quality_score=data.get('data_quality_score', 0.0),
                gaps_detected=data.get('gaps_detected', 0),
                gaps_filled=data.get('gaps_filled', 0),
                resampled_intervals=data.get('resampled_intervals', []),
                additional_metadata=data.get('additional_metadata', {})
            )
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load metadata from {metadata_path}: {e}")
            return None

    def _find_klines_files(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        batch_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Path]:
        """Find klines files matching criteria, optionally filtered by date range."""
        # Check processed directory first (preferred), then fall back to klines directory
        processed_dir = self.base_dir / exchange.lower() / symbol.lower() / "processed" / f"{symbol.lower()}_{interval}"
        klines_dir = self.base_dir / exchange.lower() / symbol.lower() / "klines"

        all_files = []
        
        # Try processed directory first (contains partitioned parquet files)
        if processed_dir.exists():
            # Processed data is partitioned by year, so we need to search recursively
            all_files = sorted(processed_dir.glob("**/*.parquet"))
            if all_files:
                tprint_info(f"📁 Loading from processed directory: {len(all_files)} files found")
        
        # Fall back to klines directory if no processed data found
        if not all_files and klines_dir.exists():
            pattern = f"klines_{exchange}_{symbol}_{interval}_*.parquet"
            if batch_id:
                pattern = f"klines_{exchange}_{symbol}_{interval}_{batch_id}.parquet"
            all_files = sorted(klines_dir.glob(pattern))
            if all_files:
                tprint_info(f"📁 Loading from klines directory: {len(all_files)} files found")
        
        if not all_files:
            return []
        
        # If no time filters, return all files
        if start_time is None and end_time is None:
            return all_files
        
        # For processed files (partitioned by year), skip metadata filtering and rely on in-memory filtering
        # since processed files don't have metadata files
        if processed_dir.exists() and all_files:
            tprint_info(f"📁 Using all {len(all_files)} processed files (will filter in-memory)")
            return all_files
        
        # Filter files by metadata date ranges (only for klines directory files)
        filtered_files = []
        for file_path in all_files:
            metadata = self._load_file_metadata(file_path)
            if metadata:
                # Check if file's time range overlaps with requested range
                file_start = metadata.start_time
                file_end = metadata.end_time
                
                # Skip files that are completely outside the requested range
                if start_time and file_end < start_time:
                    continue
                if end_time and file_start > end_time:
                    continue
                    
                filtered_files.append(file_path)
            else:
                # If no metadata, include the file (fallback to in-memory filtering)
                filtered_files.append(file_path)
        
        if start_time or end_time:
            tprint_info(f"📁 File pre-filter: {len(all_files)} total → {len(filtered_files)} relevant files")
        
        return filtered_files

    def _load_and_combine_files(
        self,
        files: List[Path],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load and combine multiple parquet files."""
        tprint(f"🐛 DEBUG [KlinesParquetManager]: Loading {len(files)} parquet files", "INFO")
        dataframes = []

        for i, file_path in enumerate(files):
            try:
                df = self.parquet_utils.safe_read_parquet(str(file_path))
                if df is not None and not df.empty:
                    # Infer span for diagnostics (best-effort)
                    span_start = span_end = None
                    ts_candidate = None
                    if isinstance(df.index, pd.DatetimeIndex):
                        ts_candidate = df.index
                    elif 'timestamp' in df.columns:
                        ts_candidate = pd.to_datetime(df['timestamp'], errors='coerce')
                    elif 'open_time' in df.columns:
                        # allow automatic unit inference if ms parse fails
                        try:
                            ts_candidate = pd.to_datetime(df['open_time'], errors='coerce', unit='ms')
                        except Exception:
                            ts_candidate = pd.to_datetime(df['open_time'], errors='coerce')
                    if ts_candidate is not None and ts_candidate.notna().any():
                        span_start = ts_candidate.min()
                        span_end = ts_candidate.max()

                    dataframes.append(df)
                    if i < 3 or i >= len(files) - 3:  # Log first and last 3 files
                        tprint(
                            f"🐛 DEBUG [KlinesParquetManager]: File {i+1}/{len(files)}: "
                            f"{file_path} - {len(df)} rows, "
                            f"span=({span_start}, {span_end})",
                            "INFO",
                        )
                elif i < 3:
                    tprint(f"🐛 DEBUG [KlinesParquetManager]: File {i+1}/{len(files)}: {file_path.name} - EMPTY/NONE", "WARNING")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load {file_path}: {e}")
                continue

        if not dataframes:
            tprint(f"🐛 DEBUG [KlinesParquetManager]: No dataframes loaded! All files failed or empty", "ERROR")
            return pd.DataFrame()

        tprint(f"🐛 DEBUG [KlinesParquetManager]: Successfully loaded {len(dataframes)} dataframes, combining...", "INFO")

        # Combine dataframes
        combined_df = pd.concat(dataframes, ignore_index=False)  # Keep index if timestamp is in index
        tprint(f"🐛 DEBUG [KlinesParquetManager]: Combined shape before filtering: {combined_df.shape}", "INFO")

        # Check if timestamp is in the index or as a column
        has_timestamp_index = combined_df.index.name == 'timestamp' or isinstance(combined_df.index, pd.DatetimeIndex)
        has_timestamp_column = 'timestamp' in combined_df.columns
        
        if has_timestamp_index:
            # CRITICAL FIX: Normalize timezone before sorting to prevent comparison errors
            # Convert all timezone-aware timestamps to timezone-naive (UTC)
            if isinstance(combined_df.index, pd.DatetimeIndex) and combined_df.index.tz is not None:
                tprint(f"🐛 DEBUG [KlinesParquetManager]: Converting timezone-aware index to timezone-naive", "INFO")
                combined_df.index = combined_df.index.tz_localize(None)
            
            # Also check for mixed timezone awareness in the index
            # This can happen when concatenating dataframes with different timezone settings
            tprint(f"🔧 TIMEZONE: Index timezone: {getattr(combined_df.index, 'tz', 'NAIVE')}", "INFO")
            
            try:
                # Try to sort - if it fails, we need to normalize
                combined_df = combined_df.sort_index()
            except TypeError as e:
                if "tz-naive and tz-aware" in str(e):
                    tprint(f"🔧 TIMEZONE: Detected mixed timezone awareness, normalizing...", "WARNING")
                    tprint(f"🔧 TIMEZONE: Before timezone fix: {len(combined_df)} rows", "INFO")
                    # Convert index to timezone-naive by forcing conversion
                    combined_df.index = pd.to_datetime(combined_df.index, utc=True).tz_localize(None)
                    tprint(f"🔧 TIMEZONE: After timezone fix: {len(combined_df)} rows", "INFO")
                    combined_df = combined_df.sort_index()
                    tprint(f"🔧 TIMEZONE: After sort: {len(combined_df)} rows", "INFO")
                else:
                    raise
            
            tprint(f"🐛 DEBUG [KlinesParquetManager]: Before duplicate removal: {len(combined_df)} rows", "INFO")
            tprint(f"🐛 DEBUG [KlinesParquetManager]: Duplicated timestamps: {combined_df.index.duplicated().sum()}", "INFO")
            tprint(f"🐛 DEBUG [KlinesParquetManager]: Unique timestamps: {combined_df.index.nunique()}", "INFO")
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            tprint(f"🐛 DEBUG [KlinesParquetManager]: After duplicate removal: {len(combined_df)} rows", "INFO")
            tprint(f"🐛 DEBUG [KlinesParquetManager]: start_time={start_time}, end_time={end_time}", "INFO")
            
            # Apply time filters - handle timezone awareness
            if start_time:
                # Convert to pandas Timestamp and localize if needed
                start_ts = pd.Timestamp(start_time)
                if hasattr(combined_df.index, 'tz') and combined_df.index.tz is not None:
                    if start_ts.tz is None:
                        # Add timezone normalization before comparison
                        tprint(f"🔧 TIMEZONE: Index timezone: {getattr(combined_df.index, 'tz', 'NAIVE')}", "INFO")
                        tprint(f"🔧 TIMEZONE: Start timestamp timezone: {getattr(start_ts, 'tz', 'NAIVE')}", "INFO")
                        
                        if hasattr(combined_df.index, 'tz') and combined_df.index.tz is not None:
                            # Convert all timestamps to UTC naive for consistent comparison
                            combined_df.index = combined_df.index.tz_convert('UTC').tz_localize(None)
                            tprint("🔧 TIMEZONE: Converted timezone-aware index to UTC naive", "INFO")
                        elif hasattr(start_ts, 'tz') and start_ts.tz is not None:
                            # Convert start_ts to UTC naive for consistent comparison
                            start_ts = start_ts.tz_convert('UTC').tz_localize(None)
                            tprint("🔧 TIMEZONE: Converted start_ts to UTC naive", "INFO")
                        else:
                            start_ts = start_ts.tz_localize('UTC').tz_convert(combined_df.index.tz)
                    else:
                        start_ts = start_ts.tz_localize('UTC').tz_convert(combined_df.index.tz)
                combined_df = combined_df[combined_df.index >= start_ts]
            if end_time:
                # Convert to pandas Timestamp and localize if needed
                end_ts = pd.Timestamp(end_time)
                
                # Add timezone normalization before comparison
                tprint(f"🔧 TIMEZONE: Index timezone: {getattr(combined_df.index, 'tz', 'NAIVE')}", "INFO")
                tprint(f"🔧 TIMEZONE: End timestamp timezone: {getattr(end_ts, 'tz', 'NAIVE')}", "INFO")
                
                if hasattr(combined_df.index, 'tz') and combined_df.index.tz is not None:
                    # Convert all timestamps to UTC naive for consistent comparison
                    combined_df.index = combined_df.index.tz_convert('UTC').tz_localize(None)
                    tprint("🔧 TIMEZONE: Converted timezone-aware index to UTC naive", "INFO")
                elif hasattr(end_ts, 'tz') and end_ts.tz is not None:
                    # Convert end_ts to UTC naive for consistent comparison
                    end_ts = end_ts.tz_convert('UTC').tz_localize(None)
                    tprint("🔧 TIMEZONE: Converted end_ts to UTC naive", "INFO")
                else:
                    if end_ts.tz is None:
                        end_ts = end_ts.tz_localize('UTC').tz_convert(combined_df.index.tz)
                combined_df = combined_df[combined_df.index <= end_ts]
                
        elif has_timestamp_column:
            # Timestamp is a column
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            
            # Sort by timestamp
            combined_df = combined_df.sort_values('timestamp')
            
            # Apply time filters - ensure timestamp is datetime for comparison
            if start_time or end_time:
                # Convert timestamp to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(combined_df['timestamp']):
                    # Check if timestamp is in milliseconds or seconds
                    if combined_df['timestamp'].max() > 1e12:  # Likely milliseconds
                        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], unit='ms', utc=True)
                    else:  # Likely seconds
                        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], unit='s', utc=True)
            
            if start_time:
                # Ensure start_time is timezone-aware if timestamp is
                if combined_df['timestamp'].dt.tz is not None and start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=pd.Timestamp.utcnow().tz)
                before_filter = len(combined_df)
                combined_df = combined_df[combined_df['timestamp'] >= start_time]
                tprint(f"🐛 DEBUG [KlinesParquetManager]: After start_time filter: {before_filter} → {len(combined_df)} rows", "INFO")
            if end_time:
                # Ensure end_time is timezone-aware if timestamp is
                if combined_df['timestamp'].dt.tz is not None and end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=pd.Timestamp.utcnow().tz)
                before_filter = len(combined_df)
                combined_df = combined_df[combined_df['timestamp'] <= end_time]
                tprint(f"🐛 DEBUG [KlinesParquetManager]: After end_time filter: {before_filter} → {len(combined_df)} rows", "INFO")

        tprint(f"🐛 DEBUG [KlinesParquetManager]: Final combined_df shape: {combined_df.shape}", "INFO")
        if not combined_df.empty:
            if has_timestamp_index:
                tprint(f"🐛 DEBUG [KlinesParquetManager]: Date range: {combined_df.index.min()} to {combined_df.index.max()}", "INFO")
            elif has_timestamp_column:
                tprint(f"🐛 DEBUG [KlinesParquetManager]: Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}", "INFO")

        return combined_df

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            available_data = self.list_available_data()

            if not available_data:
                return {"total_files": 0, "total_size_mb": 0, "total_records": 0}

            total_files = len(available_data)
            total_size_mb = sum(item["file_size_mb"] for item in available_data)
            total_records = sum(item["record_count"] for item in available_data)

            # Group by exchange and symbol
            by_exchange = {}
            by_symbol = {}

            for item in available_data:
                exchange = item["exchange"]
                symbol = item["symbol"]

                if exchange not in by_exchange:
                    by_exchange[exchange] = {"files": 0, "size_mb": 0, "records": 0}
                by_exchange[exchange]["files"] += 1
                by_exchange[exchange]["size_mb"] += item["file_size_mb"]
                by_exchange[exchange]["records"] += item["record_count"]

                if symbol not in by_symbol:
                    by_symbol[symbol] = {"files": 0, "size_mb": 0, "records": 0}
                by_symbol[symbol]["files"] += 1
                by_symbol[symbol]["size_mb"] += item["file_size_mb"]
                by_symbol[symbol]["records"] += item["record_count"]

            return {
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "total_records": total_records,
                "by_exchange": by_exchange,
                "by_symbol": by_symbol
            }

        except Exception as e:
            tprint_error(f"❌ Failed to get storage stats: {e}")
            return {"error": str(e)}

    def get_optimization_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get optimization recommendations based on data characteristics."""
        recommendations = {
            'compression': self.config.compression,
            'estimated_size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        # Compression recommendation based on data size
        if len(df) > 1000000:  # > 1M rows
            recommendations['compression'] = 'zstd'
            recommendations['reason'] = 'Large dataset - use zstd for better compression'
        elif len(df) > 100000:  # > 100K rows
            recommendations['compression'] = 'lz4'
            recommendations['reason'] = 'Medium dataset - use lz4 for good balance'
        else:
            recommendations['compression'] = 'snappy'
            recommendations['reason'] = 'Small dataset - use snappy for speed'

        # Row group size recommendation
        if len(df) > 0:
            optimal_row_groups = max(1, len(df) // 50000)  # ~50k rows per group
            recommendations['row_group_size'] = min(100000, len(df) // optimal_row_groups)

        return recommendations

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get overall compression statistics."""
        if not self._compression_stats:
            return {"message": "No compression statistics available"}

        total_original_size = sum(stats.get('original_size_bytes', 0) for stats in self._compression_stats.values())
        total_file_size = sum(stats.get('file_size_bytes', 0) for stats in self._compression_stats.values())
        overall_compression_ratio = (1 - total_file_size / total_original_size) * 100 if total_original_size > 0 else 0

        return {
            "total_files": len(self._compression_stats),
            "total_original_size_mb": total_original_size / (1024 * 1024),
            "total_file_size_mb": total_file_size / (1024 * 1024),
            "overall_compression_ratio": overall_compression_ratio,
            "average_compression_ratio": np.mean([stats.get('compression_ratio', 0) for stats in self._compression_stats.values()]) if self._compression_stats else 0,
            "compression_stats": self._compression_stats
        }

# Convenience functions
def create_klines_manager(config: Optional[StorageConfig] = None) -> KlinesParquetManager:
    """Create a new KlinesParquetManager instance."""
    return KlinesParquetManager(config)

def get_klines_manager() -> KlinesParquetManager:
    """Get a singleton KlinesParquetManager instance."""
    if not hasattr(get_klines_manager, '_instance'):
        get_klines_manager._instance = KlinesParquetManager()
    return get_klines_manager._instance

def process_klines_data(df: pd.DataFrame,
                       symbol: str,
                       exchange: str,
                       interval: str,
                       config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Process klines data using the KlinesParquetManager.
    
    Args:
        df: Input DataFrame containing klines data
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "binance")
        interval: Data interval (e.g., "1m")
        config: Optional configuration dictionary
        
    Returns:
        Processed DataFrame with optimizations applied
    """
    # Create manager instance
    manager = KlinesParquetManager(config)
    
    # Validate data first
    if not validate_klines_data(df):
        tprint_error("❌ Klines data validation failed")
        return pd.DataFrame()
    
    # Apply comprehensive optimizations
    processed_df = manager._apply_comprehensive_optimizations(df, symbol, exchange, interval)
    
    tprint_success(f"✅ Processed {len(processed_df)} klines records for {symbol} {interval}")
    return processed_df

def validate_klines_data(df: pd.DataFrame, 
                        required_columns: Optional[List[str]] = None,
                        min_rows: int = 1,
                        allow_duplicates: bool = True) -> bool:
    """Validate klines data structure and content.
    
    Args:
        df: DataFrame containing klines data
        required_columns: List of required columns (default: OHLCV + timestamp)
        min_rows: Minimum number of rows required
        allow_duplicates: Whether to allow duplicate timestamps
        
    Returns:
        True if data is valid, False otherwise
    """
    if df is None:
        tprint_error("❌ DataFrame is None")
        return False
    
    if df.empty:
        tprint_error("❌ DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        tprint_error(f"❌ DataFrame has {len(df)} rows, minimum required: {min_rows}")
        return False
    
    # Default required columns for klines data
    if required_columns is None:
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Check for required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        tprint_error(f"❌ Missing required columns: {missing_cols}")
        return False
    
    # Check for duplicate timestamps if not allowed
    if not allow_duplicates and 'timestamp' in df.columns:
        if df['timestamp'].duplicated().any():
            tprint_error("❌ DataFrame contains duplicate timestamps")
            return False
    
    # Validate OHLCV data integrity
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlcv_cols:
        if col in df.columns:
            if df[col].isnull().any():
                tprint_warning(f"⚠️ Column {col} contains null values")
            if (df[col] < 0).any():
                tprint_warning(f"⚠️ Column {col} contains negative values")
    
    # Validate OHLC relationships
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_ohlc.any():
            tprint_warning(f"⚠️ Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
    
    return True

if __name__ == "__main__":
    # Example usage
    import numpy as np
    from datetime import datetime, timedelta

    # Create test data
    dates = pd.date_range(start=datetime.now() - timedelta(days=1), periods=1440, freq='1min')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(3000, 3100, 1440),
        'high': np.random.uniform(3100, 3200, 1440),
        'low': np.random.uniform(2900, 3000, 1440),
        'close': np.random.uniform(3000, 3100, 1440),
        'volume': np.random.uniform(100, 1000, 1440)
    })

    # Test the manager
    manager = KlinesParquetManager()

    # Store data
    success = manager.store_klines(test_data, "ETHUSDT", "binance", "1m")
    print(f"Storage successful: {success}")

    # Load data
    loaded_data = manager.load_klines("ETHUSDT", "binance", "1m")
    print(f"Loaded {len(loaded_data)} records")

    # Get stats
    stats = manager.get_storage_stats()
    print(f"Storage stats: {stats}")
