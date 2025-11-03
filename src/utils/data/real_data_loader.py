"""
Real Data Loading Utilities

This module provides utilities to load real market data instead of using synthetic/mock data.
It integrates with the existing data collection infrastructure to ensure we always use real data.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib

from src.utils.logger import system_logger
from src.utils.data.klines_parquet import KlinesParquetManager

logger = logging.getLogger(__name__)

class RealDataLoader:
    """
    Real data loader that ensures we never use synthetic/mock data.
    Uses KlinesParquetManager for proper data access.
    Implements in-memory LRU caching for performance optimization.
    """
    
    # Class-level cache for market data (shared across instances)
    _market_data_cache: Dict[str, pd.DataFrame] = {}
    _cache_max_size = 50  # Maximum number of cached datasets

    def __init__(self, data_dir: str = 'historical_data'):
        """Initialize the real data loader.

        Args:
            data_dir: Base directory for data storage (default: 'historical_data')
        """
        self.logger = system_logger.getChild('RealDataLoader')
        self.data_dir = Path(data_dir)
        self.klines_manager = None  # Will be initialized per exchange
    
    @staticmethod
    def _create_cache_key(symbol: str, exchange: str, timeframe: str, 
                         start_date: Optional[str], end_date: Optional[str]) -> str:
        """Create a unique cache key for market data.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            Hash-based cache key
        """
        # Create a deterministic string representation
        key_str = f"{symbol}_{exchange}_{timeframe}_{start_date}_{end_date}"
        # Use hash for shorter keys
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if available.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached DataFrame or None
        """
        if cache_key in self._market_data_cache:
            self.logger.info(f"✅ Cache hit for key: {cache_key[:8]}...")
            return self._market_data_cache[cache_key].copy()  # Return copy to avoid mutations
        return None
    
    def _add_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Add data to cache with LRU eviction.
        
        Args:
            cache_key: Cache key
            data: DataFrame to cache
        """
        # Implement simple LRU: remove oldest entry if cache is full
        if len(self._market_data_cache) >= self._cache_max_size:
            # Remove first (oldest) entry
            oldest_key = next(iter(self._market_data_cache))
            del self._market_data_cache[oldest_key]
            self.logger.debug(f"🗑️  Evicted cache entry: {oldest_key[:8]}...")
        
        self._market_data_cache[cache_key] = data.copy()
        self.logger.info(f"💾 Added to cache: {cache_key[:8]}... (cache size: {len(self._market_data_cache)}/{self._cache_max_size})")

    async def load_market_data(
        self,
        symbol: str = 'ETHUSDT',
        exchange: str = 'binance',
        timeframe: str = '1m',
        lookback_days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_download: bool = False,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load real market data using KlinesParquetManager with LRU caching.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe (1m, 5m, 15m, 1h, etc.)
            lookback_days: Number of days to look back from now (takes precedence over start_date)
            start_date: Start date (YYYY-MM-DD) 
            end_date: End date (YYYY-MM-DD)
            force_download: Force download even if data exists
            use_cache: Use in-memory cache (default: True)

        Returns:
            DataFrame with real market data
        """
        try:
            # Calculate dates for cache key
            cache_start_date = start_date
            cache_end_date = end_date
            
            if lookback_days:
                end_dt = datetime.now() if not end_date else pd.to_datetime(end_date)
                start_dt = end_dt - timedelta(days=lookback_days)
                cache_start_date = start_dt.strftime('%Y-%m-%d')
                if not end_date:
                    cache_end_date = end_dt.strftime('%Y-%m-%d')
            
            # Check cache first (unless force_download or cache disabled)
            if use_cache and not force_download:
                cache_key = self._create_cache_key(symbol, exchange, timeframe, 
                                                   cache_start_date, cache_end_date)
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    return cached_data
            # Initialize klines manager for this exchange
            klines_manager = KlinesParquetManager(
                data_dir=str(self.data_dir),
                exchange=exchange
            )
            
            self.logger.info(f"🔍 Loading data: {symbol}/{exchange}/{timeframe}")
            
            # Convert lookback_days to start_date if provided
            if lookback_days:
                end_dt = datetime.now() if not end_date else pd.to_datetime(end_date)
                start_dt = end_dt - timedelta(days=lookback_days)
                start_date = start_dt.strftime('%Y-%m-%d')
                if not end_date:
                    end_date = end_dt.strftime('%Y-%m-%d')
                self.logger.info(f"📅 Lookback {lookback_days} days: {start_date} to {end_date}")
            
            # Parse dates
            start_datetime = pd.to_datetime(start_date) if start_date else None
            end_datetime = pd.to_datetime(end_date) if end_date else None
            
            # Use KlinesParquetManager to read processed data
            data = klines_manager.read_data(
                symbol=symbol,
                interval=timeframe,
                data_type="processed",  # Use processed data
                start_date=start_datetime,
                end_date=end_datetime
            )
            
            if data is not None and len(data) > 0:
                initial_rows = len(data)
                self.logger.info(f"✅ Loaded {initial_rows} rows from KlinesParquetManager")
                
                # Filter out epoch timestamps (1970-01-01)
                data = self._filter_epoch_timestamps(data)
                
                # Check for missing values and log details
                self._log_missing_values(data, symbol, timeframe)
                
                if len(data) == 0:
                    raise RuntimeError(
                        f"❌ No valid data remaining after filtering epoch timestamps for {symbol}/{exchange}/{timeframe}"
                    )
                
                if len(data) < initial_rows:
                    self.logger.info(f"🔧 Filtered {initial_rows - len(data)} epoch timestamp records, {len(data)} rows remaining")
                
                # Add to cache before returning (if cache is enabled)
                if use_cache:
                    cache_key = self._create_cache_key(symbol, exchange, timeframe, 
                                                       cache_start_date, cache_end_date)
                    self._add_to_cache(cache_key, data)
                
                return data
            
            # If no data, raise error
            raise RuntimeError(
                f"❌ No data found for {symbol}/{exchange}/{timeframe}. "
                f"Please ensure data has been downloaded and processed."
            )

        except Exception as e:
            self.logger.error(f"❌ Error loading market data: {e}")
            raise

    def _filter_epoch_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid epoch timestamps (1970-01-01).
        
        Args:
            data: DataFrame to filter
            
        Returns:
            Filtered DataFrame without epoch timestamps
        """
        try:
            # Check if data has timestamp column or DatetimeIndex
            if 'timestamp' in data.columns:
                # Convert timestamp to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s', errors='coerce')
                
                # Filter out 1970-01-01 (epoch)
                epoch_date = pd.Timestamp('1970-01-01')
                epoch_mask = data['timestamp'].dt.date == epoch_date.date()
                epoch_count = epoch_mask.sum()
                
                if epoch_count > 0:
                    self.logger.warning(f"⚠️ Filtering {epoch_count} records with epoch timestamp (1970-01-01)")
                    data = data[~epoch_mask].copy()
                    
            elif isinstance(data.index, pd.DatetimeIndex):
                # Filter by index
                epoch_date = pd.Timestamp('1970-01-01')
                epoch_mask = data.index.date == epoch_date.date()
                epoch_count = epoch_mask.sum()
                
                if epoch_count > 0:
                    self.logger.warning(f"⚠️ Filtering {epoch_count} records with epoch timestamp (1970-01-01)")
                    data = data[~epoch_mask].copy()
            
            return data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not filter epoch timestamps: {e}")
            return data
    
    def _log_missing_values(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Log detailed information about missing values.
        
        Args:
            data: DataFrame to check
            symbol: Trading symbol
            timeframe: Timeframe
        """
        try:
            # Count missing values per column
            missing_counts = data.isnull().sum()
            total_missing = missing_counts.sum()
            
            if total_missing > 0:
                self.logger.info(f"📊 Missing Values Report for {symbol}/{timeframe}:")
                self.logger.info(f"   Total missing values: {total_missing:,}")
                self.logger.info(f"   Total cells: {data.size:,}")
                self.logger.info(f"   Missing percentage: {(total_missing / data.size * 100):.2f}%")
                
                # Show columns with missing values
                missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=False)
                if len(missing_cols) > 0:
                    self.logger.info(f"   Columns with missing values:")
                    for col, count in missing_cols.items():
                        pct = (count / len(data)) * 100
                        self.logger.info(f"      - {col}: {count:,} ({pct:.2f}%)")
                
                # Check for rows with all missing values
                all_missing_rows = data.isnull().all(axis=1).sum()
                if all_missing_rows > 0:
                    self.logger.warning(f"   ⚠️ {all_missing_rows} rows have ALL values missing")
                
                # Check for critical columns (OHLCV)
                critical_cols = ['open', 'high', 'low', 'close', 'volume']
                critical_missing = {col: missing_counts.get(col, 0) for col in critical_cols if col in data.columns}
                critical_total = sum(critical_missing.values())
                
                if critical_total > 0:
                    self.logger.warning(f"   ⚠️ Critical OHLCV columns have {critical_total:,} missing values:")
                    for col, count in critical_missing.items():
                        if count > 0:
                            pct = (count / len(data)) * 100
                            self.logger.warning(f"      - {col}: {count:,} ({pct:.2f}%)")
                    
                    # Show WHEN these missing values occur
                    self.logger.warning(f"   ⚠️ Timestamps with missing OHLCV values:")
                    ohlcv_missing_mask = data[critical_cols].isnull().any(axis=1)
                    missing_rows = data[ohlcv_missing_mask]
                    
                    if len(missing_rows) > 0:
                        # Try to get timestamp from various columns
                        if 'timestamp' in missing_rows.columns:
                            timestamps = pd.to_datetime(missing_rows['timestamp'], unit='s', errors='coerce')
                        elif isinstance(missing_rows.index, pd.DatetimeIndex):
                            timestamps = missing_rows.index
                        elif 'open_time' in missing_rows.columns:
                            timestamps = pd.to_datetime(missing_rows['open_time'], unit='ms', errors='coerce')
                        else:
                            timestamps = missing_rows.index
                        
                        for idx, (ts, row) in enumerate(zip(timestamps, missing_rows.itertuples()), 1):
                            missing_cols = [col for col in critical_cols if col in data.columns and pd.isna(getattr(row, col, None))]
                            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                            self.logger.warning(f"      {idx}. {ts_str} - Missing: {', '.join(missing_cols)}")
                            
                            # Limit output to first 10 rows
                            if idx >= 10:
                                remaining = len(missing_rows) - 10
                                if remaining > 0:
                                    self.logger.warning(f"      ... and {remaining} more rows")
                                break
                else:
                    self.logger.info(f"   ✅ All critical OHLCV columns are complete (no missing values)")
            else:
                self.logger.info(f"✅ No missing values in {symbol}/{timeframe} data")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not analyze missing values: {e}")

# Global instance for convenience
real_data_loader = RealDataLoader()
