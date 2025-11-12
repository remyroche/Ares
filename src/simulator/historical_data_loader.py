"""
Historical Data Loader

Utility for loading cached historical klines data from the historical_data/ directory
for use in paper trading simulation and mocking.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.utils.tprint import (
    tprint, tprint_logged, tprint_timer, tprint_performance,
    tprint_data_preview, tprint_data_format, tprint_feature_counts, LogLevel
)


@dataclass
class KlineRecord:
    """Individual kline record"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    interval: str


class HistoricalDataLoader:
    """
    Load historical klines data from cached files.
    
    Supports loading from:
    - Partitioned parquet files (historical_data/{exchange}/{symbol}/processed/{symbol}_{interval}/)
    - Consolidated parquet files
    - Falls back to synthetic data if unavailable
    """
    
    def __init__(self, base_dir: str = "historical_data"):
        """
        Initialize historical data loader.

        Args:
            base_dir: Base directory for historical data (default: "historical_data")
        """
        tprint(f"[HIST_LOADER] __init__: Initializing historical data loader with base_dir={base_dir}")
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)

        # In-memory cache: {(exchange, symbol, interval): DataFrame}
        self._cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}

        # Last load time for cache invalidation
        self._cache_timestamps: Dict[Tuple[str, str, str], datetime] = {}

        # Cache TTL in seconds (5 minutes for real-time, longer for historical)
        self.cache_ttl_seconds = 300
        tprint(f"[HIST_LOADER] __init__ -> initialized (cache_ttl={self.cache_ttl_seconds}s)")
    
    def _get_data_path(self, exchange: str, symbol: str, interval: str) -> Optional[Path]:
        """
        Get path to historical data for a given exchange/symbol/interval.
        
        Args:
            exchange: Exchange name (e.g., "binance")
            symbol: Trading symbol (e.g., "ETHUSDT")
            interval: Timeframe (e.g., "1m", "15m", "1h")
            
        Returns:
            Path to data directory or None if not found
        """
        # Normalize inputs
        exchange = exchange.lower()
        symbol = symbol.lower()
        
        # Try processed data directory first
        processed_path = self.base_dir / exchange / symbol / "processed" / f"{symbol}_{interval}"
        if processed_path.exists():
            return processed_path
        
        # Try consolidated file
        consolidated_path = self.base_dir / exchange / symbol / "processed" / f"{symbol}_{interval}_consolidated.parquet"
        if consolidated_path.exists():
            return consolidated_path
        
        # Try storage directory
        storage_path = self.base_dir / "storage" / exchange / symbol.upper() / interval
        if storage_path.exists():
            return storage_path
        
        self.logger.warning(f"No historical data found for {exchange}/{symbol}/{interval}")
        return None
    
    def _load_from_path(self, data_path: Path, start_time: Optional[datetime] = None, 
                        end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load data from path (handles both files and partitioned directories).
        
        Args:
            data_path: Path to data file or directory
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            DataFrame with klines data
        """
        try:
            if data_path.is_file():
                # Load single parquet file
                df = pd.read_parquet(data_path)
            else:
                # Load partitioned parquet directory
                df = pd.read_parquet(data_path)
            
            # Type guard: ensure we have a DataFrame
            if not isinstance(df, pd.DataFrame):
                self.logger.error(f"Loaded data is not a DataFrame: {type(df)}")
                return pd.DataFrame()
            
            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                if df.index.name == 'timestamp':
                    df = df.reset_index()
                else:
                    self.logger.error(f"No timestamp column found in {data_path}")
                    return pd.DataFrame()
            
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            
            # Filter by time range if specified
            if start_time is not None:
                df = df[df['timestamp'] >= start_time]
            if end_time is not None:
                df = df[df['timestamp'] <= end_time]
            
            # Ensure we have required OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns in data: {missing_cols}")
                return pd.DataFrame()
            
            # Sort by timestamp
            df = df.sort_values('timestamp')  # type: ignore
            
            self.logger.debug(f"Loaded {len(df)} records from {data_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {data_path}: {e}")
            return pd.DataFrame()
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def load_klines(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        use_cache: bool = True
    ) -> List[KlineRecord]:
        """
        Load historical klines data.

        Args:
            exchange: Exchange name (e.g., "binance")
            symbol: Trading symbol (e.g., "ETHUSDT")
            interval: Timeframe (e.g., "1m", "15m", "1h")
            start_time: Optional start time
            end_time: Optional end time
            limit: Optional limit on number of records
            use_cache: Whether to use cached data

        Returns:
            List of KlineRecord objects
        """
        tprint(f"[HIST_LOADER] load_klines: exchange={exchange}, symbol={symbol}, interval={interval}, limit={limit}, use_cache={use_cache}")

        cache_key = (exchange.lower(), symbol.lower(), interval)
        
        # Preview load parameters
        load_params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "use_cache": use_cache,
            "start_time": start_time,
            "end_time": end_time
        }
        tprint_data_preview(load_params, "Klines Load Parameters", max_rows=8)
        
        with tprint_timer("Klines data loading"):
            # Check cache
            if use_cache and cache_key in self._cache:
                cache_time = self._cache_timestamps.get(cache_key)
                if cache_time and (datetime.now() - cache_time).total_seconds() < self.cache_ttl_seconds:
                    df = self._cache[cache_key]
                    self.logger.debug(f"Using cached data for {exchange}/{symbol}/{interval}")
                else:
                    # Cache expired, reload
                    df = self._load_data(exchange, symbol, interval, start_time, end_time)
            else:
                # Load fresh data
                df = self._load_data(exchange, symbol, interval, start_time, end_time)
        
        if df.empty:
            tprint(f"[HIST_LOADER] load_klines: No data loaded for {exchange}/{symbol}/{interval}, returning empty list", color="yellow")
            self.logger.warning(f"No data loaded for {exchange}/{symbol}/{interval}, returning empty list")
            return []
        
        # Apply time filters if not already done
        if start_time is not None:
            df = df[df['timestamp'] >= start_time]
        if end_time is not None:
            df = df[df['timestamp'] <= end_time]
        
        # Apply limit
        if limit is not None and isinstance(df, pd.DataFrame):
            # Get most recent records if limit specified
            df = df.tail(limit)  # type: ignore
        
        # Convert to KlineRecord objects
        records = []
        if isinstance(df, pd.DataFrame):
            for _, row in df.iterrows():
                try:
                    # Explicitly convert timestamp to datetime
                    ts = pd.to_datetime(row['timestamp'])
                    records.append(KlineRecord(
                        timestamp=ts if isinstance(ts, datetime) else ts.to_pydatetime(),
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume']),
                        symbol=symbol.upper(),
                        interval=interval
                    ))
                except Exception as e:
                    self.logger.warning(f"Error converting row to KlineRecord: {e}")
                    continue
        
        # Preview loaded data
        if records:
            sample_records = records[:3]  # Show first 3 records as preview
            preview_data = [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume
                }
                for r in sample_records
            ]
            tprint_data_preview(preview_data, f"Loaded Klines Preview ({exchange}/{symbol})", max_rows=3)
        
        tprint(f"[HIST_LOADER] load_klines -> loaded {len(records)} kline records for {exchange}/{symbol}/{interval}")
        self.logger.info(f"Loaded {len(records)} kline records for {exchange}/{symbol}/{interval}")
        
        # Performance logging
        tprint_performance("Klines loading", len(records) * 0.001)  # Estimate time per record
        
        return records
    
    def _load_data(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Internal method to load data and update cache."""
        data_path = self._get_data_path(exchange, symbol, interval)
        
        if data_path is None:
            return pd.DataFrame()
        
        df = self._load_from_path(data_path, start_time, end_time)
        
        # Update cache
        cache_key = (exchange.lower(), symbol.lower(), interval)
        self._cache[cache_key] = df
        self._cache_timestamps[cache_key] = datetime.now()
        
        return df
    
    def get_latest_price(self, exchange: str, symbol: str, interval: str = "1m") -> Optional[float]:
        """
        Get the most recent close price from historical data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            interval: Timeframe (default: "1m")

        Returns:
            Latest close price or None if unavailable
        """
        tprint(f"[HIST_LOADER] get_latest_price: exchange={exchange}, symbol={symbol}, interval={interval}")
        try:
            # Load most recent data
            records = self.load_klines(exchange, symbol, interval, limit=1)
            if records:
                price = records[-1].close
                tprint(f"[HIST_LOADER] get_latest_price -> {price:.6f}")
                return price
            tprint(f"[HIST_LOADER] get_latest_price -> No data available", color="yellow")
            return None
        except Exception as e:
            tprint(f"[HIST_LOADER] get_latest_price -> ERROR: {e}", color="red")
            self.logger.error(f"Error getting latest price: {e}")
            return None
    
    def get_recent_klines(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        lookback_minutes: int = 60
    ) -> List[KlineRecord]:
        """
        Get klines for the last N minutes.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            interval: Timeframe
            lookback_minutes: How many minutes to look back
            
        Returns:
            List of KlineRecord objects
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        return self.load_klines(
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def generate_synthetic_klines(
        self,
        symbol: str,
        interval: str,
        count: int,
        base_price: float = 3000.0,
        volatility: float = 0.02
    ) -> List[KlineRecord]:
        """
        Generate synthetic klines using random walk.

        Args:
            symbol: Trading symbol
            interval: Timeframe
            count: Number of klines to generate
            base_price: Starting price
            volatility: Price volatility (std dev as fraction of price)

        Returns:
            List of synthetic KlineRecord objects
        """
        tprint(f"[HIST_LOADER] generate_synthetic_klines: symbol={symbol}, interval={interval}, count={count}, base_price={base_price}, volatility={volatility}")
        self.logger.info(f"Generating {count} synthetic klines for {symbol}/{interval}")
        
        # Preview generation parameters
        gen_params = {
            "symbol": symbol,
            "interval": interval,
            "count": count,
            "base_price": base_price,
            "volatility": volatility
        }
        tprint_data_preview(gen_params, "Synthetic Klines Generation", max_rows=5)
        
        with tprint_timer("Synthetic klines generation"):
            records = []
            current_time = datetime.now()
            
            # Parse interval to get timedelta
            interval_minutes = self._parse_interval_to_minutes(interval)
            
            for i in range(count):
                # Generate timestamp (going backwards)
                timestamp = current_time - timedelta(minutes=interval_minutes * (count - i - 1))
                
                # Random walk for price
                if i == 0:
                    open_price = base_price
                else:
                    open_price = records[i-1].close
                
                # Generate OHLC with realistic constraints
                change = np.random.normal(0, base_price * volatility)
                close_price = max(open_price + change, 0.01)  # Prevent negative prices
                
                # High and low should bracket open and close
                high_price = max(open_price, close_price) + abs(np.random.normal(0, base_price * volatility * 0.5))
                low_price = min(open_price, close_price) - abs(np.random.normal(0, base_price * volatility * 0.5))
                low_price = max(low_price, 0.01)  # Prevent negative
                
                # Volume
                volume = np.random.uniform(100, 1000)
                
                records.append(KlineRecord(
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    symbol=symbol,
                    interval=interval
                ))

        # Preview first few generated records
        if records:
            preview_records = records[:3]
            preview_data = [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume
                }
                for r in preview_records
            ]
            tprint_data_preview(preview_data, "Generated Synthetic Klines Preview", max_rows=3)

        tprint(f"[HIST_LOADER] generate_synthetic_klines -> generated {len(records)} synthetic klines")
        
        # Performance logging
        tprint_performance("Synthetic klines generation", count * 0.002)  # Estimate time per record
        return records
    
    def _parse_interval_to_minutes(self, interval: str) -> int:
        """Parse interval string to minutes."""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 1440
        else:
            self.logger.warning(f"Unknown interval format: {interval}, defaulting to 1 minute")
            return 1
    
    def is_data_available(self, exchange: str, symbol: str, interval: str) -> bool:
        """
        Check if historical data is available for given parameters.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            interval: Timeframe

        Returns:
            True if data is available, False otherwise
        """
        tprint(f"[HIST_LOADER] is_data_available: exchange={exchange}, symbol={symbol}, interval={interval}")
        data_path = self._get_data_path(exchange, symbol, interval)
        available = data_path is not None
        tprint(f"[HIST_LOADER] is_data_available -> {available}")
        return available
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        tprint(f"[HIST_LOADER] clear_cache: Clearing cache with {len(self._cache)} entries")
        self._cache.clear()
        self._cache_timestamps.clear()
        tprint(f"[HIST_LOADER] clear_cache -> cache cleared")
        self.logger.info("Cache cleared")


# Global instance for easy import
_global_loader: Optional[HistoricalDataLoader] = None


def get_historical_data_loader() -> HistoricalDataLoader:
    """Get or create global HistoricalDataLoader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = HistoricalDataLoader()
    return _global_loader

