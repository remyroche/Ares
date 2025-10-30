"""
Multi-Timeframe Data Loader

Loads and caches data from multiple timeframes for real cross-TF SR level confirmation.
NOT SIMULATED - actual data from different timeframes.
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TimeframeConfig:
    """Configuration for timeframe hierarchy."""
    base_tf: str
    higher_tfs: List[str]
    lookback_bars: Dict[str, int]  # Bars to load per TF


class MultiTimeframeDataLoader:
    """Loads and caches market data from multiple timeframes.
    
    Real multi-TF implementation - loads actual data from higher timeframes
    for cross-TF level confirmation.
    """
    
    # Timeframe hierarchy mappings - CUSTOMIZED: 15m, 1h, 4h only
    TF_HIERARCHY = {
        '1m': ['1m', '5m', '15m'],
        '5m': ['5m', '15m', '1h'],
        '15m': ['15m', '1h', '4h'],  # Primary: 15m → 1h → 4h
        '1h': ['1h', '4h'],          # Primary: 1h → 4h
        '4h': ['4h'],                 # Primary: 4h only (highest)
        '1d': ['1d'],
        '1w': ['1w'],
    }
    
    # Timeframe to minutes conversion
    TF_TO_MINUTES = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
        '1w': 10080,
        '1M': 43200,
    }
    
    def __init__(self, cache_ttl: int = 300):
        """Initialize multi-timeframe data loader.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.cache = {}  # {(symbol, exchange, tf): (data, timestamp)}
        self.cache_ttl = cache_ttl
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_timeframe_hierarchy(self, base_tf: str) -> List[str]:
        """Get base timeframe + higher timeframes for confirmation.
        
        Args:
            base_tf: Base timeframe (e.g., '1h')
            
        Returns:
            List of timeframes including base + higher TFs
        """
        return self.TF_HIERARCHY.get(base_tf, [base_tf])
    
    def load_timeframe_data(self, symbol: str, exchange: str, timeframe: str,
                           lookback_days: int = 30) -> pd.DataFrame:
        """Load data for specific timeframe with caching.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (e.g., 'binance')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            lookback_days: Days of historical data to load
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = (symbol, exchange, timeframe)
        
        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.logger.debug(f"Cache hit for {symbol} {timeframe}")
                return data.copy()
        
        # Load from source
        self.logger.info(f"Loading {symbol} {timeframe} data ({lookback_days} days)...")
        try:
            data = self._load_from_database(symbol, exchange, timeframe, lookback_days)
            
            # Validate data
            if data is None or len(data) == 0:
                self.logger.warning(f"No data loaded for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Cache it
            self.cache[cache_key] = (data.copy(), time.time())
            
            self.logger.info(f"✅ Loaded {len(data)} bars for {symbol} {timeframe}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def load_multiple_timeframes(self, symbol: str, exchange: str, 
                                 base_timeframe: str,
                                 lookback_days: int = 30) -> Dict[str, pd.DataFrame]:
        """Load base timeframe + all higher timeframes.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            base_timeframe: Base timeframe to detect SR on
            lookback_days: Days of historical data
            
        Returns:
            Dictionary {timeframe: data_df}
        """
        timeframes = self.get_timeframe_hierarchy(base_timeframe)
        
        self.logger.info(f"📊 Loading {len(timeframes)} timeframes: {timeframes}")
        
        data_dict = {}
        for tf in timeframes:
            try:
                # Adjust lookback days for higher timeframes (need more bars)
                tf_lookback = self._calculate_lookback_days(tf, lookback_days)
                data = self.load_timeframe_data(symbol, exchange, tf, tf_lookback)
                
                if not data.empty:
                    data_dict[tf] = data
                else:
                    self.logger.warning(f"⚠️ Empty data for {tf}, skipping")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to load {tf}: {e}")
        
        self.logger.info(f"✅ Loaded {len(data_dict)}/{len(timeframes)} timeframes successfully")
        return data_dict
    
    def _load_from_database(self, symbol: str, exchange: str, timeframe: str,
                           lookback_days: int) -> pd.DataFrame:
        """Load data from historical_data directory (already downloaded data).
        
        Data location: historical_data/EXCHANGE/ASSET/processed/
        Uses partitioned parquet format - does NOT download new data.
        """
        try:
            symbol_lower = symbol.lower()
            exchange_lower = exchange.lower()
            
            # PRIMARY: Load from historical_data/exchange/asset/processed/
            historical_data_path = Path('historical_data') / exchange_lower / symbol_lower / 'processed' / f"{symbol_lower}_{timeframe}"
            
            if historical_data_path.exists():
                self.logger.info(f"✅ Loading from: {historical_data_path}")
                
                # Read partitioned parquet dataset (pandas handles year/month partitions)
                data = pd.read_parquet(historical_data_path)
                
                self.logger.info(f"   Loaded {len(data)} total bars")
                
                # Fix timestamp conversion - check multiple possible timestamp columns
                timestamp_col = None
                for col_name in ['timestamp', 'open_time', 'close_time', 'date', 'datetime']:
                    if col_name in data.columns:
                        timestamp_col = col_name
                        break
                
                if timestamp_col:
                    # Convert to datetime - handle both Unix timestamps (ms) and datetime strings
                    if pd.api.types.is_numeric_dtype(data[timestamp_col]):
                        # Unix timestamp in milliseconds
                        data[timestamp_col] = pd.to_datetime(data[timestamp_col], unit='ms', utc=True)
                    else:
                        # String datetime
                        data[timestamp_col] = pd.to_datetime(data[timestamp_col], utc=True)
                    
                    data = data.set_index(timestamp_col).sort_index()
                    self.logger.info(f"   Set index from '{timestamp_col}' column")
                elif not isinstance(data.index, pd.DatetimeIndex):
                    # Try to convert index if it's numeric (Unix timestamp)
                    if pd.api.types.is_numeric_dtype(data.index):
                        data.index = pd.to_datetime(data.index, unit='ms', utc=True)
                        data = data.sort_index()
                        self.logger.info(f"   Converted numeric index to datetime")
                
                # Filter to lookback period
                if isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
                    end_date = data.index[-1]
                    start_date = end_date - timedelta(days=lookback_days)
                    data = data[data.index >= start_date]
                    self.logger.info(f"   Filtered to last {lookback_days} days: {len(data)} bars")
                
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in data.columns for col in required_cols):
                    return data[required_cols].copy()
                else:
                    missing = [c for c in required_cols if c not in data.columns]
                    self.logger.warning(f"Missing required columns: {missing}")
                    self.logger.info(f"Available columns: {list(data.columns)}")
                    return data.copy()  # Return anyway, might still work
            
            # Fallback: Try data_cache paths
            fallback_paths = [
                Path('data_cache') / exchange_lower / symbol_lower / f"klines_{exchange}_{symbol}_{timeframe}.parquet",
                Path('data_cache') / exchange_lower / symbol_lower / f"{timeframe}.parquet",
            ]
            
            for cache_path in fallback_paths:
                if cache_path.exists():
                    self.logger.info(f"✅ Loading from fallback: {cache_path}")
                    data = pd.read_parquet(cache_path)
                    self.logger.info(f"   Loaded {len(data)} bars")
                    return data
            
            # No data found
            self.logger.error(f"❌ No data found for {symbol} {exchange} {timeframe}")
            self.logger.error(f"   Expected at: {historical_data_path}")
            return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            return pd.DataFrame()
    
    def _calculate_lookback_days(self, timeframe: str, base_lookback: int) -> int:
        """Calculate appropriate lookback days for higher timeframes.
        
        Higher timeframes need more calendar days to get same number of bars.
        """
        tf_minutes = self.TF_TO_MINUTES.get(timeframe, 1440)
        base_minutes = self.TF_TO_MINUTES.get('1d', 1440)
        
        # Scale lookback based on timeframe ratio
        ratio = tf_minutes / base_minutes
        scaled_lookback = int(base_lookback * ratio * 1.5)  # 1.5x for safety
        
        # Minimum and maximum constraints
        return max(30, min(scaled_lookback, 365))
    
    def clear_cache(self, symbol: Optional[str] = None, exchange: Optional[str] = None):
        """Clear cache for specific symbol/exchange or all.
        
        Args:
            symbol: Symbol to clear (None = all)
            exchange: Exchange to clear (None = all)
        """
        if symbol is None and exchange is None:
            # Clear all
            cleared = len(self.cache)
            self.cache.clear()
            self.logger.info(f"🧹 Cleared entire cache ({cleared} entries)")
        else:
            # Clear specific entries
            keys_to_remove = [
                key for key in self.cache.keys()
                if (symbol is None or key[0] == symbol) and
                   (exchange is None or key[1] == exchange)
            ]
            for key in keys_to_remove:
                del self.cache[key]
            self.logger.info(f"🧹 Cleared {len(keys_to_remove)} cache entries")
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        total_size_mb = sum(
            data.memory_usage(deep=True).sum() / 1024 / 1024
            for data, _ in self.cache.values()
        )
        
        # Check expired entries
        current_time = time.time()
        expired = sum(
            1 for _, timestamp in self.cache.values()
            if current_time - timestamp > self.cache_ttl
        )
        
        return {
            'total_entries': total_entries,
            'total_size_mb': total_size_mb,
            'expired_entries': expired,
            'cache_ttl': self.cache_ttl
        }


# Global instance for reuse
_global_loader = None

def get_multi_tf_data_loader(cache_ttl: int = 300) -> MultiTimeframeDataLoader:
    """Get global multi-timeframe data loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = MultiTimeframeDataLoader(cache_ttl=cache_ttl)
    return _global_loader

