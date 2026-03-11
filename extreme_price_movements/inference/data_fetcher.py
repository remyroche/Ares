"""
OHLCV Data Fetcher for Inference with Incremental Updates.

This module handles fetching OHLCV data for inference with:
- Incremental updates (only fetch missing data)
- 15m OHLCV fetching with immediate resampling to 1h
- Proper time indexation (floor to hour)
- Rate limiting between requests
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from extreme_price_movements.data_store import PartitionedOHLCVStore, make_spot_exchange
from extreme_price_movements.utils import tprint, retry_with_backoff


# Default configuration
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LOOKBACK_HOURS = 24 * 60
MAX_RETRIES = 3
BACKOFF_BASE = 1.0
RATE_LIMIT_DELAY = 0.1  # seconds between requests


class DataFetcher:
    """Data fetcher with incremental updates for inference."""
    
    def __init__(self, exchange: Any = None, data_root: str = "data"):
        """Initialize the DataFetcher.
        
        Args:
            exchange: ccxt exchange instance (created if None)
            data_root: Root directory for data storage
        """
        self.exchange = exchange if exchange is not None else make_spot_exchange()
        self.data_root = data_root
        self.ohlcv_store = PartitionedOHLCVStore(data_root, timeframe="1h")
    
    def initialize_with_historical_data(
        self, 
        symbols: List[str], 
        lookback_hours: int = DEFAULT_LOOKBACK_HOURS
    ):
        """On startup: Use existing data + download missing.
        
        1. Check what data we already have
        2. Only fetch what's missing until current time
        3. Resample 15m -> 1h
        
        Args:
            symbols: List of trading symbols
            lookback_hours: Number of hours to look back if no data exists
        """
        # Get current time
        now = pd.Timestamp.now(tz="UTC")
        
        for symbol in symbols:
            tprint(f"Initializing data for {symbol}...")
            
            # Check existing data range
            existing_data = self.ohlcv_store.load(
                symbol, 
                start_ts=None, 
                end_ts=now
            )
            
            # Safely check existing data
            try:
                existing_not_empty = existing_data is not None and isinstance(existing_data, (pd.DataFrame, pd.Series)) and not (hasattr(existing_data, 'empty') and existing_data.empty)
            except Exception:
                existing_not_empty = False
            
            if existing_not_empty:
                # Find gap from last timestamp to now
                last_ts = existing_data.index.max()
                if (now - last_ts) > pd.Timedelta(hours=1):
                    # Fetch missing data
                    tprint(f"  Found gap for {symbol}: last data at {last_ts}, fetching missing...")
                    missing_data = self.fetch_ohlcv(
                        symbol, 
                        start=last_ts + pd.Timedelta(hours=1),
                        end=now
                    )
                    if missing_data is not None and isinstance(missing_data, (pd.DataFrame, pd.Series)) and not (hasattr(missing_data, 'empty') and missing_data.empty):
                        # Resample and merge
                        existing_data = self._resample_and_merge(existing_data, missing_data)
                        self.ohlcv_store.save_partitioned(symbol=symbol, df=existing_data)
                        tprint(f"  Updated {symbol} with {len(missing_data)} new rows")
                else:
                    tprint(f"  {symbol} data is up to date (last: {last_ts})")
            else:
                # No data - fetch from lookback
                tprint(f"  No existing data for {symbol}, fetching last {lookback_hours}h...")
                start = now - pd.Timedelta(hours=lookback_hours)
                data = self.fetch_ohlcv(symbol, start=start, end=now)
                # Validate data is a proper DataFrame before saving
                if isinstance(data, pd.DataFrame) and not (hasattr(data, 'empty') and data.empty):
                    self.ohlcv_store.save_partitioned(symbol=symbol, df=data)
                    tprint(f"  Fetched {len(data)} rows for {symbol}")
                else:
                    tprint(f"  Warning: No valid data returned for {symbol} (type: {type(data).__name__}), skipping save")
    
    def fetch_ohlcv(
        self, 
        symbol: str, 
        start: pd.Timestamp, 
        end: pd.Timestamp
    ) -> pd.DataFrame:
        """Fetch 15m OHLCV and resample to 1h.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            start: Start timestamp
            end: End timestamp
            
        Returns:
            DataFrame with 1h OHLCV data
        """
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
        
        # Fetch 15m data from exchange
        ohlcv_15m = self._fetch_with_retry(
            symbol, 
            timeframe="15m",
            since=int(start.timestamp() * 1000),
            limit=1200  # Max for 15m
        )
        
        # Validate response
        if ohlcv_15m is None:
            tprint(f"  Warning: None returned for {symbol}, returning empty DataFrame")
            return pd.DataFrame()
        if isinstance(ohlcv_15m, str):
            tprint(f"  Warning: Error string returned for {symbol}: {ohlcv_15m[:100]}")
            return pd.DataFrame()
        if not isinstance(ohlcv_15m, list):
            tprint(f"  Warning: Unexpected type {type(ohlcv_15m)} for {symbol}, returning empty DataFrame")
            return pd.DataFrame()
        if len(ohlcv_15m) == 0:
            tprint(f"  Warning: Empty list returned for {symbol}, returning empty DataFrame")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv_15m,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        # Convert timestamp column to datetime with timezone
        timestamps = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = timestamps
        df.set_index("timestamp", inplace=True)
        
        # Filter to requested range
        df = df[(df.index >= start) & (df.index <= end)]
        
        # Safely check for empty df
        try:
            is_empty = df is None or not isinstance(df, (pd.DataFrame, pd.Series)) or (hasattr(df, 'empty') and df.empty)
        except Exception:
            is_empty = True
        
        if is_empty:
            return pd.DataFrame()
        
        # Resample to 1h
        df_1h = self._resample_to_hourly(df)
        
        return df_1h
    
    @retry_with_backoff(retries=MAX_RETRIES, backoff_in_seconds=BACKOFF_BASE)
    def _fetch_with_retry(
        self, 
        symbol: str, 
        timeframe: str, 
        since: int, 
        limit: int
    ) -> List[List]:
        """Fetch OHLCV with retry logic and rate limiting.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for OHLCV
            since: Start time in milliseconds
            limit: Number of candles
            
        Returns:
            List of OHLCV candles
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )
            return ohlcv
        except Exception as e:
            tprint(f"Error fetching OHLCV for {symbol} ({timeframe}): {e}")
            raise
    
    def _resample_to_hourly(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling 1h aggregation over 15m data to produce overlapping 1h bars.

        This retains the 15m timestamps but computes OHLCV over the preceding 60 minutes.
        
        Args:
            df_15m: DataFrame with 15m OHLCV data
            
        Returns:
            DataFrame with 1h rolling OHLCV data on 15m timestamps
        """
        # Safely check df_15m
        try:
            is_empty = df_15m is None or not isinstance(df_15m, (pd.DataFrame, pd.Series)) or (hasattr(df_15m, 'empty') and df_15m.empty)
        except Exception:
            is_empty = True
        
        if is_empty:
            return df_15m.copy()

        # Ensure we have a clean copy sorted by index
        df_15m = df_15m.copy().sort_index()
        
        # We need a 60-minute rolling window, which is 4 bars for 15m data
        # We use a time-based rolling window to be robust against missing data
        # '1h' implies closing the window on the right and including the current row
        rolling = df_15m.rolling('1h')
        
        # Compute rolling OHLCV
        df_1h = pd.DataFrame(index=df_15m.index)
        df_1h['open'] = rolling['open'].apply(lambda x: x.iloc[0] if len(x) > 0 else np.nan, raw=False)
        df_1h['high'] = rolling['high'].max()
        df_1h['low'] = rolling['low'].min()
        df_1h['close'] = rolling['close'].apply(lambda x: x.iloc[-1] if len(x) > 0 else np.nan, raw=False)
        df_1h['volume'] = rolling['volume'].sum()
        
        # Drop rows with all NaN
        df_1h.dropna(how="all", inplace=True)
        
        return df_1h
    
    def _resample_and_merge(
        self, 
        existing: pd.DataFrame, 
        new_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge existing data with new data after resampling.
        
        Args:
            existing: Existing DataFrame
            new_data: New DataFrame to merge
            
        Returns:
            Merged DataFrame
        """
        # Safely check new_data and existing
        try:
            new_not_empty = new_data is not None and isinstance(new_data, (pd.DataFrame, pd.Series)) and not (hasattr(new_data, 'empty') and new_data.empty)
            existing_not_empty = existing is not None and isinstance(existing, (pd.DataFrame, pd.Series)) and not (hasattr(existing, 'empty') and existing.empty)
        except Exception:
            new_not_empty = False
            existing_not_empty = False
        
        if not new_not_empty:
            return existing
        
        if not existing_not_empty:
            return new_data
        
        # Concatenate and remove duplicates
        merged = pd.concat([existing, new_data])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()
        
        return merged
    
    def fetch_incremental(self, symbol: str) -> pd.DataFrame:
        """At runtime: only fetch missing data since last fetch.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Updated DataFrame with all data (existing + new)
        """
        now = pd.Timestamp.now(tz="UTC")
        
        # Load existing data
        existing = self.ohlcv_store.load(symbol, start_ts=None, end_ts=now)
        
        # Safely check existing
        try:
            existing_not_empty = existing is not None and isinstance(existing, (pd.DataFrame, pd.Series)) and not (hasattr(existing, 'empty') and existing.empty)
        except Exception:
            existing_not_empty = False
        
        if existing_not_empty:
            last_ts = existing.index.max()
            
            # Check if we need to fetch new data (more than 15m gap)
            if (now - last_ts) > pd.Timedelta(minutes=15):
                # Fetch only from last timestamp + 15m buffer
                new_data = self.fetch_ohlcv(
                    symbol, 
                    start=last_ts + pd.Timedelta(minutes=15),
                    end=now
                )
                if new_data is not None and isinstance(new_data, (pd.DataFrame, pd.Series)) and not (hasattr(new_data, 'empty') and new_data.empty):
                    # Merge and save
                    merged = self._resample_and_merge(existing, new_data)
                    # Safely check merged before saving
                    try:
                        merged_valid = merged is not None and isinstance(merged, (pd.DataFrame, pd.Series)) and not (hasattr(merged, 'empty') and merged.empty)
                    except Exception:
                        merged_valid = False
                    
                    if merged_valid:
                        self.ohlcv_store.save_partitioned(merged, symbol)
                        tprint(f"Incremental update for {symbol}: added {len(new_data)} new rows")
                        return merged
                    else:
                        tprint(f"Warning: Invalid merged data for {symbol}, skipping save")
                        return existing
            return existing
        else:
            # No existing data - fetch full lookback
            tprint(f"No existing data for {symbol}, fetching full lookback...")
            return self.fetch_ohlcv(
                symbol, 
                start=now - pd.Timedelta(hours=DEFAULT_LOOKBACK_HOURS),
                end=now
            )

    def get_panel(self, symbols: List[str], lookback_hours: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Get OHLCV panel for given symbols.
        
        Args:
            symbols: List of trading symbols
            lookback_hours: Optional number of recent hours to load
            
        Returns:
            Panel dictionary with open, high, low, close, volume DataFrames
        """
        # Fetch OHLCV data for all symbols
        ohlcv_data = {}
        start_ts = None
        if lookback_hours is not None:
            start_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=int(lookback_hours))
        for symbol in symbols:
            try:
                data = self.ohlcv_store.load(symbol, start_ts=start_ts, end_ts=None)
                 # Safely check data
                try:
                    data_not_empty = data is not None and isinstance(data, (pd.DataFrame, pd.Series)) and not (hasattr(data, 'empty') and data.empty)
                except Exception:
                    data_not_empty = False
                
                if data_not_empty:
                    ohlcv_data[symbol] = data
            except Exception as e:
                tprint(f"Warning: Could not load data for {symbol}: {e}")
        
        # Convert to panel format
        return get_panel_from_dict(ohlcv_data)


# Backwards compatibility: Keep existing functions for non-class usage
def make_exchange() -> Any:
    """Create and return a Binance spot exchange instance.
    
    Returns:
        ccxt.binance exchange instance with rate limiting enabled
    """
    return make_spot_exchange()


@retry_with_backoff(retries=MAX_RETRIES, backoff_in_seconds=BACKOFF_BASE)
def fetch_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str = DEFAULT_TIMEFRAME,
    since: Optional[int] = None,
    limit: int = 100,
) -> List[List]:
    """Fetch OHLCV data for a single symbol using ccxt.
    
    Args:
        exchange: ccxt exchange instance
        symbol: Trading symbol (e.g., "BTC/USDT")
        timeframe: Timeframe (e.g., "1h", "4h", "1d")
        since: Start time in milliseconds (optional)
        limit: Number of candles to fetch
        
    Returns:
        List of OHLCV candles [timestamp, open, high, low, close, volume]
    """
    try:
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
        
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
        )
        return ohlcv
    except Exception as e:
        tprint(f"Error fetching OHLCV for {symbol}: {e}")
        raise


def convert_ohlcv_to_dataframe(ohlcv: List[List], symbol: str) -> pd.DataFrame:
    """Convert ccxt OHLCV format to pandas DataFrame.
    
    Args:
        ohlcv: List of OHLCV candles
        symbol: Symbol for the data
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if not ohlcv:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    
    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    
    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["symbol"] = symbol
    
    return df


def fetch_ohlcv_for_symbols(
    symbols: List[str],
    exchange: Optional[Any] = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    lookback_periods: int = 48,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple symbols.
    
    Args:
        symbols: List of trading symbols
        exchange: ccxt exchange instance (created if None)
        timeframe: Timeframe for OHLCV data
        lookback_periods: Number of periods to look back
        
    Returns:
        Dictionary mapping symbol to OHLCV DataFrame
    """
    if exchange is None:
        exchange = make_exchange()
    
    # Calculate start time based on lookback periods
    # Add some buffer to account for weekends/gaps
    periods_per_day = 24 if timeframe == "1h" else (24 // 4 if timeframe == "4h" else 1)
    limit = lookback_periods + 24  # Add buffer
    
    results = {}
    
    for symbol in symbols:
        try:
            ohlcv = fetch_ohlcv(exchange, symbol, timeframe, limit=limit)
            if ohlcv:
                df = convert_ohlcv_to_dataframe(ohlcv, symbol)
                results[symbol] = df
                tprint(f"Fetched {len(df)} candles for {symbol}")
            else:
                tprint(f"No data for {symbol}")
        except Exception as e:
            tprint(f"Failed to fetch {symbol}: {e}")
            continue
    
    return results


def fetch_latest_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> Optional[pd.DataFrame]:
    """Fetch the latest OHLCV candle for a symbol.
    
    Args:
        exchange: ccxt exchange instance
        symbol: Trading symbol
        timeframe: Timeframe
        
    Returns:
        DataFrame with latest candle or None if failed
    """
    try:
        ohlcv = fetch_ohlcv(exchange, symbol, timeframe, limit=1)
        if ohlcv:
            return convert_ohlcv_to_dataframe(ohlcv, symbol)
        return None
    except Exception as e:
        tprint(f"Error fetching latest OHLCV for {symbol}: {e}")
        return None


def get_panel_from_dict(
    ohlcv_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Convert dictionary of OHLCV DataFrames to panel format.
    
    Creates a dictionary of DataFrames for each price type (open, high, low, close, volume)
    indexed by datetime and columns by symbol.
    
    Args:
        ohlcv_data: Dictionary mapping symbol to OHLCV DataFrame
        
    Returns:
        Dictionary with keys: open, high, low, close, volume
    """
    panel = {
        "open": pd.DataFrame(),
        "high": pd.DataFrame(),
        "low": pd.DataFrame(),
        "close": pd.DataFrame(),
        "volume": pd.DataFrame(),
    }
    
    # Find common index (union of all datetimes)
    all_indexes = []
    for df in ohlcv_data.values():
        # Safely check df
        try:
            df_not_empty = df is not None and isinstance(df, (pd.DataFrame, pd.Series)) and not (hasattr(df, 'empty') and df.empty)
        except Exception:
            df_not_empty = False
        
        if df_not_empty:
            all_indexes.append(df.index)
    
    if not all_indexes:
        return panel
    
    # Union of all timestamps
    common_index = all_indexes[0]
    for idx in all_indexes[1:]:
        common_index = common_index.union(idx)
    
    common_index = sorted(common_index)
    
    # Build panel
    for symbol, df in ohlcv_data.items():
        # Safely check df
        try:
            df_not_empty = df is not None and isinstance(df, (pd.DataFrame, pd.Series)) and not (hasattr(df, 'empty') and df.empty)
        except Exception:
            df_not_empty = False
        
        if not df_not_empty:
            continue
        
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                series = df[col].rename(symbol)
                panel[col] = panel[col].join(series, how="outer")
    
    # Reindex to common_index
    for col in panel:
        panel[col] = panel[col].reindex(common_index)
        panel[col] = panel[col].sort_index()
    
    return panel


def fetch_and_build_panel(
    symbols: List[str],
    exchange: Optional[Any] = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    lookback_periods: int = 48,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data and build panel format.
    
    Args:
        symbols: List of trading symbols
        exchange: ccxt exchange instance
        timeframe: Timeframe for OHLCV
        lookback_periods: Number of periods to look back
        
    Returns:
        Panel dictionary with open, high, low, close, volume DataFrames
    """
    ohlcv_data = fetch_ohlcv_for_symbols(
        symbols=symbols,
        exchange=exchange,
        timeframe=timeframe,
        lookback_periods=lookback_periods,
    )
    
    return get_panel_from_dict(ohlcv_data)
