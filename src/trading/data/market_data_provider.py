"""
Market Data Provider

Unified interface for market data from various sources including exchanges,
data providers, and historical data sources.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import pandas as pd

from src.trading.utils.ohlcv import ensure_ohlcv_dataframe

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from ..config.execution_config import ExchangeType

logger = system_logger.getChild('MarketDataProvider')

@dataclass
class MarketData:
    """Market data structure."""
    timestamp: datetime
    symbol: str
    exchange: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Dict[str, Any] = None

class MarketDataProvider:
    """
    Unified market data provider that aggregates data from multiple sources.
    
    Supports real-time and historical data from various exchanges and data providers.
    """
    
    def __init__(self, exchange: ExchangeType = ExchangeType.BINANCE_TESTNET):
        self.exchange = exchange
        self.logger = logger.getChild(f'{exchange.value}')
        
        # Data sources
        self.exchange_client = None
        self.historical_data_cache: Dict[str, pd.DataFrame] = {}
        
        # Configuration
        self.cache_size = 10000  # Maximum candles to cache
        self.cache_ttl = 3600  # Cache TTL in seconds
        
        # State
        self.is_initialized = False
        self.last_update_time: Dict[str, datetime] = {}
        self._default_ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        
    @handles_errors
    async def initialize(self) -> bool:
        """Initialize market data provider."""
        try:
            self.logger.info(f"Initializing Market Data Provider for {self.exchange.value}...")
            
            # Initialize exchange client
            await self._initialize_exchange_client()
            
            self.is_initialized = True
            self.logger.info("✅ Market Data Provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Market Data Provider: {e}")
            return False
    
    async def _initialize_exchange_client(self):
        """Initialize exchange client."""
        try:
            # Use existing exchange factory
            from exchanges.factory import ExchangeFactory
            
            self.exchange_client = ExchangeFactory.get_exchange(self.exchange.value)
            if self.exchange_client:
                success = await self.exchange_client.initialize()
                if not success:
                    self.logger.warning("⚠️ Failed to initialize exchange client")
                    self.exchange_client = None
            else:
                self.logger.warning("⚠️ Could not create exchange client")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Exchange client initialization failed: {e}")
            self.exchange_client = None
    
    @handles_errors
    @log_execution_time()
    @traced(span_name="get_latest_data")
    async def get_latest_data(self, symbol: str, interval: str = "1m") -> Optional[MarketData]:
        """
        Get latest market data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "ETHUSDT")
            interval: Data interval (1m, 5m, 15m, 1h, etc.)
            
        Returns:
            MarketData: Latest market data or None if unavailable
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Market Data Provider not initialized")
            
            # Try to get from exchange first
            if self.exchange_client:
                klines = await self.exchange_client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=1
                )
                
                if klines:
                    latest_kline = klines[0]
                    return MarketData(
                        timestamp=latest_kline.timestamp,
                        symbol=symbol,
                        exchange=self.exchange.value,
                        open=latest_kline.open,
                        high=latest_kline.high,
                        low=latest_kline.low,
                        close=latest_kline.close,
                        volume=latest_kline.volume,
                        metadata={
                            'interval': interval,
                            'source': 'exchange',
                            'data_quality': 'live'
                        }
                    )
            
            # Fallback to cached data
            return await self._get_cached_latest_data(symbol, interval)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get latest data for {symbol}: {e}")
            return None
    
    @handles_errors
    @log_execution_time()
    @traced(span_name="get_historical_data")
    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            start_time: Start time for data
            end_time: End time for data
            limit: Maximum number of candles
            
        Returns:
            pd.DataFrame: Historical market data
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Market Data Provider not initialized")
            
            # Check cache first
            cache_key = f"{symbol}_{interval}"
            if cache_key in self.historical_data_cache:
                cached_data = self.historical_data_cache[cache_key]
                if self._is_cache_valid(cached_data, start_time, end_time):
                    return self._filter_cached_data(cached_data, start_time, end_time, limit)
            
            # Fetch from exchange
            if self.exchange_client:
                klines = await self.exchange_client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                
                if klines:
                    # Convert to DataFrame
                    data = []
                    for kline in klines:
                        data.append({
                            'timestamp': kline.timestamp,
                            'open': kline.open,
                            'high': kline.high,
                            'low': kline.low,
                            'close': kline.close,
                            'volume': kline.volume,
                            'symbol': symbol,
                            'interval': interval
                        })
                    
                    df = pd.DataFrame(data)
                    df.set_index('timestamp', inplace=True)
                    
                    # Update cache
                    self._update_cache(cache_key, df)

                    return df
            
            # Return empty DataFrame if no data available
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    @handles_errors
    @log_execution_time()
    @traced(span_name="get_multiple_symbols")
    async def get_multiple_symbols(
        self, 
        symbols: List[str], 
        interval: str = "1m",
        limit: int = 100
    ) -> Dict[str, MarketData]:
        """
        Get latest data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            interval: Data interval
            limit: Maximum number of candles per symbol
            
        Returns:
            Dict[str, MarketData]: Latest data for each symbol
        """
        try:
            results = {}
            
            # Fetch data for all symbols concurrently
            tasks = []
            for symbol in symbols:
                task = self.get_latest_data(symbol, interval)
                tasks.append((symbol, task))
            
            # Wait for all tasks to complete
            for symbol, task in tasks:
                try:
                    data = await task
                    if data:
                        results[symbol] = data
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to get data for {symbol}: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get multiple symbols data: {e}")
            return {}
    
    async def _get_cached_latest_data(self, symbol: str, interval: str) -> Optional[MarketData]:
        """Get latest data from cache."""
        try:
            cache_key = f"{symbol}_{interval}"
            if cache_key in self.historical_data_cache:
                cached_data = self.historical_data_cache[cache_key]
                if not cached_data.empty:
                    latest_row = cached_data.iloc[-1]
                    return MarketData(
                        timestamp=latest_row.name,  # timestamp is index
                        symbol=symbol,
                        exchange=self.exchange.value,
                        open=latest_row['open'],
                        high=latest_row['high'],
                        low=latest_row['low'],
                        close=latest_row['close'],
                        volume=latest_row['volume'],
                        metadata={
                            'interval': interval,
                            'source': 'cache',
                            'data_quality': 'historical'
                        }
                    )
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get cached data for {symbol}: {e}")
            return None
    
    def _is_cache_valid(self, cached_data: pd.DataFrame, start_time: Optional[datetime], end_time: Optional[datetime]) -> bool:
        """Check if cached data is valid for the requested time range."""
        try:
            if cached_data.empty:
                return False
            
            # Check if cache covers the requested time range
            cache_start = cached_data.index[0]
            cache_end = cached_data.index[-1]
            
            if start_time and cache_start > start_time:
                return False
            if end_time and cache_end < end_time:
                return False
            
            # Check cache age
            cache_age = (datetime.now() - cache_end).total_seconds()
            return cache_age < self.cache_ttl
            
        except:
            return False
    
    def _filter_cached_data(
        self, 
        cached_data: pd.DataFrame, 
        start_time: Optional[datetime], 
        end_time: Optional[datetime], 
        limit: int
    ) -> pd.DataFrame:
        """Filter cached data based on time range and limit."""
        try:
            filtered_data = cached_data.copy()
            
            # Apply time filters
            if start_time:
                filtered_data = filtered_data[filtered_data.index >= start_time]
            if end_time:
                filtered_data = filtered_data[filtered_data.index <= end_time]
            
            # Apply limit
            if len(filtered_data) > limit:
                filtered_data = filtered_data.tail(limit)
            
            return filtered_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to filter cached data: {e}")
            return cached_data
    
    def _update_cache(self, cache_key: str, new_data: pd.DataFrame):
        """Update data cache."""
        try:
            if cache_key in self.historical_data_cache:
                # Merge with existing cache
                existing_data = self.historical_data_cache[cache_key]
                combined_data = pd.concat([existing_data, new_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data = combined_data.sort_index()
            else:
                combined_data = new_data.copy()

            # Limit cache size
            if len(combined_data) > self.cache_size:
                combined_data = combined_data.tail(self.cache_size)

            self.historical_data_cache[cache_key] = combined_data
            self.last_update_time[cache_key] = datetime.utcnow()

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update cache for {cache_key}: {e}")

    def _is_cache_fresh(self, cache_key: str) -> bool:
        """Check whether cached data has been refreshed within the TTL."""
        last_update = self.last_update_time.get(cache_key)
        if last_update is None:
            return False
        try:
            return (datetime.utcnow() - last_update).total_seconds() < self.cache_ttl
        except Exception:
            return False

    def _get_cached_dataframe(self, cache_key: str, limit: int, allow_stale: bool = False) -> Optional[pd.DataFrame]:
        """Retrieve cached OHLCV data respecting TTL constraints."""
        cached = self.historical_data_cache.get(cache_key)
        if cached is None or cached.empty:
            return None
        if not allow_stale and not self._is_cache_fresh(cache_key):
            return None
        return ensure_ohlcv_dataframe(
            cached,
            required_columns=self._default_ohlcv_columns,
            limit=limit,
        )

    async def _get_symbol_interval_dataframe(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol/interval pair with caching."""

        cache_key = f"{symbol}_{interval}"

        if not force_refresh:
            cached_df = self._get_cached_dataframe(cache_key, limit)
            if cached_df is not None and not cached_df.empty:
                return cached_df

        df = await self.get_historical_data(symbol, interval, limit=limit)

        if df is None or df.empty:
            # Optionally return stale cache to avoid gaps if available
            cached_df = self._get_cached_dataframe(cache_key, limit, allow_stale=True)
            if cached_df is not None:
                return cached_df
            return pd.DataFrame(columns=self._default_ohlcv_columns)

        return ensure_ohlcv_dataframe(
            df,
            required_columns=self._default_ohlcv_columns,
            limit=limit,
        )

    async def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframe_limits: Optional[Dict[str, int]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Return a cached bundle of OHLCV data for the requested symbol."""

        if not self.is_initialized:
            raise RuntimeError("Market Data Provider not initialized")

        limits = timeframe_limits or {"15m": 500, "1h": 500}
        timeframes: Dict[str, pd.DataFrame] = {}

        for interval, limit in limits.items():
            timeframes[interval] = await self._get_symbol_interval_dataframe(
                symbol,
                interval,
                limit=limit,
                force_refresh=force_refresh,
            )

        latest_price = None
        latest_timestamp = None

        for preferred_interval in ("15m", "1h"):
            if preferred_interval in timeframes and not timeframes[preferred_interval].empty:
                latest_price = float(timeframes[preferred_interval]["close"].iloc[-1])
                latest_timestamp = timeframes[preferred_interval].index[-1]
                break

        if latest_price is None:
            for df in timeframes.values():
                if not df.empty:
                    latest_price = float(df["close"].iloc[-1])
                    latest_timestamp = df.index[-1]
                    break

        cache_timestamp = None
        for interval in limits.keys():
            cache_key = f"{symbol}_{interval}"
            if cache_key in self.last_update_time:
                cache_timestamp = self.last_update_time[cache_key]
                break

        return {
            "symbol": symbol,
            "latest_price": latest_price,
            "latest_timestamp": latest_timestamp,
            "timeframes": timeframes,
            "metadata": {
                "source": self.exchange.value if self.exchange else "unknown",
                "cache_timestamp": cache_timestamp,
            },
        }

    async def get_ethusdt_multi_timeframe_data(
        self,
        limit_15m: int = 500,
        limit_1h: int = 500,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper returning ETHUSDT multi-timeframe data."""

        limits = {"15m": limit_15m, "1h": limit_1h}
        return await self.get_multi_timeframe_data(
            symbol="ETHUSDT",
            timeframe_limits=limits,
            force_refresh=force_refresh,
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.historical_data_cache),
            'total_candles': sum(len(df) for df in self.historical_data_cache.values()),
            'cache_keys': list(self.historical_data_cache.keys()),
            'last_update_times': self.last_update_time.copy()
        }
    
    def clear_cache(self, symbol: Optional[str] = None, interval: Optional[str] = None):
        """Clear data cache."""
        try:
            if symbol and interval:
                cache_key = f"{symbol}_{interval}"
                if cache_key in self.historical_data_cache:
                    del self.historical_data_cache[cache_key]
            else:
                self.historical_data_cache.clear()
            
            self.logger.info(f"Cache cleared for {symbol or 'all symbols'}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to clear cache: {e}")
    
    async def stop(self):
        """Stop market data provider."""
        try:
            self.logger.info("🛑 Stopping Market Data Provider...")
            
            # Close exchange client
            if self.exchange_client:
                await self.exchange_client.close()
            
            # Clear cache
            self.clear_cache()
            
            self.is_initialized = False
            self.logger.info("✅ Market Data Provider stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping Market Data Provider: {e}")