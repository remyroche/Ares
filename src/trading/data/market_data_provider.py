"""
Market Data Provider

Unified interface for market data from various sources including exchanges,
data providers, and historical data sources.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import pandas as pd

from src.trading.utils.ohlcv import ensure_ohlcv_dataframe

from src.utils.logger import system_logger
from src.utils.tprint import tprint
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
    metadata: Optional[Dict[str, Any]] = None

class MarketDataProvider:
    """
    Unified market data provider that aggregates data from multiple sources.

    Supports real-time and historical data from various exchanges and data providers.
    """

    def __init__(self, exchange: ExchangeType = ExchangeType.BINANCE_TESTNET) -> None:
        tprint(f"🔄 Initializing Market Data Provider for {exchange.value}")
        self.exchange: ExchangeType = exchange
        self.logger = logger.getChild(f'{exchange.value}')

        # Data sources
        self.exchange_client: Optional[Any] = None
        self.historical_data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_lock: asyncio.Lock = asyncio.Lock()  # Lock for thread-safe cache operations

        # Configuration
        self.cache_size: int = 10000  # Maximum candles to cache
        self.cache_ttl: int = 3600  # Cache TTL in seconds

        # State
        self.is_initialized: bool = False
        self.last_update_time: Dict[str, datetime] = {}
        self._default_ohlcv_columns: List[str] = ['open', 'high', 'low', 'close', 'volume']

    @handles_errors
    async def initialize(self) -> bool:
        """Initialize market data provider."""
        try:
            tprint(f"🔄 Initializing Market Data Provider for {self.exchange.value}...")
            self.logger.info(f"Initializing Market Data Provider for {self.exchange.value}...")

            # Initialize exchange client
            await self._initialize_exchange_client()

            self.is_initialized = True
            tprint("✅ Market Data Provider initialized successfully")
            self.logger.info("✅ Market Data Provider initialized successfully")
            return True

        except Exception as e:
            tprint(f"❌ Failed to initialize Market Data Provider: {e}")
            self.logger.error(f"❌ Failed to initialize Market Data Provider: {e}")
            return False

    async def _initialize_exchange_client(self) -> None:
        """Initialize exchange client."""
        try:
            # Use existing exchange factory
            from exchanges.factory import ExchangeFactory

            tprint(f"🔄 Initializing exchange client for {self.exchange.value}...")
            self.exchange_client = ExchangeFactory.get_exchange(self.exchange.value)
            if self.exchange_client:
                success: bool = await self.exchange_client.initialize()
                if not success:
                    tprint("⚠️ Failed to initialize exchange client")
                    self.logger.warning("⚠️ Failed to initialize exchange client")
                    self.exchange_client = None
                else:
                    tprint(f"✅ Exchange client initialized: {self.exchange.value}")
            else:
                tprint("⚠️ Could not create exchange client")
                self.logger.warning("⚠️ Could not create exchange client")

        except Exception as e:
            tprint(f"❌ Exchange client initialization failed: {e}")
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
                tprint("❌ Market Data Provider not initialized")
                raise RuntimeError("Market Data Provider not initialized")

            # Try to get from exchange first
            if self.exchange_client:
                tprint(f"📊 Fetching latest data for {symbol} ({interval}) from exchange...")
                klines = await self.exchange_client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=1
                )

                if klines:
                    latest_kline = klines[0]
                    tprint(f"✅ Retrieved latest data for {symbol} from exchange")
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
            tprint(f"📊 Fetching cached data for {symbol} ({interval})...")
            return await self._get_cached_latest_data(symbol, interval)

        except Exception as e:
            tprint(f"❌ Failed to get latest data for {symbol}: {e}")
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
                tprint("❌ Market Data Provider not initialized")
                raise RuntimeError("Market Data Provider not initialized")

            # Check cache first (thread-safe)
            cache_key: str = f"{symbol}_{interval}"
            async with self._cache_lock:
                if cache_key in self.historical_data_cache:
                    cached_data: pd.DataFrame = self.historical_data_cache[cache_key].copy()
                    if self._is_cache_valid(cached_data, start_time, end_time):
                        tprint(f"📊 Using cached historical data for {symbol} ({interval})")
                        return self._filter_cached_data(cached_data, start_time, end_time, limit)

            # Fetch from exchange
            if self.exchange_client:
                tprint(f"📊 Fetching historical data for {symbol} ({interval}) from exchange...")
                klines = await self.exchange_client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )

                if klines:
                    # Convert to DataFrame
                    data: List[Dict[str, Any]] = []
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

                    df: pd.DataFrame = pd.DataFrame(data)
                    df.set_index('timestamp', inplace=True)

                    # Update cache (thread-safe)
                    async with self._cache_lock:
                        self._update_cache(cache_key, df)

                    tprint(f"✅ Retrieved {len(df)} historical records for {symbol} ({interval})")
                    return df
                else:
                    tprint(f"⚠️ No historical data returned for {symbol} ({interval})")

            # Return empty DataFrame if no data available
            tprint(f"⚠️ No exchange client available, returning empty DataFrame")
            return pd.DataFrame()

        except Exception as e:
            tprint(f"❌ Failed to get historical data for {symbol}: {e}")
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
            tprint(f"📊 Fetching data for {len(symbols)} symbols ({interval})...")
            results: Dict[str, MarketData] = {}

            # Fetch data for all symbols concurrently
            tasks: List[tuple[str, Any]] = []
            for symbol in symbols:
                task = self.get_latest_data(symbol, interval)
                tasks.append((symbol, task))

            # Wait for all tasks to complete
            for symbol, task in tasks:
                try:
                    data: Optional[MarketData] = await task
                    if data:
                        results[symbol] = data
                except Exception as e:
                    tprint(f"⚠️ Failed to get data for {symbol}: {e}")
                    self.logger.warning(f"⚠️ Failed to get data for {symbol}: {e}")

            tprint(f"✅ Retrieved data for {len(results)}/{len(symbols)} symbols")
            return results

        except Exception as e:
            tprint(f"❌ Failed to get multiple symbols data: {e}")
            self.logger.error(f"❌ Failed to get multiple symbols data: {e}")
            return {}

    async def _get_cached_latest_data(self, symbol: str, interval: str) -> Optional[MarketData]:
        """Get latest data from cache."""
        try:
            cache_key: str = f"{symbol}_{interval}"
            if cache_key in self.historical_data_cache:
                cached_data: pd.DataFrame = self.historical_data_cache[cache_key]
                if not cached_data.empty:
                    latest_row = cached_data.iloc[-1]
                    tprint(f"📊 Retrieved cached data for {symbol} ({interval})")
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
            tprint(f"⚠️ No cached data available for {symbol} ({interval})")
            return None

        except Exception as e:
            tprint(f"❌ Failed to get cached data for {symbol}: {e}")
            self.logger.warning(f"⚠️ Failed to get cached data for {symbol}: {e}")
            return None

    def _is_cache_valid(self, cached_data: pd.DataFrame, start_time: Optional[datetime], end_time: Optional[datetime]) -> bool:
        """Check if cached data is valid for the requested time range."""
        try:
            if cached_data.empty:
                return False

            # Check if cache covers the requested time range
            cache_start: datetime = cached_data.index[0]
            cache_end: datetime = cached_data.index[-1]

            if start_time and cache_start > start_time:
                return False
            if end_time and cache_end < end_time:
                return False

            # Check cache age (use UTC consistently)
            cache_age: float = (datetime.now(timezone.utc) - cache_end).total_seconds()
            is_valid: bool = cache_age < self.cache_ttl
            if not is_valid:
                tprint(f"⚠️ Cache expired (age: {cache_age:.0f}s, TTL: {self.cache_ttl}s)")
            return is_valid

        except Exception as e:
            tprint(f"⚠️ Error validating cache: {e}")
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

    def _update_cache(self, cache_key: str, new_data: pd.DataFrame) -> None:
        """Update data cache."""
        try:
            if cache_key in self.historical_data_cache:
                # Merge with existing cache
                existing_data: pd.DataFrame = self.historical_data_cache[cache_key]
                combined_data: pd.DataFrame = pd.concat([existing_data, new_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data = combined_data.sort_index()
                tprint(f"💾 Merged cache for {cache_key}: {len(existing_data)} -> {len(combined_data)} records")
            else:
                combined_data = new_data.copy()
                tprint(f"💾 Created new cache for {cache_key}: {len(combined_data)} records")

            # Limit cache size
            if len(combined_data) > self.cache_size:
                combined_data = combined_data.tail(self.cache_size)
                tprint(f"⚠️ Cache limit reached for {cache_key}, truncated to {self.cache_size} records")

            self.historical_data_cache[cache_key] = combined_data
            self.last_update_time[cache_key] = datetime.now(timezone.utc)

        except Exception as e:
            tprint(f"❌ Failed to update cache for {cache_key}: {e}")
            self.logger.warning(f"⚠️ Failed to update cache for {cache_key}: {e}")

    def _is_cache_fresh(self, cache_key: str) -> bool:
        """Check whether cached data has been refreshed within the TTL."""
        last_update = self.last_update_time.get(cache_key)
        if last_update is None:
            return False
        try:
            # Ensure timezone-aware comparison
            if last_update.tzinfo is None:
                last_update = last_update.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - last_update).total_seconds() < self.cache_ttl
        except Exception:
            return False

    async def _get_cached_dataframe(self, cache_key: str, limit: int, allow_stale: bool = False) -> Optional[pd.DataFrame]:
        """Retrieve cached OHLCV data respecting TTL constraints (thread-safe)."""
        async with self._cache_lock:
            cached: Optional[pd.DataFrame] = self.historical_data_cache.get(cache_key)
            if cached is None or cached.empty:
                tprint(f"⚠️ No cached data for {cache_key}")
                return None
            if not allow_stale and not self._is_cache_fresh(cache_key):
                tprint(f"⚠️ Cache stale for {cache_key}")
                return None
            tprint(f"📊 Using cached dataframe for {cache_key} (limit: {limit})")
            return ensure_ohlcv_dataframe(
                cached.copy(),
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
        cache_key: str = f"{symbol}_{interval}"

        if not force_refresh:
            cached_df: Optional[pd.DataFrame] = await self._get_cached_dataframe(cache_key, limit)
            if cached_df is not None and not cached_df.empty:
                return cached_df

        tprint(f"📊 Fetching fresh data for {cache_key} (limit: {limit})")
        df: pd.DataFrame = await self.get_historical_data(symbol, interval, limit=limit)

        if df is None or df.empty:
            # Optionally return stale cache to avoid gaps if available
            tprint(f"⚠️ No fresh data for {cache_key}, checking stale cache...")
            cached_df = await self._get_cached_dataframe(cache_key, limit, allow_stale=True)
            if cached_df is not None:
                tprint(f"✅ Using stale cache for {cache_key}")
                return cached_df
            tprint(f"⚠️ No data available for {cache_key}, returning empty DataFrame")
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
            tprint("❌ Market Data Provider not initialized")
            raise RuntimeError("Market Data Provider not initialized")

        limits: Dict[str, int] = timeframe_limits or {"15m": 500, "1h": 500}
        tprint(f"📊 Fetching multi-timeframe data for {symbol} (intervals: {list(limits.keys())})")
        timeframes: Dict[str, pd.DataFrame] = {}

        for interval, limit in limits.items():
            timeframes[interval] = await self._get_symbol_interval_dataframe(
                symbol,
                interval,
                limit=limit,
                force_refresh=force_refresh,
            )

        latest_price: Optional[float] = None
        latest_timestamp: Optional[datetime] = None

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

        cache_timestamp: Optional[datetime] = None
        for interval in limits.keys():
            cache_key: str = f"{symbol}_{interval}"
            if cache_key in self.last_update_time:
                cache_timestamp = self.last_update_time[cache_key]
                break

        result: Dict[str, Any] = {
            "symbol": symbol,
            "latest_price": latest_price,
            "latest_timestamp": latest_timestamp,
            "timeframes": timeframes,
            "metadata": {
                "source": self.exchange.value if self.exchange else "unknown",
                "cache_timestamp": cache_timestamp,
            },
        }
        tprint(f"✅ Retrieved multi-timeframe data for {symbol}: {len(timeframes)} timeframes")
        return result

    async def get_ethusdt_multi_timeframe_data(
        self,
        limit_15m: int = 500,
        limit_1h: int = 500,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper returning ETHUSDT multi-timeframe data."""
        limits: Dict[str, int] = {"15m": limit_15m, "1h": limit_1h}
        tprint(f"📊 Fetching ETHUSDT multi-timeframe data (15m: {limit_15m}, 1h: {limit_1h})")
        return await self.get_multi_timeframe_data(
            symbol="ETHUSDT",
            timeframe_limits=limits,
            force_refresh=force_refresh,
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_candles: int = sum(len(df) for df in self.historical_data_cache.values())
        stats: Dict[str, Any] = {
            'cache_size': len(self.historical_data_cache),
            'total_candles': total_candles,
            'cache_keys': list(self.historical_data_cache.keys()),
            'last_update_times': self.last_update_time.copy()
        }
        tprint(f"📊 Cache stats: {len(self.historical_data_cache)} keys, {total_candles} total candles")
        return stats

    def clear_cache(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> None:
        """Clear data cache."""
        try:
            if symbol and interval:
                cache_key: str = f"{symbol}_{interval}"
                if cache_key in self.historical_data_cache:
                    del self.historical_data_cache[cache_key]
                    if cache_key in self.last_update_time:
                        del self.last_update_time[cache_key]
                    tprint(f"🧹 Cache cleared for {symbol} ({interval})")
            else:
                self.historical_data_cache.clear()
                self.last_update_time.clear()
                tprint("🧹 All cache cleared")

            self.logger.info(f"Cache cleared for {symbol or 'all symbols'}")

        except Exception as e:
            tprint(f"❌ Failed to clear cache: {e}")
            self.logger.warning(f"⚠️ Failed to clear cache: {e}")

    async def stop(self) -> None:
        """Stop market data provider."""
        try:
            tprint("🛑 Stopping Market Data Provider...")
            self.logger.info("🛑 Stopping Market Data Provider...")

            # Close exchange client
            if self.exchange_client:
                await self.exchange_client.close()
                tprint("✅ Exchange client closed")

            # Clear cache
            self.clear_cache()

            self.is_initialized = False
            tprint("✅ Market Data Provider stopped successfully")
            self.logger.info("✅ Market Data Provider stopped successfully")

        except Exception as e:
            tprint(f"❌ Error stopping Market Data Provider: {e}")
            self.logger.error(f"❌ Error stopping Market Data Provider: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Dictionary with health status and metrics
        """
        health = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc),
            'checks': {}
        }
        
        # Check initialization
        health['checks']['initialized'] = {
            'status': 'ok' if self.is_initialized else 'failed',
            'value': self.is_initialized
        }
        
        # Check exchange client
        if self.exchange_client:
            health['checks']['exchange_client'] = {
                'status': 'ok',
                'value': 'connected'
            }
        else:
            health['checks']['exchange_client'] = {
                'status': 'warning',
                'value': 'not_connected'
            }
        
        # Check cache health
        cache_stats = self.get_cache_stats()
        cache_usage = cache_stats.get('total_candles', 0) / (self.cache_size * len(cache_stats.get('cache_keys', [])) + 1)
        health['checks']['cache_usage'] = {
            'status': 'ok' if cache_usage < 0.9 else 'warning',
            'value': f"{cache_usage:.1%}"
        }
        
        # Check cache freshness
        if self.last_update_time:
            oldest_update = min(self.last_update_time.values())
            if isinstance(oldest_update, datetime):
                if oldest_update.tzinfo is None:
                    oldest_update = oldest_update.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - oldest_update).total_seconds() / 3600
                health['checks']['cache_freshness'] = {
                    'status': 'ok' if age_hours < 24 else 'warning',
                    'value': f"{age_hours:.1f} hours"
                }
        
        # Overall status
        if any(c['status'] != 'ok' for c in health['checks'].values()):
            health['status'] = 'degraded'
        
        return health
