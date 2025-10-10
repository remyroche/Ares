"""
OHLCV Data Management

Handles OHLCV data fetching, caching, and processing for different timeframes.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger


class Timeframe(Enum):
    """Timeframe enumeration"""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass
class OHLCVData:
    """OHLCV data structure"""
    symbol: str
    timeframe: Timeframe
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float] = None
    trades_count: Optional[int] = None
    taker_buy_volume: Optional[float] = None
    taker_buy_quote_volume: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OHLCVManager:
    """
    Manages OHLCV data fetching, caching, and processing.
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"OHLCVManager.{exchange_name}")
        
        # Data cache: {symbol: {timeframe: [OHLCVData]}}
        self.ohlcv_cache: Dict[str, Dict[Timeframe, List[OHLCVData]]] = {}
        
        # Cache settings
        self.cache_ttl = {
            Timeframe.MINUTE_1: timedelta(minutes=5),
            Timeframe.MINUTE_3: timedelta(minutes=10),
            Timeframe.MINUTE_5: timedelta(minutes=15),
            Timeframe.MINUTE_15: timedelta(minutes=30),
            Timeframe.MINUTE_30: timedelta(hours=1),
            Timeframe.HOUR_1: timedelta(hours=2),
            Timeframe.HOUR_2: timedelta(hours=4),
            Timeframe.HOUR_4: timedelta(hours=8),
            Timeframe.HOUR_6: timedelta(hours=12),
            Timeframe.HOUR_12: timedelta(days=1),
            Timeframe.DAY_1: timedelta(days=2),
            Timeframe.DAY_3: timedelta(days=5),
            Timeframe.WEEK_1: timedelta(weeks=2),
            Timeframe.MONTH_1: timedelta(days=30)
        }
        
        # Data fetching functions
        self.fetch_functions: Dict[str, callable] = {}
        
        # Data limits
        self.max_candles_per_request = 1000
        self.max_cached_candles = 5000
        
    def register_fetch_functions(
        self,
        get_klines: callable,
        get_historical_klines: Optional[callable] = None
    ) -> None:
        """
        Register exchange-specific OHLCV fetching functions.
        
        Args:
            get_klines: Function to get recent klines
            get_historical_klines: Optional function to get historical klines
        """
        self.fetch_functions = {
            "get_klines": get_klines,
            "get_historical_klines": get_historical_klines
        }
        
        self.logger.info("Registered OHLCV fetching functions")
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 100,
        use_cache: bool = True
    ) -> List[OHLCVData]:
        """
        Get OHLCV data for symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Number of candles to fetch
            use_cache: Whether to use cached data
            
        Returns:
            List of OHLCVData
        """
        try:
            # Check cache first
            if use_cache and self._is_cache_valid(symbol, timeframe):
                cached_data = self._get_cached_data(symbol, timeframe, limit)
                if cached_data:
                    return cached_data
            
            # Fetch fresh data
            if "get_klines" not in self.fetch_functions:
                self.logger.warning("No klines fetch function registered")
                return []
            
            raw_data = await self.fetch_functions["get_klines"](symbol, timeframe.value, limit)
            if not raw_data:
                self.logger.warning(f"No OHLCV data received for {symbol}")
                return []
            
            # Parse OHLCV data
            ohlcv_data = self._parse_ohlcv_data(symbol, timeframe, raw_data)
            
            # Cache the data
            self._cache_data(symbol, timeframe, ohlcv_data)
            
            # Return requested limit
            return ohlcv_data[:limit]
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return []
    
    async def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> List[OHLCVData]:
        """
        Get historical OHLCV data for symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start time
            end_time: End time
            limit: Maximum number of candles
            
        Returns:
            List of OHLCVData
        """
        try:
            if "get_historical_klines" not in self.fetch_functions:
                self.logger.warning("No historical klines fetch function registered")
                return []
            
            raw_data = await self.fetch_functions["get_historical_klines"](
                symbol, timeframe.value, start_time, end_time, limit
            )
            if not raw_data:
                self.logger.warning(f"No historical OHLCV data received for {symbol}")
                return []
            
            # Parse OHLCV data
            ohlcv_data = self._parse_ohlcv_data(symbol, timeframe, raw_data)
            
            return ohlcv_data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical OHLCV data for {symbol}: {e}")
            return []
    
    def _parse_ohlcv_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        raw_data: List[List[Any]]
    ) -> List[OHLCVData]:
        """Parse raw OHLCV data into OHLCVData structures."""
        ohlcv_list = []
        
        for item in raw_data:
            try:
                # Parse timestamp
                if isinstance(item[0], (int, float)):
                    timestamp = datetime.fromtimestamp(item[0] / 1000)
                else:
                    timestamp = datetime.fromisoformat(str(item[0]))
                
                # Parse OHLCV values
                open_price = float(item[1])
                high_price = float(item[2])
                low_price = float(item[3])
                close_price = float(item[4])
                volume = float(item[5])
                
                # Parse additional data if available
                quote_volume = float(item[6]) if len(item) > 6 and item[6] else None
                trades_count = int(item[7]) if len(item) > 7 and item[7] else None
                taker_buy_volume = float(item[8]) if len(item) > 8 and item[8] else None
                taker_buy_quote_volume = float(item[9]) if len(item) > 9 and item[9] else None
                
                ohlcv_data = OHLCVData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    quote_volume=quote_volume,
                    trades_count=trades_count,
                    taker_buy_volume=taker_buy_volume,
                    taker_buy_quote_volume=taker_buy_quote_volume,
                    metadata={"raw_data": item}
                )
                
                ohlcv_list.append(ohlcv_data)
                
            except (ValueError, TypeError, IndexError) as e:
                self.logger.warning(f"Error parsing OHLCV item: {e}")
                continue
        
        return ohlcv_list
    
    def _is_cache_valid(self, symbol: str, timeframe: Timeframe) -> bool:
        """Check if cache is valid for symbol and timeframe."""
        if symbol not in self.ohlcv_cache:
            return False
        
        if timeframe not in self.ohlcv_cache[symbol]:
            return False
        
        data = self.ohlcv_cache[symbol][timeframe]
        if not data:
            return False
        
        # Check if most recent data is within TTL
        latest_timestamp = data[-1].timestamp
        ttl = self.cache_ttl.get(timeframe, timedelta(minutes=5))
        
        return datetime.now() - latest_timestamp < ttl
    
    def _get_cached_data(self, symbol: str, timeframe: Timeframe, limit: int) -> List[OHLCVData]:
        """Get cached data for symbol and timeframe."""
        if symbol not in self.ohlcv_cache:
            return []
        
        if timeframe not in self.ohlcv_cache[symbol]:
            return []
        
        data = self.ohlcv_cache[symbol][timeframe]
        return data[-limit:] if len(data) >= limit else data
    
    def _cache_data(self, symbol: str, timeframe: Timeframe, data: List[OHLCVData]) -> None:
        """Cache OHLCV data."""
        if symbol not in self.ohlcv_cache:
            self.ohlcv_cache[symbol] = {}
        
        if timeframe not in self.ohlcv_cache[symbol]:
            self.ohlcv_cache[symbol][timeframe] = []
        
        # Add new data to cache
        self.ohlcv_cache[symbol][timeframe].extend(data)
        
        # Sort by timestamp
        self.ohlcv_cache[symbol][timeframe].sort(key=lambda x: x.timestamp)
        
        # Remove duplicates
        seen_timestamps = set()
        unique_data = []
        for item in self.ohlcv_cache[symbol][timeframe]:
            if item.timestamp not in seen_timestamps:
                seen_timestamps.add(item.timestamp)
                unique_data.append(item)
        
        self.ohlcv_cache[symbol][timeframe] = unique_data
        
        # Limit cache size
        if len(self.ohlcv_cache[symbol][timeframe]) > self.max_cached_candles:
            self.ohlcv_cache[symbol][timeframe] = self.ohlcv_cache[symbol][timeframe][-self.max_cached_candles:]
    
    async def get_latest_candle(self, symbol: str, timeframe: Timeframe) -> Optional[OHLCVData]:
        """Get the latest candle for symbol and timeframe."""
        data = await self.get_ohlcv(symbol, timeframe, limit=1)
        return data[0] if data else None
    
    async def get_candles_in_range(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime
    ) -> List[OHLCVData]:
        """Get candles within a time range."""
        all_data = await self.get_ohlcv(symbol, timeframe, limit=1000)
        
        filtered_data = [
            candle for candle in all_data
            if start_time <= candle.timestamp <= end_time
        ]
        
        return filtered_data
    
    def get_cached_candles(self, symbol: str, timeframe: Timeframe) -> List[OHLCVData]:
        """Get cached candles without fetching."""
        if symbol not in self.ohlcv_cache:
            return []
        
        if timeframe not in self.ohlcv_cache[symbol]:
            return []
        
        return self.ohlcv_cache[symbol][timeframe].copy()
    
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get OHLCV cache statistics."""
        total_symbols = len(self.ohlcv_cache)
        total_candles = sum(
            len(timeframe_data)
            for symbol_data in self.ohlcv_cache.values()
            for timeframe_data in symbol_data.values()
        )
        
        timeframe_counts = {}
        for symbol_data in self.ohlcv_cache.values():
            for timeframe, data in symbol_data.items():
                timeframe_counts[timeframe.value] = timeframe_counts.get(timeframe.value, 0) + len(data)
        
        return {
            "total_symbols": total_symbols,
            "total_candles": total_candles,
            "timeframe_distribution": timeframe_counts,
            "max_cached_candles": self.max_cached_candles
        }
    
    def cleanup_stale_cache(self) -> int:
        """Clean up stale cache entries."""
        now = datetime.now()
        cleaned_count = 0
        
        for symbol in list(self.ohlcv_cache.keys()):
            for timeframe in list(self.ohlcv_cache[symbol].keys()):
                data = self.ohlcv_cache[symbol][timeframe]
                if not data:
                    continue
                
                ttl = self.cache_ttl.get(timeframe, timedelta(minutes=5))
                if now - data[-1].timestamp > ttl:
                    del self.ohlcv_cache[symbol][timeframe]
                    cleaned_count += 1
            
            # Remove empty symbol entries
            if not self.ohlcv_cache[symbol]:
                del self.ohlcv_cache[symbol]
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} stale cache entries")
        
        return cleaned_count
    
    def invalidate_cache(self, symbol: Optional[str] = None, timeframe: Optional[Timeframe] = None) -> None:
        """Invalidate OHLCV cache."""
        if symbol and timeframe:
            if symbol in self.ohlcv_cache and timeframe in self.ohlcv_cache[symbol]:
                del self.ohlcv_cache[symbol][timeframe]
                self.logger.debug(f"Invalidated cache for {symbol} {timeframe.value}")
        elif symbol:
            self.ohlcv_cache.pop(symbol, None)
            self.logger.debug(f"Invalidated cache for {symbol}")
        else:
            self.ohlcv_cache.clear()
            self.logger.debug("Invalidated all OHLCV cache")