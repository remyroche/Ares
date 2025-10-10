"""
OHLCV Data Management

Handles OHLCV data fetching, caching, and processing for different timeframes.
Now uses the enhanced OHLCV manager with comprehensive fixes for all identified issues.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger

# Import the enhanced OHLCV manager
from .enhanced_ohlcv_manager import (
    EnhancedOHLCVManager, OHLCVData as EnhancedOHLCVData, Timeframe as EnhancedTimeframe,
    get_enhanced_ohlcv_manager, cleanup_all_managers
)


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
    Now uses the enhanced OHLCV manager with comprehensive fixes.
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"OHLCVManager.{exchange_name}")
        
        # Use the enhanced OHLCV manager
        self.enhanced_manager = get_enhanced_ohlcv_manager(exchange_name)
        
        # Maintain backward compatibility
        self.ohlcv_cache = {}  # Deprecated, use enhanced_manager.cache
        self.cache_ttl = {}    # Deprecated, use enhanced_manager.config.ttl_seconds
        self.fetch_functions = {}  # Deprecated, use enhanced_manager.fetch_functions
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
        # Delegate to enhanced manager
        self.enhanced_manager.register_fetch_functions(get_klines, get_historical_klines)
        
        # Maintain backward compatibility
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
        # Convert to enhanced timeframe
        enhanced_timeframe = self._convert_timeframe(timeframe)
        
        # Delegate to enhanced manager
        enhanced_data = await self.enhanced_manager.get_ohlcv(
            symbol, enhanced_timeframe, limit, use_cache
        )
        
        # Convert back to legacy format for backward compatibility
        return [self._convert_to_legacy_ohlcv(data) for data in enhanced_data]
    
    def _convert_timeframe(self, timeframe: Timeframe) -> EnhancedTimeframe:
        """Convert legacy timeframe to enhanced timeframe."""
        return EnhancedTimeframe(timeframe.value)
    
    def _convert_to_legacy_ohlcv(self, enhanced_data: EnhancedOHLCVData) -> OHLCVData:
        """Convert enhanced OHLCV data to legacy format."""
        return OHLCVData(
            symbol=enhanced_data.symbol,
            timeframe=Timeframe(enhanced_data.timeframe.value),
            timestamp=enhanced_data.timestamp,
            open=enhanced_data.open,
            high=enhanced_data.high,
            low=enhanced_data.low,
            close=enhanced_data.close,
            volume=enhanced_data.volume,
            quote_volume=enhanced_data.quote_volume,
            trades_count=enhanced_data.trades_count,
            taker_buy_volume=enhanced_data.taker_buy_volume,
            taker_buy_quote_volume=enhanced_data.taker_buy_quote_volume,
            metadata=enhanced_data.metadata
        )
    
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
        # Convert to enhanced timeframe
        enhanced_timeframe = self._convert_timeframe(timeframe)
        
        # Delegate to enhanced manager
        enhanced_data = await self.enhanced_manager.get_historical_ohlcv(
            symbol, enhanced_timeframe, start_time, end_time, limit
        )
        
        # Convert back to legacy format for backward compatibility
        return [self._convert_to_legacy_ohlcv(data) for data in enhanced_data]
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get OHLCV cache statistics."""
        return self.enhanced_manager.get_cache_statistics()
    
    def cleanup_stale_cache(self) -> int:
        """Clean up stale cache entries."""
        return self.enhanced_manager.cache.cleanup_expired(3600)  # 1 hour TTL
    
    def invalidate_cache(self, symbol: Optional[str] = None, timeframe: Optional[Timeframe] = None) -> None:
        """Invalidate OHLCV cache."""
        if timeframe:
            enhanced_timeframe = self._convert_timeframe(timeframe)
            self.enhanced_manager.cache.invalidate(symbol, enhanced_timeframe)
        else:
            self.enhanced_manager.cache.invalidate(symbol)
    
    def cleanup(self) -> None:
        """Clean up resources and connections."""
        self.enhanced_manager.cleanup()
    
    # Legacy methods for backward compatibility
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
        enhanced_timeframe = self._convert_timeframe(timeframe)
        enhanced_data = self.enhanced_manager.cache.get(symbol, enhanced_timeframe)
        return [self._convert_to_legacy_ohlcv(data) for data in enhanced_data]