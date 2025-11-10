"""
Enhanced OHLCV Data Management with Comprehensive Fixes

This module provides an enhanced OHLCV data management system that addresses all identified issues:
- Robust timestamp parsing with multiple unit support
- Comprehensive error handling with recovery mechanisms
- Memory leak prevention and management
- Thread-safe cache operations
- Advanced data validation and integrity checks

Uses utilities from src/utils/ and src/hardware/ for optimal performance.
"""

import asyncio
import gc
import logging
import threading
import time
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
import queue
import uuid

import pandas as pd
import numpy as np

# Import utilities
from src.utils.logger import system_logger
from src.utils.tprint import tprint
from src.utils.error_handler import UnifiedErrorHandler, handles_errors, safe_execution
from src.utils.hardware.advanced_memory_optimizer import (
    AdvancedM1MemoryOptimizer, MemoryPoolType, MemoryStrategy, 
    get_advanced_memory_optimizer
)
from src.utils.data.quality.data_quality import DataQualityFramework, QualityResult
from src.utils.numba_timestamps import get_numba_timestamp_string
from src.utils.memory_management.streaming_data_processor import StreamingDataProcessor

logger = system_logger.getChild('EnhancedOHLCVManager')

class Timeframe(Enum):
    """Timeframe enumeration with validation."""
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

class TimestampUnit(Enum):
    """Timestamp unit enumeration."""
    MILLISECONDS = "ms"
    SECONDS = "s"
    MICROSECONDS = "us"
    NANOSECONDS = "ns"

class DataIntegrityLevel(Enum):
    """Data integrity validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CRITICAL = "critical"

@dataclass
class OHLCVData:
    """Enhanced OHLCV data structure with validation."""
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    _validated: bool = False
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self._validated:
            self._validate_integrity()
    
    def _validate_integrity(self) -> None:
        """Validate OHLCV data integrity."""
        if self.high < self.low:
            raise ValueError(f"High price ({self.high}) cannot be less than low price ({self.low})")
        
        if any(price <= 0 for price in [self.open, self.high, self.low, self.close]):
            raise ValueError("All OHLC prices must be positive")
        
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
        
        # Check if close is within high-low range
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"Close price ({self.close}) must be within high-low range [{self.low}, {self.high}]")
        
        # Check if open is within high-low range
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"Open price ({self.open}) must be within high-low range [{self.low}, {self.high}]")
        
        self._validated = True

@dataclass
class CacheConfig:
    """Configuration for OHLCV cache management."""
    max_candles_per_symbol: int = 10000
    max_total_candles: int = 100000
    ttl_seconds: Dict[Timeframe, int] = field(default_factory=lambda: {
        Timeframe.MINUTE_1: 300,    # 5 minutes
        Timeframe.MINUTE_5: 900,    # 15 minutes
        Timeframe.MINUTE_15: 1800,  # 30 minutes
        Timeframe.MINUTE_30: 3600,  # 1 hour
        Timeframe.HOUR_1: 7200,     # 2 hours
        Timeframe.HOUR_4: 14400,    # 4 hours
        Timeframe.DAY_1: 86400,     # 24 hours
    })
    cleanup_interval: int = 300  # 5 minutes
    memory_pressure_threshold: float = 0.8
    enable_compression: bool = True
    thread_safe: bool = True

class TimestampParser:
    """Enhanced timestamp parser with multiple unit support and error recovery."""
    
    def __init__(self, error_handler: UnifiedErrorHandler = None):
        self.error_handler = error_handler or UnifiedErrorHandler()
        self.logger = system_logger.getChild('TimestampParser')
        
        # Common timestamp patterns and their likely units
        self.timestamp_patterns = {
            # Millisecond timestamps (13 digits)
            r'^\d{13}$': TimestampUnit.MILLISECONDS,
            # Second timestamps (10 digits)
            r'^\d{10}$': TimestampUnit.SECONDS,
            # Microsecond timestamps (16 digits)
            r'^\d{16}$': TimestampUnit.MICROSECONDS,
            # Nanosecond timestamps (19 digits)
            r'^\d{19}$': TimestampUnit.NANOSECONDS,
        }
    
    @handles_errors(default_return=None)
    def parse_timestamp(self, timestamp_value: Any,
                       preferred_unit: Optional[TimestampUnit] = None,
                       timezone_info: Optional[timezone] = None) -> Optional[datetime]:
        """
        Parse timestamp with comprehensive unit detection and error recovery.

        Args:
            timestamp_value: Timestamp value (int, float, str, datetime)
            preferred_unit: Preferred timestamp unit if known
            timezone_info: Timezone information for the timestamp

        Returns:
            Parsed datetime object or None if parsing fails
        """
        tprint(f"🔧 parse_timestamp called with timestamp_value={timestamp_value}, preferred_unit={preferred_unit}", "INFO")
        if timestamp_value is None:
            tprint(f"⚠️ parse_timestamp received None timestamp_value", "WARNING")
            return None
        
        # Handle datetime objects
        if isinstance(timestamp_value, datetime):
            tprint(f"✅ parse_timestamp: timestamp already datetime object", "SUCCESS")
            return timestamp_value

        # Handle string timestamps
        if isinstance(timestamp_value, str):
            tprint(f"🔧 parse_timestamp: parsing string timestamp", "INFO")
            return self._parse_string_timestamp(timestamp_value, timezone_info)

        # Handle numeric timestamps
        if isinstance(timestamp_value, (int, float)):
            tprint(f"🔧 parse_timestamp: parsing numeric timestamp", "INFO")
            return self._parse_numeric_timestamp(timestamp_value, preferred_unit, timezone_info)

        # Try to convert to numeric
        try:
            numeric_value = float(timestamp_value)
            tprint(f"🔧 parse_timestamp: converted to numeric", "INFO")
            return self._parse_numeric_timestamp(numeric_value, preferred_unit, timezone_info)
        except (ValueError, TypeError):
            self.logger.warning(f"Unable to parse timestamp: {timestamp_value}")
            tprint(f"❌ parse_timestamp failed: unable to parse {timestamp_value}", "ERROR")
            return None
    
    def _parse_string_timestamp(self, timestamp_str: str,
                               timezone_info: Optional[timezone] = None) -> Optional[datetime]:
        """Parse string timestamp with multiple format support."""
        tprint(f"🔧 _parse_string_timestamp called with timestamp_str={timestamp_str}", "INFO")
        try:
            # Try ISO format first
            if 'T' in timestamp_str or ' ' in timestamp_str:
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                tprint(f"✅ _parse_string_timestamp: parsed ISO format successfully", "SUCCESS")
                return dt if timezone_info is None else dt.astimezone(timezone_info)
            
            # Try common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S.%f%z',
                '%Y-%m-%d %H:%M:%S%z',
                '%Y%m%d%H%M%S',
                '%Y%m%d%H%M%S%f'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    return dt if timezone_info is None else dt.astimezone(timezone_info)
                except ValueError:
                    continue
            
            # Try pandas parsing as last resort
            dt = pd.to_datetime(timestamp_str, utc=True)
            tprint(f"✅ _parse_string_timestamp: parsed with pandas successfully", "SUCCESS")
            return dt if timezone_info is None else dt.astimezone(timezone_info)

        except Exception as e:
            self.logger.warning(f"String timestamp parsing failed: {e}")
            tprint(f"❌ _parse_string_timestamp failed: {e}", "ERROR")
            return None
    
    def _parse_numeric_timestamp(self, timestamp_value: Union[int, float],
                                preferred_unit: Optional[TimestampUnit] = None,
                                timezone_info: Optional[timezone] = None) -> Optional[datetime]:
        """Parse numeric timestamp with automatic unit detection."""
        tprint(f"🔧 _parse_numeric_timestamp called with timestamp_value={timestamp_value}", "INFO")
        try:
            # Determine the unit
            unit = self._detect_timestamp_unit(timestamp_value, preferred_unit)
            tprint(f"🔧 _parse_numeric_timestamp: detected unit={unit.value}", "INFO")
            
            # Convert to seconds
            if unit == TimestampUnit.MILLISECONDS:
                timestamp_seconds = timestamp_value / 1000
            elif unit == TimestampUnit.MICROSECONDS:
                timestamp_seconds = timestamp_value / 1000000
            elif unit == TimestampUnit.NANOSECONDS:
                timestamp_seconds = timestamp_value / 1000000000
            else:  # SECONDS
                timestamp_seconds = timestamp_value
            
            # Create datetime object
            dt = datetime.fromtimestamp(timestamp_seconds, tz=timezone_info or timezone.utc)
            
            # Validate timestamp is reasonable (not too far in past/future)
            now = datetime.now(timezone.utc)
            if dt > now + timedelta(days=365):
                self.logger.warning(f"Timestamp {dt} is more than 1 year in the future")
                tprint(f"⚠️ _parse_numeric_timestamp: timestamp is more than 1 year in future", "WARNING")
            elif dt < datetime(2000, 1, 1, tzinfo=timezone.utc):
                self.logger.warning(f"Timestamp {dt} is before year 2000")
                tprint(f"⚠️ _parse_numeric_timestamp: timestamp is before year 2000", "WARNING")

            tprint(f"✅ _parse_numeric_timestamp: successfully parsed to {dt}", "SUCCESS")
            return dt

        except (ValueError, OSError, OverflowError) as e:
            self.logger.warning(f"Numeric timestamp parsing failed: {e}")
            tprint(f"❌ _parse_numeric_timestamp failed: {e}", "ERROR")
            return None
    
    def _detect_timestamp_unit(self, timestamp_value: Union[int, float],
                              preferred_unit: Optional[TimestampUnit] = None) -> TimestampUnit:
        """Detect timestamp unit based on value magnitude and patterns."""
        if preferred_unit:
            return preferred_unit
        
        # Convert to int for pattern matching
        int_value = int(timestamp_value)
        timestamp_str = str(int_value)
        
        # Check patterns
        for pattern, unit in self.timestamp_patterns.items():
            if pattern.match(timestamp_str):
                return unit
        
        # Fallback based on magnitude
        if timestamp_value > 1e15:  # Nanoseconds
            return TimestampUnit.NANOSECONDS
        elif timestamp_value > 1e12:  # Microseconds
            return TimestampUnit.MICROSECONDS
        elif timestamp_value > 1e9:  # Milliseconds
            return TimestampUnit.MILLISECONDS
        else:  # Seconds
            return TimestampUnit.SECONDS

class DataValidator:
    """Comprehensive OHLCV data validator."""
    
    def __init__(self, integrity_level: DataIntegrityLevel = DataIntegrityLevel.STANDARD):
        self.integrity_level = integrity_level
        self.error_handler = UnifiedErrorHandler()
        self.quality_framework = DataQualityFramework()
        self.logger = system_logger.getChild('DataValidator')
    
    def validate_ohlcv_data(self, data: OHLCVData) -> Tuple[bool, List[str]]:
        """Validate OHLCV data with comprehensive checks."""
        tprint(f"🔧 validate_ohlcv_data called for symbol={data.symbol}, integrity_level={self.integrity_level.value}", "INFO")
        issues = []

        try:
            # Basic validation
            if not self._validate_basic_ohlcv(data, issues):
                return False, issues
            
            # Level-specific validation
            if self.integrity_level in [DataIntegrityLevel.STANDARD, DataIntegrityLevel.STRICT, DataIntegrityLevel.CRITICAL]:
                if not self._validate_standard_ohlcv(data, issues):
                    return False, issues
            
            if self.integrity_level in [DataIntegrityLevel.STRICT, DataIntegrityLevel.CRITICAL]:
                if not self._validate_strict_ohlcv(data, issues):
                    return False, issues
            
            if self.integrity_level == DataIntegrityLevel.CRITICAL:
                if not self._validate_critical_ohlcv(data, issues):
                    return False, issues

            tprint(f"✅ validate_ohlcv_data: validation passed", "SUCCESS")
            return True, issues

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            tprint(f"❌ validate_ohlcv_data failed: {e}", "ERROR")
            issues.append(f"Validation error: {e}")
            return False, issues
    
    def _validate_basic_ohlcv(self, data: OHLCVData, issues: List[str]) -> bool:
        """Basic OHLCV validation."""
        # Check for positive prices
        if any(price <= 0 for price in [data.open, data.high, data.low, data.close]):
            issues.append("All OHLC prices must be positive")
            tprint(f"❌ _validate_basic_ohlcv: negative or zero prices detected", "ERROR")
            return False
        
        # Check high >= low
        if data.high < data.low:
            issues.append(f"High price ({data.high}) cannot be less than low price ({data.low})")
            return False
        
        # Check volume
        if data.volume < 0:
            issues.append("Volume cannot be negative")
            return False
        
        return True
    
    def _validate_standard_ohlcv(self, data: OHLCVData, issues: List[str]) -> bool:
        """Standard OHLCV validation."""
        # Check close is within high-low range
        if not (data.low <= data.close <= data.high):
            issues.append(f"Close price ({data.close}) must be within high-low range [{data.low}, {data.high}]")
            return False
        
        # Check open is within high-low range
        if not (data.low <= data.open <= data.high):
            issues.append(f"Open price ({data.open}) must be within high-low range [{data.low}, {data.high}]")
            return False
        
        # Check for reasonable price movements
        price_range = data.high - data.low
        if price_range > data.close * 0.5:  # More than 50% price movement
            issues.append(f"Unusually large price range: {price_range:.4f} (close: {data.close:.4f})")
        
        return True
    
    def _validate_strict_ohlcv(self, data: OHLCVData, issues: List[str]) -> bool:
        """Strict OHLCV validation."""
        # Check for reasonable volume
        if data.volume == 0:
            issues.append("Volume cannot be zero")
            return False
        
        # Check quote volume consistency
        if data.quote_volume is not None and data.quote_volume <= 0:
            issues.append("Quote volume must be positive if provided")
            return False
        
        # Check trades count consistency
        if data.trades_count is not None and data.trades_count < 0:
            issues.append("Trades count must be non-negative if provided")
            return False
        
        return True
    
    def _validate_critical_ohlcv(self, data: OHLCVData, issues: List[str]) -> bool:
        """Critical OHLCV validation for high-stakes applications."""
        # Check for extreme price movements (more than 20% in a single candle)
        price_change = abs(data.close - data.open) / data.open
        if price_change > 0.2:
            issues.append(f"Extreme price movement: {price_change:.2%}")
        
        # Check for suspicious volume patterns
        if data.volume > 0 and data.quote_volume is not None:
            avg_price = data.quote_volume / data.volume
            price_deviation = abs(avg_price - data.close) / data.close
            if price_deviation > 0.1:  # More than 10% deviation
                issues.append(f"Volume-price inconsistency: avg_price={avg_price:.4f}, close={data.close:.4f}")
        
        return True

class ThreadSafeCache:
    """Thread-safe cache implementation for OHLCV data."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, Dict[Timeframe, List[OHLCVData]]] = {}
        self._access_times: Dict[str, Dict[Timeframe, float]] = {}
        self._lock = threading.RLock()
        self._cleanup_lock = threading.Lock()
        self.logger = system_logger.getChild('ThreadSafeCache')
    
    def get(self, symbol: str, timeframe: Timeframe, limit: int = None) -> List[OHLCVData]:
        """Get cached data for symbol and timeframe."""
        tprint(f"🔧 ThreadSafeCache.get called with symbol={symbol}, timeframe={timeframe.value}, limit={limit}", "INFO")
        with self._lock:
            if symbol not in self._cache or timeframe not in self._cache[symbol]:
                tprint(f"⚠️ ThreadSafeCache.get: cache miss for {symbol}", "WARNING")
                return []
            
            data = self._cache[symbol][timeframe].copy()
            self._update_access_time(symbol, timeframe)

            if limit and len(data) > limit:
                tprint(f"✅ ThreadSafeCache.get: returning {limit} candles (limited from {len(data)})", "SUCCESS")
                return data[-limit:]

            tprint(f"✅ ThreadSafeCache.get: returning {len(data)} candles", "SUCCESS")
            return data
    
    def put(self, symbol: str, timeframe: Timeframe, data: List[OHLCVData]) -> None:
        """Put data into cache."""
        tprint(f"🔧 ThreadSafeCache.put called with symbol={symbol}, timeframe={timeframe.value}, data_count={len(data)}", "INFO")
        with self._lock:
            if symbol not in self._cache:
                self._cache[symbol] = {}
                self._access_times[symbol] = {}
            
            # Store data
            self._cache[symbol][timeframe] = data.copy()
            self._update_access_time(symbol, timeframe)
            tprint(f"✅ ThreadSafeCache.put: stored {len(data)} candles for {symbol}", "SUCCESS")

            # Enforce size limits
            self._enforce_size_limits()
    
    def append(self, symbol: str, timeframe: Timeframe, data: List[OHLCVData]) -> None:
        """Append data to existing cache."""
        with self._lock:
            if symbol not in self._cache:
                self._cache[symbol] = {}
                self._access_times[symbol] = {}
            
            if timeframe not in self._cache[symbol]:
                self._cache[symbol][timeframe] = []
            
            self._cache[symbol][timeframe].extend(data)
            self._update_access_time(symbol, timeframe)
            
            # Enforce size limits
            self._enforce_size_limits()
    
    def invalidate(self, symbol: str = None, timeframe: Timeframe = None) -> None:
        """Invalidate cache entries."""
        with self._lock:
            if symbol is None:
                self._cache.clear()
                self._access_times.clear()
            elif timeframe is None:
                self._cache.pop(symbol, None)
                self._access_times.pop(symbol, None)
            else:
                if symbol in self._cache and timeframe in self._cache[symbol]:
                    del self._cache[symbol][timeframe]
                    del self._access_times[symbol][timeframe]
    
    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Clean up expired cache entries."""
        tprint(f"🔧 ThreadSafeCache.cleanup_expired called with ttl_seconds={ttl_seconds}", "INFO")
        with self._cleanup_lock:
            current_time = time.time()
            cleaned_count = 0
            
            with self._lock:
                for symbol in list(self._cache.keys()):
                    for timeframe in list(self._cache[symbol].keys()):
                        if (symbol in self._access_times and 
                            timeframe in self._access_times[symbol]):
                            
                            last_access = self._access_times[symbol][timeframe]
                            if current_time - last_access > ttl_seconds:
                                del self._cache[symbol][timeframe]
                                del self._access_times[symbol][timeframe]
                                cleaned_count += 1
                    
                    # Remove empty symbol entries
                    if not self._cache[symbol]:
                        del self._cache[symbol]
                        del self._access_times[symbol]
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} expired cache entries")
                tprint(f"✅ ThreadSafeCache.cleanup_expired: cleaned {cleaned_count} entries", "SUCCESS")

            return cleaned_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_symbols = len(self._cache)
            total_candles = sum(
                len(timeframe_data)
                for symbol_data in self._cache.values()
                for timeframe_data in symbol_data.values()
            )
            
            return {
                'total_symbols': total_symbols,
                'total_candles': total_candles,
                'max_size': self.max_size,
                'utilization': total_candles / self.max_size if self.max_size > 0 else 0
            }
    
    def _update_access_time(self, symbol: str, timeframe: Timeframe) -> None:
        """Update access time for cache entry."""
        current_time = time.time()
        if symbol not in self._access_times:
            self._access_times[symbol] = {}
        self._access_times[symbol][timeframe] = current_time
    
    def _enforce_size_limits(self) -> None:
        """Enforce cache size limits."""
        total_candles = sum(
            len(timeframe_data)
            for symbol_data in self._cache.values()
            for timeframe_data in symbol_data.values()
        )
        
        if total_candles > self.max_size:
            # Remove oldest entries
            self._remove_oldest_entries(total_candles - self.max_size)
    
    def _remove_oldest_entries(self, count: int) -> None:
        """Remove oldest cache entries."""
        # Get all entries with their access times
        entries = []
        for symbol, timeframes in self._access_times.items():
            for timeframe, access_time in timeframes.items():
                entries.append((symbol, timeframe, access_time))
        
        # Sort by access time (oldest first)
        entries.sort(key=lambda x: x[2])
        
        # Remove oldest entries
        for symbol, timeframe, _ in entries[:count]:
            if symbol in self._cache and timeframe in self._cache[symbol]:
                del self._cache[symbol][timeframe]
                del self._access_times[symbol][timeframe]

class EnhancedOHLCVManager:
    """
    Enhanced OHLCV data manager with comprehensive fixes for all identified issues.
    """
    
    def __init__(self, exchange_name: str, config: Optional[CacheConfig] = None):
        tprint(f"🔧 EnhancedOHLCVManager.__init__ called with exchange_name={exchange_name}", "INFO")
        self.exchange_name = exchange_name
        self.config = config or CacheConfig()
        self.logger = system_logger.getChild(f'EnhancedOHLCVManager.{exchange_name}')
        
        # Initialize components
        self.error_handler = UnifiedErrorHandler(self.logger)
        self.timestamp_parser = TimestampParser(self.error_handler)
        self.data_validator = DataValidator()
        self.cache = ThreadSafeCache(self.config.max_total_candles)
        
        # Memory management
        self.memory_optimizer = get_advanced_memory_optimizer()
        self.streaming_processor = StreamingDataProcessor()
        
        # Data fetching functions
        self.fetch_functions: Dict[str, Callable] = {}
        
        # Cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0,
            'timestamp_parse_errors': 0,
            'memory_cleanups': 0
        }
        
        self.logger.info(f"🚀 Enhanced OHLCV Manager initialized for {exchange_name}")
        tprint(f"✅ EnhancedOHLCVManager initialized successfully for {exchange_name}", "SUCCESS")
    
    def __del__(self):
        """Cleanup on destruction."""
        self._stop_cleanup_thread()
    
    def register_fetch_functions(self, get_klines: Callable,
                                get_historical_klines: Optional[Callable] = None) -> None:
        """Register exchange-specific OHLCV fetching functions."""
        tprint(f"🔧 register_fetch_functions called for {self.exchange_name}", "INFO")
        self.fetch_functions = {
            "get_klines": get_klines,
            "get_historical_klines": get_historical_klines
        }
        self.logger.info("Registered OHLCV fetching functions")
        tprint(f"✅ register_fetch_functions: registered fetch functions successfully", "SUCCESS")
    
    @safe_execution(default_return=[])
    async def get_ohlcv(self, symbol: str, timeframe: Timeframe,
                       limit: int = 100, use_cache: bool = True) -> List[OHLCVData]:
        """Get OHLCV data with comprehensive error handling and caching."""
        tprint(f"🔧 get_ohlcv called with symbol={symbol}, timeframe={timeframe.value}, limit={limit}, use_cache={use_cache}", "INFO")
        self.stats['total_requests'] += 1

        try:
            # Check cache first
            if use_cache:
                cached_data = self.cache.get(symbol, timeframe, limit)
                if cached_data:
                    self.stats['cache_hits'] += 1
                    tprint(f"✅ get_ohlcv: cache hit, returning {len(cached_data)} candles", "SUCCESS")
                    return cached_data
            
            self.stats['cache_misses'] += 1
            tprint(f"⚠️ get_ohlcv: cache miss, fetching fresh data", "WARNING")

            # Fetch fresh data
            if "get_klines" not in self.fetch_functions:
                self.logger.warning("No klines fetch function registered")
                tprint(f"❌ get_ohlcv: no klines fetch function registered", "ERROR")
                return []
            
            raw_data = await self.fetch_functions["get_klines"](symbol, timeframe.value, limit)
            if not raw_data:
                self.logger.warning(f"No OHLCV data received for {symbol}")
                tprint(f"⚠️ get_ohlcv: no data received from exchange for {symbol}", "WARNING")
                return []

            tprint(f"✅ get_ohlcv: received {len(raw_data)} raw candles from exchange", "SUCCESS")

            # Parse OHLCV data with enhanced error handling
            ohlcv_data = self._parse_ohlcv_data_enhanced(symbol, timeframe, raw_data)

            # Cache the data
            if use_cache and ohlcv_data:
                self.cache.put(symbol, timeframe, ohlcv_data)

            tprint(f"✅ get_ohlcv: returning {len(ohlcv_data)} parsed candles", "SUCCESS")
            return ohlcv_data[:limit] if limit else ohlcv_data

        except Exception as e:
            self.error_handler.handle_error(e, f"get_ohlcv({symbol}, {timeframe.value})", reraise=False)
            tprint(f"❌ get_ohlcv failed: {e}", "ERROR")
            return []
    
    @safe_execution(default_return=[])
    async def get_historical_ohlcv(self, symbol: str, timeframe: Timeframe,
                                  start_time: datetime, end_time: datetime,
                                  limit: int = 1000) -> List[OHLCVData]:
        """Get historical OHLCV data with enhanced error handling."""
        tprint(f"🔧 get_historical_ohlcv called with symbol={symbol}, timeframe={timeframe.value}, start_time={start_time}, end_time={end_time}, limit={limit}", "INFO")
        try:
            if "get_historical_klines" not in self.fetch_functions:
                self.logger.warning("No historical klines fetch function registered")
                tprint(f"❌ get_historical_ohlcv: no historical klines fetch function registered", "ERROR")
                return []
            
            raw_data = await self.fetch_functions["get_historical_klines"](
                symbol, timeframe.value, start_time, end_time, limit
            )
            if not raw_data:
                self.logger.warning(f"No historical OHLCV data received for {symbol}")
                tprint(f"⚠️ get_historical_ohlcv: no historical data received for {symbol}", "WARNING")
                return []

            tprint(f"✅ get_historical_ohlcv: received {len(raw_data)} raw historical candles", "SUCCESS")

            # Parse OHLCV data
            ohlcv_data = self._parse_ohlcv_data_enhanced(symbol, timeframe, raw_data)
            tprint(f"✅ get_historical_ohlcv: returning {len(ohlcv_data)} parsed candles", "SUCCESS")
            return ohlcv_data

        except Exception as e:
            self.error_handler.handle_error(e, f"get_historical_ohlcv({symbol}, {timeframe.value})", reraise=False)
            tprint(f"❌ get_historical_ohlcv failed: {e}", "ERROR")
            return []
    
    def _parse_ohlcv_data_enhanced(self, symbol: str, timeframe: Timeframe,
                                  raw_data: List[List[Any]]) -> List[OHLCVData]:
        """Parse raw OHLCV data with comprehensive error handling and validation."""
        tprint(f"🔧 _parse_ohlcv_data_enhanced called with symbol={symbol}, timeframe={timeframe.value}, raw_data_count={len(raw_data)}", "INFO")
        ohlcv_list = []
        parse_errors = 0
        
        for i, item in enumerate(raw_data):
            try:
                # Parse timestamp with enhanced error handling
                timestamp = self._parse_timestamp_enhanced(item[0])
                if timestamp is None:
                    parse_errors += 1
                    continue
                
                # Parse OHLCV values with validation
                ohlcv_values = self._parse_ohlcv_values(item)
                if ohlcv_values is None:
                    parse_errors += 1
                    continue
                
                # Create OHLCVData object with validation
                ohlcv_data = OHLCVData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    **ohlcv_values,
                    metadata={"raw_data": item, "parse_index": i}
                )
                
                # Additional validation
                is_valid, issues = self.data_validator.validate_ohlcv_data(ohlcv_data)
                if not is_valid:
                    self.logger.warning(f"OHLCV validation failed for item {i}: {issues}")
                    self.stats['validation_errors'] += 1
                    continue
                
                ohlcv_list.append(ohlcv_data)
                
            except Exception as e:
                self.logger.warning(f"Error parsing OHLCV item {i}: {e}")
                parse_errors += 1
                continue
        
        if parse_errors > 0:
            self.stats['timestamp_parse_errors'] += parse_errors
            self.logger.warning(f"Failed to parse {parse_errors} out of {len(raw_data)} OHLCV items")
            tprint(f"⚠️ _parse_ohlcv_data_enhanced: {parse_errors} parse errors out of {len(raw_data)} items", "WARNING")

        tprint(f"✅ _parse_ohlcv_data_enhanced: successfully parsed {len(ohlcv_list)} candles", "SUCCESS")
        return ohlcv_list
    
    def _parse_timestamp_enhanced(self, timestamp_value: Any) -> Optional[datetime]:
        """Parse timestamp with enhanced error handling and recovery."""
        try:
            return self.timestamp_parser.parse_timestamp(timestamp_value)
        except Exception as e:
            self.logger.warning(f"Timestamp parsing failed: {e}")
            return None
    
    def _parse_ohlcv_values(self, item: List[Any]) -> Optional[Dict[str, Any]]:
        """Parse OHLCV values with validation."""
        try:
            if len(item) < 6:
                self.logger.warning(f"Insufficient data in OHLCV item: {len(item)} elements")
                return None
            
            # Parse required OHLCV values
            open_price = float(item[1])
            high_price = float(item[2])
            low_price = float(item[3])
            close_price = float(item[4])
            volume = float(item[5])
            
            # Validate basic values
            if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                self.logger.warning("Invalid price values in OHLCV item")
                return None
            
            if volume < 0:
                self.logger.warning("Invalid volume value in OHLCV item")
                return None
            
            result = {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
            # Parse optional values
            if len(item) > 6 and item[6] is not None:
                result['quote_volume'] = float(item[6])
            
            if len(item) > 7 and item[7] is not None:
                result['trades_count'] = int(item[7])
            
            if len(item) > 8 and item[8] is not None:
                result['taker_buy_volume'] = float(item[8])
            
            if len(item) > 9 and item[9] is not None:
                result['taker_buy_quote_volume'] = float(item[9])
            
            return result
            
        except (ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"Error parsing OHLCV values: {e}")
            return None
    
    def _start_cleanup_thread(self) -> None:
        """Start the cleanup thread for cache management."""
        if self.config.thread_safe:
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True
            )
            self._cleanup_thread.start()
            self.logger.info("Started cache cleanup thread")
    
    def _stop_cleanup_thread(self) -> None:
        """Stop the cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)
    
    def _cleanup_loop(self) -> None:
        """Cleanup loop for cache management."""
        while not self._stop_cleanup.is_set():
            try:
                # Clean up expired cache entries
                for timeframe, ttl_seconds in self.config.ttl_seconds.items():
                    cleaned = self.cache.cleanup_expired(ttl_seconds)
                    if cleaned > 0:
                        self.stats['memory_cleanups'] += 1
                
                # Check memory pressure
                if self.memory_optimizer.is_memory_pressure():
                    self._handle_memory_pressure()
                
                # Sleep for cleanup interval
                self._stop_cleanup.wait(self.config.cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                self._stop_cleanup.wait(60)  # Wait 1 minute on error
    
    def _handle_memory_pressure(self) -> None:
        """Handle memory pressure by cleaning up cache and triggering GC."""
        self.logger.warning("Memory pressure detected, performing cleanup")
        tprint(f"⚠️ _handle_memory_pressure: memory pressure detected, performing cleanup", "WARNING")
        
        # Clean up cache
        self.cache.invalidate()
        
        # Trigger garbage collection
        gc.collect()
        
        # Apply memory optimizations
        self.memory_optimizer.optimize_memory()

        self.stats['memory_cleanups'] += 1
        tprint(f"✅ _handle_memory_pressure: memory cleanup completed", "SUCCESS")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            **cache_stats,
            'performance_stats': self.stats.copy(),
            'memory_usage': self.memory_optimizer.get_memory_stats(),
            'config': {
                'max_candles_per_symbol': self.config.max_candles_per_symbol,
                'max_total_candles': self.config.max_total_candles,
                'thread_safe': self.config.thread_safe
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources and connections."""
        tprint(f"🔧 cleanup called for {self.exchange_name}", "INFO")
        self._stop_cleanup_thread()
        self.cache.invalidate()
        self.memory_optimizer.cleanup()
        self.logger.info("Enhanced OHLCV Manager cleaned up")
        tprint(f"✅ cleanup: Enhanced OHLCV Manager cleaned up successfully", "SUCCESS")

# Global instance management
_enhanced_managers: Dict[str, EnhancedOHLCVManager] = {}

def get_enhanced_ohlcv_manager(exchange_name: str,
                              config: Optional[CacheConfig] = None) -> EnhancedOHLCVManager:
    """Get or create an enhanced OHLCV manager for the exchange."""
    tprint(f"🔧 get_enhanced_ohlcv_manager called for exchange_name={exchange_name}", "INFO")
    if exchange_name not in _enhanced_managers:
        tprint(f"🔧 get_enhanced_ohlcv_manager: creating new manager for {exchange_name}", "INFO")
        _enhanced_managers[exchange_name] = EnhancedOHLCVManager(exchange_name, config)
    else:
        tprint(f"✅ get_enhanced_ohlcv_manager: returning existing manager for {exchange_name}", "SUCCESS")
    return _enhanced_managers[exchange_name]

def cleanup_all_managers() -> None:
    """Clean up all enhanced OHLCV managers."""
    tprint(f"🔧 cleanup_all_managers called for {len(_enhanced_managers)} managers", "INFO")
    for manager in _enhanced_managers.values():
        manager.cleanup()
    _enhanced_managers.clear()
    tprint(f"✅ cleanup_all_managers: all managers cleaned up", "SUCCESS")

# Export main classes and functions
__all__ = [
    'EnhancedOHLCVManager',
    'OHLCVData',
    'Timeframe',
    'TimestampUnit',
    'DataIntegrityLevel',
    'CacheConfig',
    'TimestampParser',
    'DataValidator',
    'ThreadSafeCache',
    'get_enhanced_ohlcv_manager',
    'cleanup_all_managers'
]