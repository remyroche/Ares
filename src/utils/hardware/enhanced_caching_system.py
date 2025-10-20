"""
Enhanced Caching System with LRU, Data Type Optimization, and Memory Efficiency.

This module provides a comprehensive caching system that automatically optimizes
data types, implements LRU eviction, and provides memory-efficient storage
with compression and intelligent eviction policies.
"""

import logging
import time
import threading
import hashlib
import pickle
import zlib
import gc
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, deque
from functools import wraps, lru_cache
import weakref
from pathlib import Path
import json

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class DummyModule:
        def __getattr__(self, name):
            return None
    np = DummyModule()

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    class DummyModule:
        def __getattr__(self, name):
            return None
    pd = DummyModule()

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_performance, tprint_timer, LogLevel
)
from .advanced_memory_manager import (
    get_advanced_memory_manager, MemoryConfig as AdvancedMemoryConfig,
    memory_efficient_processing, chunked_processing, track_memory_usage
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    SIZE_BASED = "size_based"      # Based on object size
    ADAPTIVE = "adaptive"          # Adaptive based on usage patterns

class DataTypeOptimization(Enum):
    """Data type optimization levels."""
    NONE = "none"                  # No optimization
    BASIC = "basic"                # Basic int32/float32 optimization
    AGGRESSIVE = "aggressive"      # Aggressive optimization with compression
    MAXIMUM = "maximum"            # Maximum optimization with all techniques

class CompressionType(Enum):
    """Compression types for cached data."""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    GZIP = "gzip"
    PICKLE = "pickle"

@dataclass
class CacheConfig:
    """Configuration for the caching system."""
    # Cache size limits
    max_memory_mb: float = 512.0
    max_items: int = 10000
    max_item_size_mb: float = 50.0
    
    # Eviction strategy
    strategy: CacheStrategy = CacheStrategy.LRU
    ttl_seconds: float = 3600.0  # 1 hour default TTL
    
    # Data type optimization
    data_type_optimization: DataTypeOptimization = DataTypeOptimization.AGGRESSIVE
    auto_optimize_dtypes: bool = True
    prefer_int32: bool = True
    prefer_float32: bool = True
    
    # Compression
    enable_compression: bool = True
    compression_type: CompressionType = CompressionType.ZLIB
    compression_threshold_mb: float = 1.0  # Compress items larger than 1MB
    
    # Memory management
    enable_memory_monitoring: bool = True
    memory_check_interval: float = 5.0
    aggressive_cleanup_threshold: float = 0.9
    
    # Performance
    enable_statistics: bool = True
    enable_hit_rate_tracking: bool = True
    statistics_retention_hours: int = 24

@dataclass
class CacheItem:
    """Represents a cached item with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    compression_ratio: float = 1.0
    data_type_optimized: bool = False
    ttl: Optional[float] = None

@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    compressions: int = 0
    data_type_optimizations: int = 0
    total_memory_used_mb: float = 0.0
    peak_memory_used_mb: float = 0.0
    average_item_size_mb: float = 0.0
    hit_rate: float = 0.0
    compression_ratio: float = 1.0

class DataTypeOptimizer:
    """Optimizes data types for memory efficiency."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logger.getChild('DataTypeOptimizer')
        
        # Type optimization mappings
        self.int_optimizations = {
            np.int64: np.int32,
            np.int32: np.int16 if self.config.prefer_int32 else np.int32,
            np.int16: np.int8
        }
        
        self.float_optimizations = {
            np.float64: np.float32,
            np.complex128: np.complex64
        }
        
        # Pandas type optimizations
        self.pandas_optimizations = {
            'int64': 'int32',
            'float64': 'float32',
            'object': 'category'  # For repeated strings
        }
    
    def optimize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Optimize DataFrame data types for memory efficiency."""
        if not self.config.auto_optimize_dtypes:
            return df, {}
        
        original_memory = df.memory_usage(deep=True).sum()
        optimization_info = {
            'original_memory_mb': original_memory / (1024 * 1024),
            'optimizations_applied': [],
            'memory_saved_mb': 0.0
        }
        
        try:
            optimized_df = df.copy()
            
            # Optimize numeric columns
            for col in optimized_df.select_dtypes(include=[np.number]).columns:
                col_data = optimized_df[col]
                
                # Skip if already optimized
                if col_data.dtype in [np.int32, np.int16, np.int8, np.float32]:
                    continue
                
                # Integer optimization
                if col_data.dtype in [np.int64, np.int32]:
                    if self._can_downcast_int(col_data):
                        if self.config.prefer_int32 and col_data.dtype == np.int64:
                            optimized_df[col] = col_data.astype(np.int32)
                            optimization_info['optimizations_applied'].append(f'{col}: int64->int32')
                        elif col_data.dtype == np.int32 and col_data.max() <= 32767 and col_data.min() >= -32768:
                            optimized_df[col] = col_data.astype(np.int16)
                            optimization_info['optimizations_applied'].append(f'{col}: int32->int16')
                
                # Float optimization
                elif col_data.dtype == np.float64:
                    if self.config.prefer_float32 and self._can_downcast_float(col_data):
                        optimized_df[col] = col_data.astype(np.float32)
                        optimization_info['optimizations_applied'].append(f'{col}: float64->float32')
            
            # Optimize categorical columns
            if self.config.data_type_optimization == DataTypeOptimization.AGGRESSIVE:
                for col in optimized_df.select_dtypes(include=['object']).columns:
                    if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # Less than 50% unique values
                        optimized_df[col] = optimized_df[col].astype('category')
                        optimization_info['optimizations_applied'].append(f'{col}: object->category')
            
            # Calculate memory savings
            optimized_memory = optimized_df.memory_usage(deep=True).sum()
            optimization_info['memory_saved_mb'] = (original_memory - optimized_memory) / (1024 * 1024)
            optimization_info['final_memory_mb'] = optimized_memory / (1024 * 1024)
            
            if optimization_info['memory_saved_mb'] > 0:
                tprint_debug(f"DataFrame optimized: {optimization_info['memory_saved_mb']:.2f}MB saved")
            
            return optimized_df, optimization_info
            
        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}")
            return df, {'error': str(e)}
    
    def optimize_numpy_array(self, arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize NumPy array data types."""
        if not self.config.auto_optimize_dtypes:
            return arr, {}
        
        original_dtype = arr.dtype
        original_size = arr.nbytes
        
        try:
            optimized_arr = arr.copy()
            
            # Integer optimization
            if arr.dtype in self.int_optimizations:
                new_dtype = self.int_optimizations[arr.dtype]
                if self._can_downcast_int(arr):
                    optimized_arr = arr.astype(new_dtype)
            
            # Float optimization
            elif arr.dtype in self.float_optimizations:
                new_dtype = self.float_optimizations[arr.dtype]
                if self._can_downcast_float(arr):
                    optimized_arr = arr.astype(new_dtype)
            
            # Complex optimization
            elif arr.dtype == np.complex128:
                if self._can_downcast_complex(arr):
                    optimized_arr = arr.astype(np.complex64)
            
            optimized_size = optimized_arr.nbytes
            memory_saved = (original_size - optimized_size) / (1024 * 1024)
            
            return optimized_arr, {
                'original_dtype': str(original_dtype),
                'optimized_dtype': str(optimized_arr.dtype),
                'memory_saved_mb': memory_saved,
                'optimized': original_dtype != optimized_arr.dtype
            }
            
        except Exception as e:
            self.logger.warning(f"NumPy array optimization failed: {e}")
            return arr, {'error': str(e)}
    
    def _can_downcast_int(self, arr: np.ndarray) -> bool:
        """Check if integer array can be downcast safely."""
        if arr.dtype == np.int64:
            return arr.max() <= 2147483647 and arr.min() >= -2147483648
        elif arr.dtype == np.int32:
            return arr.max() <= 32767 and arr.min() >= -32768
        return False
    
    def _can_downcast_float(self, arr: np.ndarray) -> bool:
        """Check if float array can be downcast safely."""
        if arr.dtype == np.float64:
            # Check if values fit in float32 range
            return np.isfinite(arr).all() and np.abs(arr).max() <= 3.4e38
        return False
    
    def _can_downcast_complex(self, arr: np.ndarray) -> bool:
        """Check if complex array can be downcast safely."""
        if arr.dtype == np.complex128:
            real_part = arr.real
            imag_part = arr.imag
            return (np.isfinite(real_part).all() and np.isfinite(imag_part).all() and
                    np.abs(real_part).max() <= 3.4e38 and np.abs(imag_part).max() <= 3.4e38)
        return False

class CompressionManager:
    """Manages data compression for cached items."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logger.getChild('CompressionManager')
        
        # Try to import compression libraries
        self.lz4_available = False
        try:
            import lz4.frame
            self.lz4 = lz4.frame
            self.lz4_available = True
        except ImportError:
            self.lz4 = None
    
    def compress_data(self, data: Any) -> Tuple[bytes, float]:
        """Compress data and return compressed bytes with compression ratio."""
        try:
            # Serialize data first
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            original_size = len(serialized)
            
            if original_size < self.config.compression_threshold_mb * 1024 * 1024:
                return serialized, 1.0  # No compression for small items
            
            # Choose compression method
            if self.config.compression_type == CompressionType.ZLIB:
                compressed = zlib.compress(serialized, level=6)
            elif self.config.compression_type == CompressionType.LZ4 and self.lz4_available:
                compressed = self.lz4.compress(serialized)
            elif self.config.compression_type == CompressionType.GZIP:
                import gzip
                compressed = gzip.compress(serialized)
            else:
                return serialized, 1.0
            
            compression_ratio = len(compressed) / original_size if original_size > 0 else 1.0
            
            return compressed, compression_ratio
            
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return pickle.dumps(data), 1.0
    
    def decompress_data(self, compressed_data: bytes, compression_type: CompressionType) -> Any:
        """Decompress data."""
        try:
            if compression_type == CompressionType.NONE:
                return pickle.loads(compressed_data)
            elif compression_type == CompressionType.ZLIB:
                decompressed = zlib.decompress(compressed_data)
            elif compression_type == CompressionType.LZ4 and self.lz4_available:
                decompressed = self.lz4.decompress(compressed_data)
            elif compression_type == CompressionType.GZIP:
                import gzip
                decompressed = gzip.decompress(compressed_data)
            else:
                decompressed = compressed_data
            
            return pickle.loads(decompressed)
            
        except Exception as e:
            self.logger.warning(f"Decompression failed: {e}")
            return pickle.loads(compressed_data)

class EnhancedCacheSystem:
    """Enhanced caching system with LRU, data type optimization, and memory efficiency."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.logger = logger.getChild('EnhancedCacheSystem')
        
        # Cache storage
        self._cache: OrderedDict[str, CacheItem] = OrderedDict()
        self._lock = threading.RLock()
        
        # Optimization components
        self.data_type_optimizer = DataTypeOptimizer(self.config)
        self.compression_manager = CompressionManager(self.config)
        
        # Advanced memory management
        memory_config = AdvancedMemoryConfig(
            enable_aggressive_gc=True,
            gc_threshold_mb=self.config.max_memory_mb * 0.8,
            enable_memory_pressure_detection=True,
            enable_chunking=True,
            default_chunk_size_mb=self.config.max_memory_mb * 0.1,
            enable_memory_pools=True,
            pool_size_mb=self.config.max_memory_mb * 0.2
        )
        self.memory_manager = get_advanced_memory_manager(memory_config)
        
        # Statistics
        self.statistics = CacheStatistics()
        self._access_history = deque(maxlen=1000)  # For LFU strategy
        
        # Memory monitoring
        self._memory_monitor_thread = None
        self._monitoring_active = False
        
        # Start monitoring if enabled
        if self.config.enable_memory_monitoring:
            self._start_memory_monitoring()
        
        tprint_success("✅ Enhanced Cache System initialized")
        self.logger.info(f"Cache initialized: max_memory={self.config.max_memory_mb}MB, "
                        f"strategy={self.config.strategy.value}")
    
    def _start_memory_monitoring(self):
        """Start memory monitoring thread."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._memory_monitor_thread = threading.Thread(
            target=self._memory_monitoring_loop,
            daemon=True,
            name="CacheMemoryMonitor"
        )
        self._memory_monitor_thread.start()
        self.logger.debug("Memory monitoring started")
    
    def _memory_monitoring_loop(self):
        """Memory monitoring loop."""
        while self._monitoring_active:
            try:
                self._check_memory_pressure()
                time.sleep(self.config.memory_check_interval)
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(10)
    
    def _check_memory_pressure(self):
        """Check memory pressure and trigger cleanup if needed."""
        try:
            # Use advanced memory manager for pressure detection
            memory_stats = self.memory_manager.get_memory_stats()
            current_memory = self._get_current_memory_usage()
            memory_ratio = current_memory / (self.config.max_memory_mb * 1024 * 1024)
            
            # Check both cache memory and system memory pressure
            if (memory_ratio > self.config.aggressive_cleanup_threshold or 
                memory_stats.pressure_level.value in ['high', 'critical']):
                tprint_warning(f"High memory usage: {memory_ratio:.1%}, system pressure: {memory_stats.pressure_level.value}")
                self._aggressive_cleanup()
            elif memory_ratio > 0.8 or memory_stats.pressure_level.value == 'medium':
                self._evict_items(0.1)  # Evict 10% of items
                
        except Exception as e:
            self.logger.error(f"Memory pressure check failed: {e}")
    
    def _get_current_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        total_size = 0
        for item in self._cache.values():
            total_size += item.size_bytes
        return total_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self.statistics.misses += 1
                return None
            
            item = self._cache[key]
            
            # Check TTL
            if item.ttl and time.time() > item.created_at + item.ttl:
                del self._cache[key]
                self.statistics.misses += 1
                return None
            
            # Update access information
            item.last_accessed = time.time()
            item.access_count += 1
            self._access_history.append(key)
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            
            # Decompress if needed
            if item.compressed:
                try:
                    value = self.compression_manager.decompress_data(
                        item.value, self.config.compression_type
                    )
                except Exception as e:
                    self.logger.error(f"Decompression failed for key {key}: {e}")
                    del self._cache[key]
                    self.statistics.misses += 1
                    return None
            else:
                value = item.value
            
            self.statistics.hits += 1
            self._update_hit_rate()
            
            return value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put item in cache with optimization."""
        with self._lock:
            try:
                # Optimize data types
                optimized_value, optimization_info = self._optimize_value(value)
                
                # Calculate size
                size_bytes = self._calculate_size(optimized_value)
                
                # Check size limits
                if size_bytes > self.config.max_item_size_mb * 1024 * 1024:
                    tprint_warning(f"Item {key} too large: {size_bytes / (1024*1024):.2f}MB")
                    return False
                
                # Compress if needed
                if (self.config.enable_compression and 
                    size_bytes > self.config.compression_threshold_mb * 1024 * 1024):
                    compressed_value, compression_ratio = self.compression_manager.compress_data(optimized_value)
                    final_value = compressed_value
                    compressed = True
                    self.statistics.compressions += 1
                else:
                    final_value = optimized_value
                    compressed = False
                    compression_ratio = 1.0
                
                # Create cache item
                item = CacheItem(
                    key=key,
                    value=final_value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    size_bytes=len(final_value) if isinstance(final_value, bytes) else size_bytes,
                    compressed=compressed,
                    compression_ratio=compression_ratio,
                    data_type_optimized=optimization_info.get('optimized', False),
                    ttl=ttl or self.config.ttl_seconds
                )
                
                # Remove existing item if present
                if key in self._cache:
                    del self._cache[key]
                
                # Add new item
                self._cache[key] = item
                
                # Evict if needed
                self._ensure_memory_limits()
                
                # Update statistics
                if optimization_info.get('optimized', False):
                    self.statistics.data_type_optimizations += 1
                
                self._update_statistics()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache item {key}: {e}")
                return False
    
    def _optimize_value(self, value: Any) -> Tuple[Any, Dict[str, Any]]:
        """Optimize value data types."""
        if isinstance(value, pd.DataFrame):
            return self.data_type_optimizer.optimize_dataframe(value)
        elif isinstance(value, np.ndarray):
            return self.data_type_optimizer.optimize_numpy_array(value)
        elif isinstance(value, dict):
            # Recursively optimize dictionary values
            optimized_dict = {}
            optimization_info = {'optimized': False, 'items_optimized': 0}
            
            for k, v in value.items():
                opt_v, opt_info = self._optimize_value(v)
                optimized_dict[k] = opt_v
                if opt_info.get('optimized', False):
                    optimization_info['optimized'] = True
                    optimization_info['items_optimized'] += 1
            
            return optimized_dict, optimization_info
        else:
            return value, {'optimized': False}
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            if hasattr(value, 'memory_usage'):
                return value.memory_usage(deep=True).sum()
            elif hasattr(value, 'nbytes'):
                return value.nbytes
            elif hasattr(value, '__sizeof__'):
                return value.__sizeof__()
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimate
    
    def _ensure_memory_limits(self):
        """Ensure cache stays within memory limits."""
        current_memory = self._get_current_memory_usage()
        max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
        
        if current_memory > max_memory_bytes:
            self._evict_items(0.2)  # Evict 20% of items
        
        # Check item count limit
        if len(self._cache) > self.config.max_items:
            self._evict_items(0.1)  # Evict 10% of items
    
    def _evict_items(self, ratio: float):
        """Evict items based on strategy."""
        if not self._cache:
            return
        
        num_to_evict = max(1, int(len(self._cache) * ratio))
        
        if self.config.strategy == CacheStrategy.LRU:
            # Remove oldest items
            for _ in range(num_to_evict):
                if self._cache:
                    self._cache.popitem(last=False)
                    self.statistics.evictions += 1
        
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used items
            access_counts = {key: item.access_count for key, item in self._cache.items()}
            sorted_items = sorted(access_counts.items(), key=lambda x: x[1])
            
            for key, _ in sorted_items[:num_to_evict]:
                if key in self._cache:
                    del self._cache[key]
                    self.statistics.evictions += 1
        
        elif self.config.strategy == CacheStrategy.SIZE_BASED:
            # Remove largest items
            size_items = [(key, item.size_bytes) for key, item in self._cache.items()]
            sorted_items = sorted(size_items, key=lambda x: x[1], reverse=True)
            
            for key, _ in sorted_items[:num_to_evict]:
                if key in self._cache:
                    del self._cache[key]
                    self.statistics.evictions += 1
        
        self._update_statistics()
    
    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        tprint_warning("Performing aggressive cache cleanup")
        
        # Evict 50% of items
        self._evict_items(0.5)
        
        # Use advanced memory manager for comprehensive cleanup
        self.memory_manager.cleanup_all()
        
        # Clear access history
        self._access_history.clear()
    
    def _update_hit_rate(self):
        """Update hit rate statistics."""
        total_requests = self.statistics.hits + self.statistics.misses
        if total_requests > 0:
            self.statistics.hit_rate = self.statistics.hits / total_requests
    
    def _update_statistics(self):
        """Update cache statistics."""
        if not self._cache:
            return
        
        total_memory = self._get_current_memory_usage()
        self.statistics.total_memory_used_mb = total_memory / (1024 * 1024)
        self.statistics.peak_memory_used_mb = max(
            self.statistics.peak_memory_used_mb,
            self.statistics.total_memory_used_mb
        )
        
        if self._cache:
            avg_size = sum(item.size_bytes for item in self._cache.values()) / len(self._cache)
            self.statistics.average_item_size_mb = avg_size / (1024 * 1024)
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_history.clear()
            self.statistics = CacheStatistics()
            gc.collect()
            tprint_info("Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'hits': self.statistics.hits,
                'misses': self.statistics.misses,
                'hit_rate': self.statistics.hit_rate,
                'evictions': self.statistics.evictions,
                'compressions': self.statistics.compressions,
                'data_type_optimizations': self.statistics.data_type_optimizations,
                'total_memory_used_mb': self.statistics.total_memory_used_mb,
                'peak_memory_used_mb': self.statistics.peak_memory_used_mb,
                'average_item_size_mb': self.statistics.average_item_size_mb,
                'compression_ratio': self.statistics.compression_ratio,
                'item_count': len(self._cache),
                'max_memory_mb': self.config.max_memory_mb,
                'strategy': self.config.strategy.value
            }
    
    def shutdown(self):
        """Shutdown the cache system."""
        self._monitoring_active = False
        if self._memory_monitor_thread:
            self._memory_monitor_thread.join(timeout=2.0)
        
        self.clear()
        tprint_info("Cache system shutdown complete")

# Global cache instance
_global_cache: Optional[EnhancedCacheSystem] = None

def get_global_cache(config: Optional[CacheConfig] = None) -> EnhancedCacheSystem:
    """Get or create the global cache instance."""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = EnhancedCacheSystem(config)
    
    return _global_cache

def cache_result(key_func: Optional[Callable] = None, 
                ttl: Optional[float] = None,
                cache_instance: Optional[EnhancedCacheSystem] = None):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Get cache instance
            cache = cache_instance or get_global_cache()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                tprint_debug(f"Cache hit for {func.__name__}")
                return result
            
            # Compute result
            tprint_debug(f"Cache miss for {func.__name__}, computing...")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

def optimize_dataframe_default(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame with default settings."""
    cache = get_global_cache()
    optimized_df, _ = cache.data_type_optimizer.optimize_dataframe(df)
    return optimized_df

def optimize_numpy_array_default(arr: np.ndarray) -> np.ndarray:
    """Optimize NumPy array with default settings."""
    cache = get_global_cache()
    optimized_arr, _ = cache.data_type_optimizer.optimize_numpy_array(arr)
    return optimized_arr