"""
Caching mixin for intelligent caching and performance optimization.

This mixin provides comprehensive caching capabilities including
memory caching, disk caching, and intelligent cache management.
"""

import time
import hashlib
import pickle
import os
import logging
from typing import Dict, Any, Optional, Union, Callable, Tuple, List
import pandas as pd
import numpy as np
from functools import wraps

from ..config import get_unified_config

logger = logging.getLogger(__name__)

class CachingMixin:
    """
    Mixin class providing intelligent caching capabilities.

    This mixin can be added to any class to provide automatic caching
    of expensive operations with intelligent cache management.
    """

    def __init__(self, *args, **kwargs):
        """Initialize caching mixin."""
        super().__init__(*args, **kwargs)

        # Get unified configuration
        self.config = get_unified_config()

        # Cache storage
        self._memory_cache = {}
        self._disk_cache_dir = None

        # Cache statistics
        self._cache_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'cache_evictions': 0,
            'cache_size': 0,
            'disk_cache_size': 0
        }

        # Cache configuration
        self._cache_config = {
            'max_memory_size': self.config.optimization.cache_size,
            'max_disk_size': 1000,  # MB
            'ttl': self.config.vectorbt.cache_ttl,
            'enable_memory_cache': self.config.optimization.enable_caching,
            'enable_disk_cache': self.config.vectorbt.enable_disk_cache,
            'compression': True,
            'serialization': 'pickle'
        }

        # Initialize disk cache if enabled
        if self._cache_config['enable_disk_cache']:
            self._initialize_disk_cache()

    def _initialize_disk_cache(self) -> None:
        """Initialize disk cache directory."""
        try:
            if self.config.vectorbt.disk_cache_path:
                self._disk_cache_dir = self.config.vectorbt.disk_cache_path
            else:
                import tempfile
                self._disk_cache_dir = os.path.join(tempfile.gettempdir(), 'features_common_cache')

            os.makedirs(self._disk_cache_dir, exist_ok=True)
            logger.debug(f"Disk cache initialized at {self._disk_cache_dir}")

        except Exception as e:
            logger.warning(f"Failed to initialize disk cache: {e}")
            self._cache_config['enable_disk_cache'] = False

    def cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a string representation of all arguments
        key_parts = []

        # Add positional arguments
        for arg in args:
            if isinstance(arg, (pd.Series, pd.DataFrame)):
                # Use data hash for pandas objects
                key_parts.append(f"data_{self._hash_dataframe(arg)}")
            elif isinstance(arg, (list, tuple)):
                key_parts.append(f"list_{hash(tuple(arg))}")
            elif isinstance(arg, dict):
                key_parts.append(f"dict_{hash(tuple(sorted(arg.items())))}")
            else:
                key_parts.append(str(arg))

        # Add keyword arguments
        for key, value in sorted(kwargs.items()):
            if isinstance(value, (pd.Series, pd.DataFrame)):
                key_parts.append(f"{key}_data_{self._hash_dataframe(value)}")
            elif isinstance(value, (list, tuple)):
                key_parts.append(f"{key}_list_{hash(tuple(value))}")
            elif isinstance(value, dict):
                key_parts.append(f"{key}_dict_{hash(tuple(sorted(value.items())))}")
            else:
                key_parts.append(f"{key}_{value}")

        # Create hash of all parts
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _hash_dataframe(self, data: Union[pd.Series, pd.DataFrame]) -> str:
        """Generate hash for pandas data."""
        try:
            # Use pandas hash for efficiency
            if isinstance(data, pd.Series):
                return str(pd.util.hash_pandas_object(data).sum())
            else:
                return str(pd.util.hash_pandas_object(data).sum())
        except Exception:
            # Fallback to content hash
            return hashlib.md5(str(data.values).encode()).hexdigest()

    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get value from cache."""
        self._cache_stats['total_requests'] += 1

        # Try memory cache first
        if self._cache_config['enable_memory_cache'] and cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]

            # Check TTL
            if time.time() - entry['timestamp'] < self._cache_config['ttl']:
                self._cache_stats['cache_hits'] += 1
                self._cache_stats['memory_hits'] += 1
                return entry['value']
            else:
                # Expired entry
                del self._memory_cache[cache_key]
                self._cache_stats['cache_evictions'] += 1

        # Try disk cache
        if self._cache_config['enable_disk_cache'] and self._disk_cache_dir:
            disk_path = os.path.join(self._disk_cache_dir, f"{cache_key}.pkl")
            if os.path.exists(disk_path):
                try:
                    with open(disk_path, 'rb') as f:
                        entry = pickle.load(f)

                    # Check TTL
                    if time.time() - entry['timestamp'] < self._cache_config['ttl']:
                        self._cache_stats['cache_hits'] += 1
                        self._cache_stats['disk_hits'] += 1

                        # Store in memory cache for faster access
                        if self._cache_config['enable_memory_cache']:
                            self._store_in_memory_cache(cache_key, entry['value'])

                        return entry['value']
                    else:
                        # Expired entry
                        os.remove(disk_path)
                        self._cache_stats['cache_evictions'] += 1

                except Exception as e:
                    logger.warning(f"Failed to load from disk cache: {e}")

        # Cache miss
        self._cache_stats['cache_misses'] += 1
        return None

    def store_in_cache(self, cache_key: str, value: Any) -> None:
        """Store value in cache."""
        # Store in memory cache
        if self._cache_config['enable_memory_cache']:
            self._store_in_memory_cache(cache_key, value)

        # Store in disk cache
        if self._cache_config['enable_disk_cache'] and self._disk_cache_dir:
            self._store_in_disk_cache(cache_key, value)

    def _store_in_memory_cache(self, cache_key: str, value: Any) -> None:
        """Store value in memory cache."""
        # Check cache size limit
        if len(self._memory_cache) >= self._cache_config['max_memory_size']:
            self._evict_oldest_entry()

        # Store entry
        self._memory_cache[cache_key] = {
            'value': value,
            'timestamp': time.time()
        }
        self._cache_stats['cache_size'] = len(self._memory_cache)

    def _store_in_disk_cache(self, cache_key: str, value: Any) -> None:
        """Store value in disk cache."""
        try:
            disk_path = os.path.join(self._disk_cache_dir, f"{cache_key}.pkl")

            entry = {
                'value': value,
                'timestamp': time.time()
            }

            with open(disk_path, 'wb') as f:
                pickle.dump(entry, f)

            # Update disk cache size
            self._cache_stats['disk_cache_size'] += os.path.getsize(disk_path)

        except Exception as e:
            logger.warning(f"Failed to store in disk cache: {e}")

    def _evict_oldest_entry(self) -> None:
        """Evict the oldest entry from memory cache."""
        if not self._memory_cache:
            return

        # Find oldest entry
        oldest_key = min(self._memory_cache.keys(),
                        key=lambda k: self._memory_cache[k]['timestamp'])

        del self._memory_cache[oldest_key]
        self._cache_stats['cache_evictions'] += 1
        self._cache_stats['cache_size'] = len(self._memory_cache)

    def cached_operation(self,
                        operation_func: Callable,
                        *args, **kwargs) -> Any:
        """
        Execute an operation with caching.

        Args:
            operation_func: The operation function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation (from cache or execution)
        """
        # Generate cache key
        cache_key = self.cache_key(operation_func.__name__, *args, **kwargs)

        # Try to get from cache
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Execute operation
        result = operation_func(*args, **kwargs)

        # Store in cache
        self.store_in_cache(cache_key, result)

        return result

    def cache_method(self, ttl: Optional[float] = None):
        """
        Decorator for caching method results.

        Args:
            ttl: Time to live for cached results (overrides default)
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Use custom TTL if provided
                original_ttl = self._cache_config['ttl']
                if ttl is not None:
                    self._cache_config['ttl'] = ttl

                try:
                    return self.cached_operation(func, *args, **kwargs)
                finally:
                    # Restore original TTL
                    self._cache_config['ttl'] = original_ttl
            return wrapper
        return decorator

    def clear_cache(self, memory_only: bool = False) -> None:
        """Clear cache."""
        # Clear memory cache
        if self._cache_config['enable_memory_cache']:
            self._memory_cache.clear()
            self._cache_stats['cache_size'] = 0

        # Clear disk cache
        if not memory_only and self._cache_config['enable_disk_cache'] and self._disk_cache_dir:
            try:
                for filename in os.listdir(self._disk_cache_dir):
                    if filename.endswith('.pkl'):
                        os.remove(os.path.join(self._disk_cache_dir, filename))
                self._cache_stats['disk_cache_size'] = 0
                logger.info("Disk cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear disk cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._cache_stats.copy()

        # Calculate hit rates
        if stats['total_requests'] > 0:
            stats['hit_rate'] = stats['cache_hits'] / stats['total_requests']
            stats['miss_rate'] = stats['cache_misses'] / stats['total_requests']
            stats['memory_hit_rate'] = stats['memory_hits'] / stats['total_requests']
            stats['disk_hit_rate'] = stats['disk_hits'] / stats['total_requests']
        else:
            stats['hit_rate'] = 0.0
            stats['miss_rate'] = 0.0
            stats['memory_hit_rate'] = 0.0
            stats['disk_hit_rate'] = 0.0

        # Add configuration info
        stats['config'] = self._cache_config.copy()

        return stats

    def reset_cache_stats(self) -> None:
        """Reset cache statistics."""
        self._cache_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'cache_evictions': 0,
            'cache_size': 0,
            'disk_cache_size': 0
        }

    def get_cache_recommendations(self) -> List[str]:
        """Get recommendations for cache optimization."""
        recommendations = []
        stats = self.get_cache_stats()

        # Check hit rate
        if stats['hit_rate'] < 0.5:
            recommendations.append("Low cache hit rate - consider increasing cache size or TTL")

        # Check memory usage
        if stats['cache_size'] >= self._cache_config['max_memory_size'] * 0.9:
            recommendations.append("Memory cache near capacity - consider increasing max_memory_size")

        # Check disk usage
        if stats['disk_cache_size'] > self._cache_config['max_disk_size'] * 1024 * 1024:  # Convert to bytes
            recommendations.append("Disk cache exceeds size limit - consider clearing or increasing max_disk_size")

        # Check eviction rate
        if stats['cache_evictions'] > stats['total_requests'] * 0.1:
            recommendations.append("High cache eviction rate - consider increasing cache size")

        return recommendations

    def optimize_cache_settings(self) -> None:
        """Optimize cache settings based on current performance."""
        stats = self.get_cache_stats()

        # Adjust memory cache size based on hit rate
        if stats['hit_rate'] > 0.8 and stats['cache_size'] < self._cache_config['max_memory_size'] * 0.5:
            # High hit rate, low usage - can increase size
            new_size = min(self._cache_config['max_memory_size'] * 2, 10000)
            self._cache_config['max_memory_size'] = new_size
            logger.info(f"Increased memory cache size to {new_size}")

        elif stats['hit_rate'] < 0.3 and stats['cache_evictions'] > stats['total_requests'] * 0.2:
            # Low hit rate, high evictions - decrease size
            new_size = max(self._cache_config['max_memory_size'] // 2, 100)
            self._cache_config['max_memory_size'] = new_size
            logger.info(f"Decreased memory cache size to {new_size}")

        # Adjust TTL based on hit rate
        if stats['hit_rate'] > 0.9:
            # Very high hit rate - can increase TTL
            new_ttl = min(self._cache_config['ttl'] * 2, 7200)  # Max 2 hours
            self._cache_config['ttl'] = new_ttl
            logger.info(f"Increased cache TTL to {new_ttl} seconds")

        elif stats['hit_rate'] < 0.2:
            # Very low hit rate - decrease TTL
            new_ttl = max(self._cache_config['ttl'] // 2, 60)  # Min 1 minute
            self._cache_config['ttl'] = new_ttl
            logger.info(f"Decreased cache TTL to {new_ttl} seconds")

    def cached_operation(self, operation_func: Callable, *args, **kwargs) -> Any:
        """
        Execute an operation with caching.
        
        This method provides a simple interface for caching expensive operations.
        It automatically generates cache keys and handles cache hits/misses.
        
        Args:
            operation_func: The function to execute and cache
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the operation (from cache or execution)
        """
        from ..utils import TPRINT_AVAILABLE, tprint
        
        if not self._cache_config['enable_memory_cache']:
            # Caching disabled, execute directly
            return operation_func(*args, **kwargs)
        
        # Generate cache key
        cache_key = self.cache_key(operation_func.__name__, *args, **kwargs)
        
        if TPRINT_AVAILABLE:
            tprint(f"🔍 [CachingMixin] Checking cache for {operation_func.__name__}", color="blue")
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            cache_entry = self._memory_cache[cache_key]
            
            # Check if cache entry is still valid
            if time.time() - cache_entry['timestamp'] < self._cache_config['ttl']:
                self._cache_stats['cache_hits'] += 1
                self._cache_stats['memory_hits'] += 1
                
                if TPRINT_AVAILABLE:
                    tprint(f"✅ [CachingMixin] Cache HIT for {operation_func.__name__}", color="green")
                
                return cache_entry['result']
            else:
                # Cache entry expired, remove it
                del self._memory_cache[cache_key]
                self._cache_stats['cache_evictions'] += 1
        
        # Check disk cache if enabled
        if self._cache_config['enable_disk_cache'] and self._disk_cache_dir:
            disk_cache_path = os.path.join(self._disk_cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(disk_cache_path):
                try:
                    # Check file age
                    file_age = time.time() - os.path.getmtime(disk_cache_path)
                    if file_age < self._cache_config['ttl']:
                        # Load from disk cache
                        with open(disk_cache_path, 'rb') as f:
                            result = pickle.load(f)
                        
                        # Store in memory cache for faster access
                        self._store_in_memory_cache(cache_key, result)
                        
                        self._cache_stats['cache_hits'] += 1
                        self._cache_stats['disk_hits'] += 1
                        
                        if TPRINT_AVAILABLE:
                            tprint(f"✅ [CachingMixin] Disk cache HIT for {operation_func.__name__}", color="green")
                        
                        return result
                    else:
                        # File expired, remove it
                        os.remove(disk_cache_path)
                        
                except Exception as e:
                    logger.warning(f"Failed to load from disk cache: {e}")
        
        # Cache miss - execute operation
        self._cache_stats['cache_misses'] += 1
        
        if TPRINT_AVAILABLE:
            tprint(f"❌ [CachingMixin] Cache MISS for {operation_func.__name__}, executing", color="yellow")
        
        # Execute the operation
        result = operation_func(*args, **kwargs)
        
        # Store result in cache
        self._store_in_memory_cache(cache_key, result)
        
        # Store in disk cache if enabled
        if self._cache_config['enable_disk_cache'] and self._disk_cache_dir:
            self._store_in_disk_cache(cache_key, result)
        
        return result

    def _store_in_memory_cache(self, cache_key: str, result: Any) -> None:
        """Store result in memory cache."""
        try:
            # Check cache size limit
            if len(self._memory_cache) >= self._cache_config['max_memory_size']:
                # Remove oldest entry (simple LRU)
                oldest_key = min(self._memory_cache.keys(), 
                               key=lambda k: self._memory_cache[k]['timestamp'])
                del self._memory_cache[oldest_key]
                self._cache_stats['cache_evictions'] += 1
            
            # Store new entry
            self._memory_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            self._cache_stats['cache_size'] = len(self._memory_cache)
            
        except Exception as e:
            logger.warning(f"Failed to store in memory cache: {e}")

    def _store_in_disk_cache(self, cache_key: str, result: Any) -> None:
        """Store result in disk cache."""
        try:
            disk_cache_path = os.path.join(self._disk_cache_dir, f"{cache_key}.pkl")
            
            with open(disk_cache_path, 'wb') as f:
                pickle.dump(result, f)
            
            # Update disk cache size
            file_size = os.path.getsize(disk_cache_path)
            self._cache_stats['disk_cache_size'] += file_size
            
        except Exception as e:
            logger.warning(f"Failed to store in disk cache: {e}")
