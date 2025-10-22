"""
Intelligent Caching System for Profit Labeling (Phase 1)

This module provides hardware-optimized caching for repeated calculations
using M1MemoryOptimizer and advanced memory management tools.

Key Features:
- Hardware-optimized memory management
- Intelligent cache eviction strategies
- Performance monitoring and metrics
- Memory pressure-aware caching
- Automatic cache optimization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import hashlib
import weakref
from collections import OrderedDict
import gc

# Import hardware optimization tools
try:
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.memory_management import MemoryManager, MemoryManagerConfig, MemoryStrategy
    HARDWARE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware optimization tools not available: {e}")
    HARDWARE_AVAILABLE = False

from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_performance
)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    SIZE = "size"  # Largest items first
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


@dataclass
class CacheConfig:
    """Configuration for intelligent caching system."""
    
    # Basic settings
    max_cache_size: int = 1000
    max_memory_mb: float = 100.0
    enable_compression: bool = False
    
    # Eviction strategy
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    
    # Performance settings
    enable_monitoring: bool = True
    enable_auto_cleanup: bool = True
    cleanup_interval: float = 30.0  # seconds
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    memory_pressure_threshold: float = 0.8
    
    # Cache behavior
    enable_weak_references: bool = True
    enable_serialization: bool = True
    serialization_threshold_mb: float = 10.0


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    compressed: bool = False
    serialized: bool = False


class IntelligentCachingSystem:
    """
    Hardware-optimized intelligent caching system.
    
    Provides advanced caching with memory pressure awareness,
    automatic optimization, and performance monitoring.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize intelligent caching system."""
        self.config = config or CacheConfig()
        self.logger = logging.getLogger("IntelligentCachingSystem")
        
        # Initialize hardware optimization tools
        self._initialize_hardware_tools()
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._weak_refs: Dict[str, weakref.ref] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'serializations': 0,
            'total_operations': 0,
            'memory_usage_mb': 0.0,
            'peak_memory_mb': 0.0
        }
        
        # Memory monitoring
        self._last_cleanup = time.time()
        self._memory_usage_history = []
        
        tprint_success("✅ IntelligentCachingSystem initialized with hardware optimizations")
    
    def _initialize_hardware_tools(self):
        """Initialize hardware optimization tools."""
        try:
            if HARDWARE_AVAILABLE and self.config.enable_hardware_optimization:
                # Initialize memory optimizer
                self.memory_optimizer = M1MemoryOptimizer(
                    memory_limit_gb=self.config.max_memory_mb / 1024
                )
                tprint_info("   → M1MemoryOptimizer: Initialized")
                
                # Initialize CPU optimizer
                self.cpu_optimizer = M1CPUOptimizer()
                tprint_info("   → M1CPUOptimizer: Initialized")
                
                # Initialize memory manager
                memory_config = MemoryManagerConfig(
                    strategy=MemoryStrategy.MODERATE,
                    enable_monitoring=True,
                    memory_threshold_mb=self.config.max_memory_mb * 0.8,
                    max_memory_mb=self.config.max_memory_mb
                )
                self.memory_manager = MemoryManager(memory_config)
                tprint_info("   → MemoryManager: Initialized")
            else:
                self.memory_optimizer = None
                self.cpu_optimizer = None
                self.memory_manager = None
                tprint_warning("   → Hardware optimization: Not available")
                
        except Exception as e:
            tprint_error(f"Failed to initialize hardware tools: {e}")
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.memory_manager = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with hardware optimization."""
        self.performance_metrics['total_operations'] += 1
        
        # Check if key exists
        if key not in self._cache:
            self.performance_metrics['misses'] += 1
            return None
        
        # Get entry and update access info
        entry = self._cache[key]
        entry.last_accessed = time.time()
        entry.access_count += 1
        
        # Move to end (LRU behavior)
        self._cache.move_to_end(key)
        
        # Deserialize if needed
        if entry.serialized:
            try:
                entry.value = self._deserialize_value(entry.value)
                entry.serialized = False
            except Exception as e:
                tprint_warning(f"Failed to deserialize cache entry {key}: {e}")
                self._remove_entry(key)
                return None
        
        self.performance_metrics['hits'] += 1
        
        # Check if auto-cleanup is needed
        if self.config.enable_auto_cleanup:
            self._check_auto_cleanup()
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache with hardware optimization."""
        try:
            # Calculate value size
            size_bytes = self._calculate_size(value)
            
            # Check if value is too large
            if size_bytes > self.config.max_memory_mb * 1024 * 1024:
                tprint_warning(f"Value too large for cache: {size_bytes / 1024 / 1024:.2f}MB")
                return False
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                size_bytes=size_bytes
            )
            
            # Apply optimizations based on size
            if size_bytes > self.config.serialization_threshold_mb * 1024 * 1024:
                entry.value = self._serialize_value(value)
                entry.serialized = True
                self.performance_metrics['serializations'] += 1
            
            # Check memory pressure and evict if needed
            self._check_memory_pressure()
            
            # Add to cache
            self._cache[key] = entry
            
            # Update memory usage
            self._update_memory_usage()
            
            return True
            
        except Exception as e:
            tprint_error(f"Failed to set cache entry {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            self._remove_entry(key)
            return True
        return False
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._weak_refs.clear()
        self._update_memory_usage()
        tprint_info("🧹 Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        hit_rate = (
            self.performance_metrics['hits'] / 
            (self.performance_metrics['hits'] + self.performance_metrics['misses'])
            if (self.performance_metrics['hits'] + self.performance_metrics['misses']) > 0 else 0.0
        )
        
        return {
            **self.performance_metrics,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache),
            'max_cache_size': self.config.max_cache_size,
            'memory_usage_mb': self.performance_metrics['memory_usage_mb'],
            'max_memory_mb': self.config.max_memory_mb,
            'memory_utilization': (
                self.performance_metrics['memory_usage_mb'] / self.config.max_memory_mb
                if self.config.max_memory_mb > 0 else 0.0
            ),
            'hardware_optimization_enabled': self.memory_optimizer is not None
        }
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
                return value.nbytes
            elif isinstance(value, (list, tuple, dict)):
                return len(str(value).encode('utf-8'))
            else:
                return len(str(value).encode('utf-8'))
        except:
            return 1024  # Default estimate
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
                return value.tobytes() if hasattr(value, 'tobytes') else str(value).encode('utf-8')
            else:
                return str(value).encode('utf-8')
        except:
            return str(value).encode('utf-8')
    
    def _deserialize_value(self, serialized_value: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try to deserialize as numpy array first
            return np.frombuffer(serialized_value, dtype=np.float64)
        except:
            # Fallback to string
            return serialized_value.decode('utf-8')
    
    def _check_memory_pressure(self):
        """Check memory pressure and evict if needed."""
        current_memory = self.performance_metrics['memory_usage_mb']
        max_memory = self.config.max_memory_mb
        
        if current_memory > max_memory * self.config.memory_pressure_threshold:
            # Evict entries based on strategy
            self._evict_entries()
    
    def _evict_entries(self):
        """Evict entries based on configured strategy."""
        if not self._cache:
            return
        
        # Calculate how many entries to evict
        target_size = int(len(self._cache) * 0.8)  # Evict 20%
        entries_to_evict = len(self._cache) - target_size
        
        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used entries
            for _ in range(min(entries_to_evict, len(self._cache))):
                if self._cache:
                    key, _ = self._cache.popitem(last=False)
                    self._remove_entry(key)
        
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used entries
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].access_count
            )
            for i in range(min(entries_to_evict, len(sorted_entries))):
                key, _ = sorted_entries[i]
                self._remove_entry(key)
        
        elif self.config.strategy == CacheStrategy.SIZE:
            # Remove largest entries
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].size_bytes,
                reverse=True
            )
            for i in range(min(entries_to_evict, len(sorted_entries))):
                key, _ = sorted_entries[i]
                self._remove_entry(key)
        
        else:  # ADAPTIVE
            # Adaptive strategy based on access patterns
            self._adaptive_eviction(entries_to_evict)
        
        self.performance_metrics['evictions'] += entries_to_evict
        self._update_memory_usage()
    
    def _adaptive_eviction(self, entries_to_evict: int):
        """Adaptive eviction based on access patterns."""
        # Score entries based on recency, frequency, and size
        scored_entries = []
        
        for key, entry in self._cache.items():
            # Recency score (higher is more recent)
            recency_score = entry.last_accessed - entry.created_at
            
            # Frequency score (higher is more frequent)
            frequency_score = entry.access_count
            
            # Size penalty (larger items are more likely to be evicted)
            size_penalty = entry.size_bytes / (1024 * 1024)  # Convert to MB
            
            # Combined score (lower is more likely to be evicted)
            score = recency_score + frequency_score - size_penalty
            scored_entries.append((key, score))
        
        # Sort by score (ascending) and evict lowest scores
        scored_entries.sort(key=lambda x: x[1])
        
        for i in range(min(entries_to_evict, len(scored_entries))):
            key, _ = scored_entries[i]
            self._remove_entry(key)
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and update metrics."""
        if key in self._cache:
            entry = self._cache[key]
            self.performance_metrics['memory_usage_mb'] -= entry.size_bytes / (1024 * 1024)
            del self._cache[key]
            
            # Remove weak reference if exists
            if key in self._weak_refs:
                del self._weak_refs[key]
    
    def _update_memory_usage(self):
        """Update memory usage metrics."""
        total_size = sum(entry.size_bytes for entry in self._cache.values())
        self.performance_metrics['memory_usage_mb'] = total_size / (1024 * 1024)
        self.performance_metrics['peak_memory_mb'] = max(
            self.performance_metrics['peak_memory_mb'],
            self.performance_metrics['memory_usage_mb']
        )
    
    def _check_auto_cleanup(self):
        """Check if auto-cleanup is needed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.config.cleanup_interval:
            self._auto_cleanup()
            self._last_cleanup = current_time
    
    def _auto_cleanup(self):
        """Perform automatic cleanup operations."""
        # Memory optimization
        if self.memory_optimizer:
            self.memory_optimizer.optimize_memory_usage()
        
        # Garbage collection
        gc.collect()
        
        # Update memory usage
        self._update_memory_usage()
        
        tprint_info("🧹 Auto-cleanup performed")


# Decorator for automatic caching
def cached(cache_system: IntelligentCachingSystem, ttl: Optional[float] = None):
    """Decorator for automatic function result caching."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = cache_system.get(cache_key)
            if result is not None:
                return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_system.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Factory function for easy instantiation
def get_intelligent_caching_system(config: Optional[CacheConfig] = None) -> IntelligentCachingSystem:
    """
    Get an intelligent caching system instance.
    
    Args:
        config: Optional configuration for the caching system
        
    Returns:
        IntelligentCachingSystem instance
    """
    return IntelligentCachingSystem(config)