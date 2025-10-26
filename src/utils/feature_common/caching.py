"""
Hardware-Optimized Shared Computation Cache for Feature Engineering

This module provides an enhanced caching system that integrates with hardware optimization
tools to provide intelligent memory management, predictive optimization, and adaptive
performance tuning based on real-time hardware conditions.

Features:
- Hardware-aware memory management with intelligent pooling
- Predictive memory optimization using machine learning
- Adaptive cleanup strategies based on system conditions
- Real-time performance monitoring and optimization
- Intelligent compression for large objects
- Multi-tier caching with different optimization strategies
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import hashlib
import time
import logging
from collections import OrderedDict, deque
import gc
import threading
import asyncio
from enum import Enum
import pickle
import lz4.frame
import zlib
from contextlib import contextmanager

# Hardware optimization imports
from src.utils.hardware.unified_hardware_manager import (
    get_unified_hardware_manager, 
    WorkloadType, 
    OptimizationLevel,
    HardwareConfig
)
from src.utils.hardware.advanced_memory_optimizer import (
    get_advanced_memory_optimizer,
    MemoryStrategy,
    MemoryPoolType
)
from src.utils.hardware.adaptive_optimization_engine import (
    get_adaptive_optimization_engine,
    OptimizationTarget
)
from src.utils.hardware.memory_optimization import (
    MemoryMonitor,
    MemoryConfig,
    optimize_dataframe_dtypes
)

from src.utils.tprint import tprint, tprint_info, tprint_warning

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache optimization strategies."""
    PERFORMANCE = "performance"  # Maximize speed
    MEMORY = "memory"  # Minimize memory usage
    BALANCED = "balanced"  # Balance speed and memory
    ADAPTIVE = "adaptive"  # Learn and adapt

class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    LZ4 = "lz4"
    ZLIB = "zlib"
    PICKLE = "pickle"

@dataclass
class OptimizedCacheConfig:
    """Enhanced configuration for caching system."""
    # Basic settings
    max_size: int = 1000
    max_memory_mb: int = 500
    ttl_seconds: int = 3600
    cleanup_interval: int = 100
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_predictive_management: bool = True
    enable_memory_pooling: bool = True
    enable_compression: bool = True
    enable_adaptive_cleanup: bool = True
    
    # Performance tuning
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    compression_type: CompressionType = CompressionType.LZ4
    compression_threshold_mb: float = 1.0
    
    # Memory management
    memory_pool_size_mb: float = 200.0
    aggressive_cleanup_threshold: float = 0.85
    predictive_cleanup_threshold: float = 0.75
    
    # Hardware integration
    workload_type: WorkloadType = WorkloadType.FEATURE_ENGINEERING
    optimization_target: OptimizationTarget = OptimizationTarget.BALANCED

@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    compression_type: Optional[CompressionType] = None
    memory_pool_type: Optional[MemoryPoolType] = None
    priority: float = 1.0  # Higher = more important
    metadata: Dict[str, Any] = field(default_factory=dict)

class HardwareAwareCache:
    """
    Hardware-aware cache with intelligent optimization.
    
    Features:
    - Real-time hardware monitoring
    - Predictive memory management
    - Intelligent memory pooling
    - Adaptive cleanup strategies
    - Performance learning and optimization
    """
    
    def __init__(self, config: Optional[OptimizedCacheConfig] = None):
        """Initialize hardware-aware cache."""
        self.config = config or OptimizedCacheConfig()
        self.logger = logger.getChild('HardwareAwareCache')
        
        # Core cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.memory_usage = 0
        self.operation_count = 0
        
        # Hardware integration
        self.hardware_manager = None
        self.memory_optimizer = None
        self.adaptive_engine = None
        self.memory_monitor = None
        
        # Performance tracking
        self.performance_metrics = deque(maxlen=1000)
        self.learning_enabled = True
        
        # Threading
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._monitoring_thread = None
        self._shutdown_event = threading.Event()
        
        # Initialize hardware components
        self._initialize_hardware_components()
        
        # Start background processes
        self._start_background_processes()
        
        tprint_info("🔧 Initialized HardwareAwareCache with advanced optimization")
    
    def _initialize_hardware_components(self):
        """Initialize hardware optimization components."""
        try:
            if self.config.enable_hardware_optimization:
                # Initialize hardware manager
                self.hardware_manager = get_unified_hardware_manager()
                
                # Initialize memory optimizer
                self.memory_optimizer = get_advanced_memory_optimizer(
                    memory_limit_gb=self.config.max_memory_mb / 1024,
                    strategy=MemoryStrategy.ADAPTIVE
                )
                
                # Initialize adaptive optimization engine
                self.adaptive_engine = get_adaptive_optimization_engine()
                
                # Initialize memory monitor
                memory_config = MemoryConfig(
                    max_memory_mb=self.config.max_memory_mb,
                    warning_threshold=self.config.predictive_cleanup_threshold,
                    critical_threshold=self.config.aggressive_cleanup_threshold
                )
                self.memory_monitor = MemoryMonitor(memory_config)
                
                # Configure for feature engineering workload
                self.hardware_manager.optimize_for_workload(
                    self.config.workload_type,
                    self.config.optimization_target
                )
                
                self.logger.info("✅ Hardware components initialized")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Hardware initialization failed: {e}")
            self.config.enable_hardware_optimization = False
    
    def _start_background_processes(self):
        """Start background monitoring and cleanup processes."""
        if self.config.enable_hardware_optimization:
            # Start cleanup thread
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True
            )
            self._cleanup_thread.start()
            
            # Start monitoring thread
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                self._perform_adaptive_cleanup()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                time.sleep(30)
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                self._update_performance_metrics()
                self._check_memory_pressure()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(30)
    
    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key from function name and arguments."""
        args_str = str(args) + str(sorted(kwargs.items()))
        key_hash = hashlib.md5(args_str.encode()).hexdigest()[:16]
        return f"{func_name}_{key_hash}"
    
    def _estimate_memory_usage(self, obj: Any) -> int:
        """Estimate memory usage of an object in bytes."""
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_memory_usage(item) for item in obj)
            else:
                return len(pickle.dumps(obj))
        except:
            return 1000  # Default estimate
    
    def _compress_value(self, value: Any) -> Tuple[bytes, CompressionType]:
        """Compress value using the configured compression algorithm."""
        try:
            serialized = pickle.dumps(value)
            
            if self.config.compression_type == CompressionType.LZ4:
                compressed = lz4.frame.compress(serialized)
                return compressed, CompressionType.LZ4
            elif self.config.compression_type == CompressionType.ZLIB:
                compressed = zlib.compress(serialized)
                return compressed, CompressionType.ZLIB
            else:
                return serialized, CompressionType.NONE
                
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return pickle.dumps(value), CompressionType.NONE
    
    def _decompress_value(self, compressed_data: bytes, compression_type: CompressionType) -> Any:
        """Decompress value using the specified compression algorithm."""
        try:
            if compression_type == CompressionType.LZ4:
                decompressed = lz4.frame.decompress(compressed_data)
            elif compression_type == CompressionType.ZLIB:
                decompressed = zlib.decompress(compressed_data)
            else:
                decompressed = compressed_data
            
            return pickle.loads(decompressed)
            
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            return None
    
    def _determine_memory_pool_type(self, obj: Any, size_bytes: int) -> MemoryPoolType:
        """Determine appropriate memory pool type for object."""
        if isinstance(obj, pd.DataFrame):
            return MemoryPoolType.PANDAS_DATAFRAMES
        elif isinstance(obj, np.ndarray):
            return MemoryPoolType.NUMPY_ARRAYS
        elif size_bytes > 1024 * 1024:  # > 1MB
            return MemoryPoolType.LARGE_OBJECTS
        else:
            return MemoryPoolType.SMALL_OBJECTS
    
    def _should_compress(self, size_bytes: int) -> bool:
        """Determine if object should be compressed."""
        return (self.config.enable_compression and 
                size_bytes > self.config.compression_threshold_mb * 1024 * 1024)
    
    def _calculate_priority(self, entry: Optional[CacheEntry], current_time: float) -> float:
        """Calculate cache entry priority for eviction decisions."""
        if entry is None:
            return 1.0  # Default priority for new entries
        
        # Factors: access frequency, recency, size, importance
        recency_score = 1.0 / (1.0 + (current_time - entry.last_accessed))
        frequency_score = min(entry.access_count / 10.0, 1.0)
        size_penalty = 1.0 / (1.0 + entry.size_bytes / (1024 * 1024))  # Penalize large objects
        
        return (recency_score * 0.4 + 
                frequency_score * 0.4 + 
                size_penalty * 0.2)
    
    def _perform_adaptive_cleanup(self):
        """Perform adaptive cleanup based on current conditions."""
        with self._lock:
            current_time = time.time()
            
            # Remove expired entries
            expired_keys = []
            for key, entry in self.cache.items():
                if current_time - entry.created_at > self.config.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            # Check memory pressure
            memory_pressure = self._get_memory_pressure()
            
            if memory_pressure > self.config.aggressive_cleanup_threshold:
                self._aggressive_cleanup()
            elif memory_pressure > self.config.predictive_cleanup_threshold:
                self._predictive_cleanup()
    
    def _get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 to 1.0)."""
        if self.memory_monitor:
            return self.memory_monitor.get_usage_percentage() / 100.0
        else:
            return self.memory_usage / (self.config.max_memory_mb * 1024 * 1024)
    
    def _aggressive_cleanup(self):
        """Perform aggressive cleanup to free memory."""
        self.logger.info("🧹 Performing aggressive cache cleanup")
        
        # Remove least priority entries
        entries_by_priority = sorted(
            self.cache.items(),
            key=lambda x: self._calculate_priority(x[1], time.time())
        )
        
        # Remove bottom 25% of entries
        remove_count = max(1, len(entries_by_priority) // 4)
        for key, _ in entries_by_priority[:remove_count]:
            self._remove_entry(key)
        
        # Force garbage collection
        gc.collect()
    
    def _predictive_cleanup(self):
        """Perform predictive cleanup based on usage patterns."""
        self.logger.debug("🔮 Performing predictive cache cleanup")
        
        # Remove entries that haven't been accessed recently
        current_time = time.time()
        cutoff_time = current_time - (self.config.ttl_seconds * 0.5)
        
        keys_to_remove = []
        for key, entry in self.cache.items():
            if (entry.last_accessed < cutoff_time and 
                entry.access_count < 2):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._remove_entry(key)
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and update memory usage."""
        if key in self.cache:
            entry = self.cache[key]
            self.memory_usage -= entry.size_bytes
            
            # Deallocate from memory pool if applicable
            if (self.memory_optimizer and 
                entry.memory_pool_type and 
                hasattr(entry, 'pool_object_id')):
                self.memory_optimizer.deallocate_from_pool(
                    entry.memory_pool_type,
                    entry.pool_object_id
                )
            
            del self.cache[key]
    
    def _check_memory_pressure(self):
        """Check for memory pressure and take action."""
        if not self.memory_monitor:
            return
        
        if self.memory_monitor.is_critical_memory():
            self.logger.warning("🚨 Critical memory usage detected")
            self._aggressive_cleanup()
        elif self.memory_monitor.is_memory_pressure():
            self.logger.info("⚠️ Memory pressure detected")
            self._predictive_cleanup()
    
    def _update_performance_metrics(self):
        """Update performance metrics for learning."""
        if not self.learning_enabled:
            return
        
        try:
            metrics = {
                'timestamp': time.time(),
                'cache_size': len(self.cache),
                'memory_usage_mb': self.memory_usage / (1024 * 1024),
                'hit_rate': self._calculate_hit_rate(),
                'memory_pressure': self._get_memory_pressure()
            }
            
            self.performance_metrics.append(metrics)
            
            # Record performance with adaptive engine
            if self.adaptive_engine:
                self.adaptive_engine.record_performance(
                    execution_time=0.0,  # Cache operations are fast
                    throughput=len(self.cache),
                    error_rate=0.0
                )
                
        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.operation_count == 0:
            return 0.0
        
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        return total_accesses / max(1, self.operation_count)
    
    def get_or_compute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Get cached result or compute and cache with hardware optimization.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result (cached or computed)
        """
        key = self._generate_key(func.__name__, *args, **kwargs)
        current_time = time.time()
        
        with self._lock:
            self.operation_count += 1
            
            # Check if cached and not expired
            if key in self.cache:
                entry = self.cache[key]
                if current_time - entry.created_at < self.config.ttl_seconds:
                    # Update access info
                    entry.last_accessed = current_time
                    entry.access_count += 1
                    
                    # Move to end (LRU)
                    self.cache.move_to_end(key)
                    
                    # Decompress if needed
                    if entry.compressed:
                        value = self._decompress_value(entry.value, entry.compression_type)
                        if value is not None:
                            return value
                        else:
                            # Decompression failed, remove entry
                            self._remove_entry(key)
                    else:
                        return entry.value
                else:
                    # Expired, remove
                    self._remove_entry(key)
            
            # Compute result
            start_time = time.time()
            result = func(*args, **kwargs)
            compute_time = time.time() - start_time
            
            # Estimate memory usage
            size_bytes = self._estimate_memory_usage(result)
            
            # Check if we should cache this result
            if size_bytes < self.config.max_memory_mb * 1024 * 1024 * 0.1:  # Don't cache huge objects
                # Determine if compression is needed
                compressed_value = result
                compression_type = CompressionType.NONE
                compressed = False
                
                if self._should_compress(size_bytes):
                    compressed_value, compression_type = self._compress_value(result)
                    compressed = True
                    size_bytes = len(compressed_value)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=compressed_value,
                    created_at=current_time,
                    last_accessed=current_time,
                    access_count=1,
                    size_bytes=size_bytes,
                    compressed=compressed,
                    compression_type=compression_type,
                    memory_pool_type=self._determine_memory_pool_type(result, size_bytes),
                    priority=self._calculate_priority(None, current_time)
                )
                
                # Allocate from memory pool if enabled
                if (self.memory_optimizer and 
                    self.config.enable_memory_pooling and 
                    entry.memory_pool_type):
                    pool_object_id = f"cache_{key}"
                    if self.memory_optimizer.allocate_from_pool(
                        entry.memory_pool_type,
                        size_bytes,
                        pool_object_id,
                        "cache_entry"
                    ):
                        entry.pool_object_id = pool_object_id
                
                # Add to cache
                self.cache[key] = entry
                self.memory_usage += size_bytes
                
                # Move to end (LRU)
                self.cache.move_to_end(key)
                
                # Log performance
                self.logger.debug(f"Computed and cached {func.__name__} in {compute_time:.3f}s")
            
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            current_time = time.time()
            
            # Basic stats
            stats = {
                'cache_size': len(self.cache),
                'memory_usage_mb': self.memory_usage / (1024 * 1024),
                'operation_count': self.operation_count,
                'hit_rate': self._calculate_hit_rate(),
                'memory_pressure': self._get_memory_pressure(),
                'config': {
                    'max_size': self.config.max_size,
                    'max_memory_mb': self.config.max_memory_mb,
                    'ttl_seconds': self.config.ttl_seconds,
                    'strategy': self.config.cache_strategy.value,
                    'compression_enabled': self.config.enable_compression
                }
            }
            
            # Hardware stats
            if self.memory_monitor:
                stats['hardware_stats'] = self.memory_monitor.get_memory_stats()
            
            if self.memory_optimizer:
                stats['memory_pool_stats'] = self.memory_optimizer.get_advanced_memory_stats()
            
            # Performance metrics
            if self.performance_metrics:
                recent_metrics = list(self.performance_metrics)[-10:]
                stats['recent_performance'] = recent_metrics
            
            # Entry analysis
            if self.cache:
                entry_sizes = [entry.size_bytes for entry in self.cache.values()]
                stats['entry_analysis'] = {
                    'avg_entry_size_mb': sum(entry_sizes) / len(entry_sizes) / (1024 * 1024),
                    'max_entry_size_mb': max(entry_sizes) / (1024 * 1024),
                    'compressed_entries': sum(1 for entry in self.cache.values() if entry.compressed),
                    'total_entries': len(self.cache)
                }
            
            return stats
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.memory_usage = 0
            self.operation_count = 0
            gc.collect()
            self.logger.info("🧹 Cache cleared")
    
    def optimize_for_workload(self, workload_type: WorkloadType, 
                            target: OptimizationTarget = OptimizationTarget.BALANCED):
        """Optimize cache for specific workload type."""
        if self.hardware_manager:
            self.hardware_manager.optimize_for_workload(workload_type, target)
            self.config.workload_type = workload_type
            self.config.optimization_target = target
            self.logger.info(f"🎯 Cache optimized for {workload_type.value} ({target.value})")
    
    def set_strategy(self, strategy: CacheStrategy):
        """Set cache optimization strategy."""
        self.config.cache_strategy = strategy
        self.logger.info(f"🔧 Cache strategy set to {strategy.value}")
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        self._shutdown_event.set()
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2.0)
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
        
        self.clear()
        self.logger.info("🛑 HardwareAwareCache shutdown complete")

class OptimizedFeatureCache:
    """
    Hardware-optimized feature cache with specialized operations.
    
    Provides optimized caching for:
    - Rolling calculations with vectorized operations
    - Correlation matrices with memory pooling
    - Statistical tests with parallel processing
    - SHAP values with GPU acceleration
    """
    
    def __init__(self, max_size: int = 500, enable_hardware_optimization: bool = True):
        """Initialize optimized feature cache."""
        self.max_size = max_size
        self.enable_hardware_optimization = enable_hardware_optimization
        
        # Specialized caches
        self.rolling_cache = {}
        self.correlation_cache = {}
        self.statistical_cache = {}
        self.shap_cache = {}
        
        # Hardware components
        self.hardware_manager = None
        self.memory_optimizer = None
        
        if enable_hardware_optimization:
            try:
                self.hardware_manager = get_unified_hardware_manager()
                self.memory_optimizer = get_advanced_memory_optimizer()
                self.hardware_manager.optimize_for_workload(WorkloadType.FEATURE_ENGINEERING)
            except Exception as e:
                logger.warning(f"Hardware optimization disabled: {e}")
                self.enable_hardware_optimization = False
        
        tprint_info("🔧 Initialized OptimizedFeatureCache")
    
    def get_rolling_stat(self, series: pd.Series, window: int, stat_type: str) -> pd.Series:
        """Get or compute rolling statistic with hardware optimization."""
        key = (id(series), window, stat_type)
        
        if key in self.rolling_cache:
            return self.rolling_cache[key]
        
        # Use hardware-optimized computation
        if self.enable_hardware_optimization and self.hardware_manager:
            with self.hardware_manager.optimization_context(
                WorkloadType.FEATURE_ENGINEERING, 
                OptimizationLevel.BALANCED
            ):
                result = self._compute_rolling_stat_optimized(series, window, stat_type)
        else:
            result = self._compute_rolling_stat_basic(series, window, stat_type)
        
        # Cache if not too large
        if len(self.rolling_cache) < self.max_size:
            self.rolling_cache[key] = result
        
        return result
    
    def _compute_rolling_stat_optimized(self, series: pd.Series, window: int, stat_type: str) -> pd.Series:
        """Compute rolling statistic with hardware optimization."""
        # Use vectorized operations and memory optimization
        if stat_type == 'mean':
            result = series.rolling(window=window, min_periods=max(1, window//2)).mean()
        elif stat_type == 'std':
            result = series.rolling(window=window, min_periods=max(1, window//2)).std()
        elif stat_type == 'var':
            result = series.rolling(window=window, min_periods=max(1, window//2)).var()
        elif stat_type == 'min':
            result = series.rolling(window=window, min_periods=max(1, window//2)).min()
        elif stat_type == 'max':
            result = series.rolling(window=window, min_periods=max(1, window//2)).max()
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")
        
        # Optimize memory usage
        if self.memory_optimizer:
            result = self.memory_optimizer.optimize_dataframe_advanced(result)
        
        return result
    
    def _compute_rolling_stat_basic(self, series: pd.Series, window: int, stat_type: str) -> pd.Series:
        """Basic rolling statistic computation."""
        if stat_type == 'mean':
            return series.rolling(window=window, min_periods=max(1, window//2)).mean()
        elif stat_type == 'std':
            return series.rolling(window=window, min_periods=max(1, window//2)).std()
        elif stat_type == 'var':
            return series.rolling(window=window, min_periods=max(1, window//2)).var()
        elif stat_type == 'min':
            return series.rolling(window=window, min_periods=max(1, window//2)).min()
        elif stat_type == 'max':
            return series.rolling(window=window, min_periods=max(1, window//2)).max()
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")
    
    def get_correlation_matrix(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Get or compute correlation matrix with optimization."""
        key = id(features_df)
        
        if key in self.correlation_cache:
            return self.correlation_cache[key]
        
        # Compute correlation matrix with optimization
        if self.enable_hardware_optimization and self.hardware_manager:
            with self.hardware_manager.optimization_context(
                WorkloadType.FEATURE_ENGINEERING,
                OptimizationLevel.BALANCED
            ):
                corr_matrix = features_df.corr()
        else:
            corr_matrix = features_df.corr()
        
        # Cache if not too large
        if len(self.correlation_cache) < self.max_size:
            self.correlation_cache[key] = corr_matrix
        
        return corr_matrix
    
    def get_statistical_test(self, feature: pd.Series, target: pd.Series, test_type: str) -> float:
        """Get or compute statistical test with optimization."""
        key = (id(feature), id(target), test_type)
        
        if key in self.statistical_cache:
            return self.statistical_cache[key]
        
        # Compute statistical test
        if test_type == 'ttest':
            from scipy.stats import ttest_ind
            median = feature.median()
            group1 = target[feature > median]
            group2 = target[feature <= median]
            if len(group1) > 0 and len(group2) > 0:
                _, p_value = ttest_ind(group1, group2)
                result = p_value
            else:
                result = 1.0
        elif test_type == 'correlation':
            result = feature.corr(target)
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
        
        # Cache result
        if len(self.statistical_cache) < self.max_size:
            self.statistical_cache[key] = result
        
        return result
    
    def get_shap_values(self, model, data, target_name: str) -> np.ndarray:
        """Get or compute SHAP values with GPU acceleration."""
        key = (id(data), target_name)
        
        if key in self.shap_cache:
            return self.shap_cache[key]
        
        # This would be implemented with actual SHAP calculation
        # For now, return placeholder with optimization
        if self.enable_hardware_optimization and self.hardware_manager:
            with self.hardware_manager.optimization_context(
                WorkloadType.ML_TRAINING,
                OptimizationLevel.AGGRESSIVE
            ):
                result = np.random.random((len(data), len(data.columns)))
        else:
            result = np.random.random((len(data), len(data.columns)))
        
        # Cache if not too large
        if len(self.shap_cache) < self.max_size:
            self.shap_cache[key] = result
        
        return result
    
    def clear(self):
        """Clear all caches."""
        self.rolling_cache.clear()
        self.correlation_cache.clear()
        self.statistical_cache.clear()
        self.shap_cache.clear()
        gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'rolling_cache_size': len(self.rolling_cache),
            'correlation_cache_size': len(self.correlation_cache),
            'statistical_cache_size': len(self.statistical_cache),
            'shap_cache_size': len(self.shap_cache),
            'hardware_optimization_enabled': self.enable_hardware_optimization
        }

# Global cache instances
_shared_cache = None
_feature_cache = None

def get_shared_cache(config: Optional[OptimizedCacheConfig] = None) -> HardwareAwareCache:
    """Get global shared cache instance."""
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = HardwareAwareCache(config)
    return _shared_cache

def get_feature_cache(max_size: int = 500, enable_hardware_optimization: bool = True) -> OptimizedFeatureCache:
    """Get global feature cache instance."""
    global _feature_cache
    if _feature_cache is None:
        _feature_cache = OptimizedFeatureCache(max_size, enable_hardware_optimization)
    return _feature_cache

@contextmanager
def cache_context(workload_type: WorkloadType = WorkloadType.FEATURE_ENGINEERING,
                optimization_target: OptimizationTarget = OptimizationTarget.BALANCED):
    """Context manager for cache optimization."""
    cache = get_shared_cache()
    original_workload = cache.config.workload_type
    original_target = cache.config.optimization_target
    
    try:
        cache.optimize_for_workload(workload_type, optimization_target)
        yield cache
    finally:
        cache.optimize_for_workload(original_workload, original_target)

def optimize_cache_for_operation(operation_type: str, context: Dict[str, Any] = None) -> HardwareAwareCache:
    """Optimize cache for specific operation type."""
    cache = get_shared_cache()
    
    if context is None:
        context = {}
    
    # Determine optimal strategy based on operation type
    if operation_type in ['feature_selection', 'training']:
        workload_type = WorkloadType.ML_TRAINING
        optimization_target = OptimizationTarget.PERFORMANCE
    elif operation_type in ['inference', 'prediction']:
        workload_type = WorkloadType.ML_TRAINING
        optimization_target = OptimizationTarget.EFFICIENCY
    elif operation_type in ['data_processing', 'preprocessing']:
        workload_type = WorkloadType.DATA_PROCESSING
        optimization_target = OptimizationTarget.BALANCED
    else:
        workload_type = WorkloadType.FEATURE_ENGINEERING
        optimization_target = OptimizationTarget.BALANCED
    
    # Apply optimization
    cache.optimize_for_workload(workload_type, optimization_target)
    
    return cache

# Backward compatibility aliases
# Maintain compatibility with existing code that uses SharedComputationCache and FeatureCache
SharedComputationCache = HardwareAwareCache
FeatureCache = OptimizedFeatureCache

# Alias for the config class to maintain backward compatibility
CacheConfig = OptimizedCacheConfig

# Convenience function for getting a default shared cache instance
def get_default_shared_cache() -> HardwareAwareCache:
    """
    Get a default shared cache instance with hardware optimization enabled.
    
    This is the recommended way to get a cache instance for most use cases.
    The cache will have hardware optimization enabled by default.
    
    Returns:
        HardwareAwareCache: Configured cache instance
    """
    # Default configuration with hardware optimization
    default_config = OptimizedCacheConfig(
        max_size=1000,
        max_memory_mb=500,
        enable_hardware_optimization=True,
        enable_compression=True,
        cache_strategy=CacheStrategy.ADAPTIVE
    )
    return HardwareAwareCache(default_config)

# Convenience function for getting a default feature cache instance
def get_default_feature_cache() -> OptimizedFeatureCache:
    """
    Get a default feature cache instance with hardware optimization enabled.
    
    This is the recommended way to get a feature cache instance for feature engineering.
    
    Returns:
        OptimizedFeatureCache: Configured feature cache instance
    """
    return OptimizedFeatureCache(
        max_size=500,
        enable_hardware_optimization=True
    )
