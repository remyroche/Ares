"""
Performance Optimization Module for ML Common Operations

This module provides comprehensive performance optimizations including:
- Advanced caching strategies
- Memory profiling and optimization
- Async/await patterns for I/O operations
- Progressive loading for large datasets
- VectorBT integration for financial operations
- M1 hardware optimizations
"""

from .caching_strategies import (
    CacheStrategy, CacheEvictionPolicy, CacheConfig, CacheEntry,
    MemoryCacheStrategy, VectorBTCacheStrategy, HybridCacheStrategy,
    MLCommonCache, get_ml_common_cache, cached, cache_get, cache_set,
    cache_delete, cache_clear, get_cache_stats
)

from .memory_profiler import (
    MemoryProfileLevel, MemoryOptimizationStrategy, MemoryProfile,
    MemoryOptimizationConfig, MemoryProfiler, get_memory_profiler,
    memory_profile, memory_optimize, start_memory_monitoring,
    stop_memory_monitoring, get_memory_stats, generate_memory_report
)

from .async_patterns import (
    AsyncOperationType, AsyncExecutionStrategy, AsyncConfig, AsyncOperation,
    ThreadPoolAsyncExecutor, ProcessPoolAsyncExecutor, AsyncioExecutor,
    HybridAsyncExecutor, AdaptiveAsyncExecutor, AsyncOperationManager,
    get_async_operation_manager, async_execute, execute_async,
    execute_batch_async, async_read_file, async_write_file,
    async_read_json, async_write_json, async_http_get,
    async_process_dataframe, async_vectorbt_operation, async_m1_optimized_operation
)

from .progressive_loading import (
    LoadingStrategy, DataFormat, ProgressiveLoadingConfig, DataChunk,
    ProgressiveLoader, CSVProgressiveLoader, ParquetProgressiveLoader,
    HDF5ProgressiveLoader, AdaptiveProgressiveLoader, ProgressiveDataProcessor,
    get_progressive_data_processor, progressive_load, load_data_progressively,
    stream_data_progressively
)

# VectorBT integration
try:
    from .vectorbt_integration import (
        VectorBTPerformanceOptimizer, VectorBTCacheStrategy,
        get_vectorbt_performance_optimizer, vectorbt_optimize,
        vectorbt_cached, vectorbt_async_execute
    )
    VECTORBT_INTEGRATION_AVAILABLE = True
except ImportError:
    VECTORBT_INTEGRATION_AVAILABLE = False

# M1 optimizations
try:
    from .m1_integration import (
        M1PerformanceOptimizer, M1MemoryStrategy, M1CPUStrategy,
        get_m1_performance_optimizer, m1_optimize, m1_cached,
        m1_async_execute
    )
    M1_INTEGRATION_AVAILABLE = True
except ImportError:
    M1_INTEGRATION_AVAILABLE = False

__all__ = [
    # Caching
    'CacheStrategy', 'CacheEvictionPolicy', 'CacheConfig', 'CacheEntry',
    'MemoryCacheStrategy', 'VectorBTCacheStrategy', 'HybridCacheStrategy',
    'MLCommonCache', 'get_ml_common_cache', 'cached', 'cache_get', 'cache_set',
    'cache_delete', 'cache_clear', 'get_cache_stats',
    
    # Memory Profiling
    'MemoryProfileLevel', 'MemoryOptimizationStrategy', 'MemoryProfile',
    'MemoryOptimizationConfig', 'MemoryProfiler', 'get_memory_profiler',
    'memory_profile', 'memory_optimize', 'start_memory_monitoring',
    'stop_memory_monitoring', 'get_memory_stats', 'generate_memory_report',
    
    # Async Patterns
    'AsyncOperationType', 'AsyncExecutionStrategy', 'AsyncConfig', 'AsyncOperation',
    'ThreadPoolAsyncExecutor', 'ProcessPoolAsyncExecutor', 'AsyncioExecutor',
    'HybridAsyncExecutor', 'AdaptiveAsyncExecutor', 'AsyncOperationManager',
    'get_async_operation_manager', 'async_execute', 'execute_async',
    'execute_batch_async', 'async_read_file', 'async_write_file',
    'async_read_json', 'async_write_json', 'async_http_get',
    'async_process_dataframe', 'async_vectorbt_operation', 'async_m1_optimized_operation',
    
    # Progressive Loading
    'LoadingStrategy', 'DataFormat', 'ProgressiveLoadingConfig', 'DataChunk',
    'ProgressiveLoader', 'CSVProgressiveLoader', 'ParquetProgressiveLoader',
    'HDF5ProgressiveLoader', 'AdaptiveProgressiveLoader', 'ProgressiveDataProcessor',
    'get_progressive_data_processor', 'progressive_load', 'load_data_progressively',
    'stream_data_progressively',
    
    # Integration flags
    'VECTORBT_INTEGRATION_AVAILABLE', 'M1_INTEGRATION_AVAILABLE'
]

# Add VectorBT integration if available
if VECTORBT_INTEGRATION_AVAILABLE:
    __all__.extend([
        'VectorBTPerformanceOptimizer', 'VectorBTCacheStrategy',
        'get_vectorbt_performance_optimizer', 'vectorbt_optimize',
        'vectorbt_cached', 'vectorbt_async_execute'
    ])

# Add M1 integration if available
if M1_INTEGRATION_AVAILABLE:
    __all__.extend([
        'M1PerformanceOptimizer', 'M1MemoryStrategy', 'M1CPUStrategy',
        'get_m1_performance_optimizer', 'm1_optimize', 'm1_cached',
        'm1_async_execute'
    ])