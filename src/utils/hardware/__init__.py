"""
Enhanced Hardware Utilities with Caching and Optimization.

This module provides a comprehensive hardware optimization and caching system
that automatically optimizes data types, implements LRU caching, and provides
memory-efficient operations throughout the codebase.
"""

from typing import Optional, Dict, Any

# Import core caching and optimization components
from .enhanced_caching_system import (
    EnhancedCacheSystem, CacheConfig, DataTypeOptimization, CacheStrategy,
    get_global_cache, optimize_dataframe_default, optimize_numpy_array_default
)

# Import advanced memory management
from .advanced_memory_manager import (
    AdvancedMemoryManager, MemoryConfig, MemoryPressureLevel, ChunkingStrategy,
    get_advanced_memory_manager, memory_efficient_processing, chunked_processing,
    track_memory_usage, force_garbage_collection, cleanup_all_memory
)

# Import dynamic memory allocation
from .dynamic_memory_allocator import (
    DynamicMemoryAllocator, SystemTier, WorkloadType, MemoryAllocation,
    get_dynamic_allocator, get_optimal_memory_allocation, get_system_recommendations,
    update_memory_usage
)

# Import memory-optimized decorators
from .memory_optimized_decorators import (
    memory_optimized, gc_optimized, chunked_processing_auto,
    comprehensive_memory_optimization, MemoryOptimizationLevel,
    optimize_large_dataframes, optimize_large_arrays, optimize_memory_intensive,
    optimize_streaming_processing, get_memory_optimization_stats
)

from .optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    cache_dataframe_result, cache_numpy_result, optimize_heavy_computation,
    memory_aware, optimize_all_dataframes, optimize_all_arrays,
    get_optimization_stats, clear_optimization_cache
)

from .integrated_hardware_manager import (
    IntegratedHardwareManager, IntegratedHardwareConfig,
    get_integrated_hardware_manager, process_market_data,
    process_ml_training_data, process_backtesting_data,
    get_system_optimization_status, clear_optimization_caches
)

from .optimization_patches import (
    apply_optimization_patches, remove_optimization_patches,
    optimize_dataframe_auto, optimize_numpy_array_auto, optimize_data_dict,
    cache_function_result, optimize_heavy_function, make_memory_efficient,
    auto_optimize_function, track_performance
)

# Import existing hardware utilities
from .unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType as UnifiedWorkloadType, OptimizationLevel,
    get_unified_hardware_manager, optimize_for_workload, get_system_status
)

from .m1_memory_optimizer import (
    M1MemoryOptimizer, get_m1_memory_optimizer, optimize_dataframe_memory,
    start_m1_memory_monitoring, stop_m1_memory_monitoring, get_memory_usage
)

from .m1_cpu_optimizer import (
    M1CPUOptimizer, get_m1_cpu_optimizer, optimize_function_for_m1,
    parallel_map_m1, create_m1_optimized_thread_pool, run_cpu_intensive_task
)

from .m1_gpu_utils import (
    M1GPUManager, get_m1_gpu_manager, is_m1_available, is_mps_available
)

from .enhanced_gpu_manager import (
    EnhancedM1GPUManager, get_enhanced_gpu_manager, GPUOperationType,
    GPUMemoryPool, BatchOperationConfig, create_gpu_operation, batch_gpu_operations
)

# Import comprehensive M1 enhancements
from .m1_unified_memory_manager import (
    M1UnifiedMemoryManager, UnifiedMemoryConfig, MemoryTier, MemoryAllocationStrategy,
    get_unified_memory_manager, optimize_for_unified_memory, allocate_unified_memory,
    get_unified_memory_stats, unified_memory_optimized, memory_tier_aware
)

from .m1_advanced_cpu_optimizer import (
    M1AdvancedCPUOptimizer, CPUConfig, CoreType, WorkloadType as CPUWorkloadType,
    ThermalState, get_advanced_cpu_optimizer, optimize_cpu_execution,
    parallel_cpu_execution, get_cpu_performance_metrics
)

from .m1_enhanced_gpu_manager import (
    M1EnhancedGPUManager, GPUConfig, GPUOperationType as EnhancedGPUOperationType,
    MemoryLayout, ComputePipeline, get_enhanced_gpu_manager, gpu_accelerated,
    get_gpu_performance_metrics
)

from .m1_neural_engine_manager import (
    M1NeuralEngineManager, NeuralEngineConfig, NeuralEngineOperation,
    ModelFormat, get_neural_engine_manager, neural_engine_optimized,
    get_neural_engine_metrics
)

from .m1_comprehensive_optimizer import (
    M1ComprehensiveOptimizer, ComprehensiveConfig, OptimizationStrategy,
    WorkloadCategory, OptimizationResult, get_comprehensive_optimizer,
    m1_optimized, get_m1_comprehensive_metrics
)

# Import enhanced hardware optimizations
from .vectorbt_gpu_accelerator import (
    VectorBTGPUAccelerator, VectorBTGPUConfig, VectorBTOperationType,
    get_vectorbt_gpu_accelerator, gpu_vectorbt_optimization,
    get_vectorbt_gpu_performance_metrics, gpu_accelerated_vectorbt
)

from .enhanced_cpu_optimizer import (
    EnhancedCPUOptimizer, EnhancedCPUConfig, CPUIntensity, PowerMode,
    get_enhanced_cpu_optimizer, cpu_optimized_feature_correlation,
    get_enhanced_cpu_performance_metrics, optimize_cpu_execution_enhanced
)

from .enhanced_unified_memory_manager import (
    EnhancedUnifiedMemoryManager, EnhancedUnifiedMemoryConfig, MemoryComponent,
    MemoryAccessPattern, MemoryCompressionType, get_enhanced_unified_memory_manager,
    unified_memory_feature_processing, get_enhanced_unified_memory_stats
)

from .adaptive_optimization_engine import (
    AdaptiveOptimizationEngine, OptimizationStrategy, WorkloadCategory,
    get_adaptive_optimization_engine, adaptive_feature_selection,
    get_adaptive_optimization_metrics
)

# Import backward compatibility layer
from .backward_compatibility import (
    get_hardware_optimization_manager, get_hardware_optimization_status,
    clear_optimization_caches, initialize_optimization_system,
    gpu_accelerated, optimize_cpu_execution, unified_memory_optimized,
    adaptive_optimization, smart_cache, performance_tracked,
    comprehensive_memory_optimization
)

# Import examples
from .optimization_examples import run_all_examples

# Version information
__version__ = "2.0.0"
__author__ = "Ares Trading System"
__description__ = "Enhanced Hardware Utilities with Caching and Optimization"

# Default configurations with intelligent dynamic memory allocation
def get_dynamic_cache_config(workload_type: WorkloadType = WorkloadType.MODERATE,
                           data_size_mb: Optional[float] = None,
                           user_preferences: Optional[Dict[str, Any]] = None) -> CacheConfig:
    """Get cache configuration with intelligent dynamic memory allocation."""
    # Get optimal memory allocation
    allocation = get_optimal_memory_allocation(workload_type, data_size_mb, user_preferences)
    
    return CacheConfig(
        max_memory_mb=allocation.cache_memory_mb,
        strategy=CacheStrategy.LRU,
        data_type_optimization=DataTypeOptimization.AGGRESSIVE,
        enable_compression=True,
        auto_optimize_dtypes=True,
        prefer_int32=True,
        prefer_float32=True
    )

# Default configuration for moderate workloads
DEFAULT_CACHE_CONFIG = get_dynamic_cache_config()

DEFAULT_HARDWARE_CONFIG = HardwareConfig(
    memory_limit_gb=8.0,
    enable_adaptive_optimization=True,
    performance_monitoring_enabled=True
)

# Convenience functions for common operations
def optimize_dataframe(df):
    """Optimize DataFrame with default settings."""
    return optimize_dataframe_default(df)

def optimize_array(arr):
    """Optimize NumPy array with default settings."""
    return optimize_numpy_array_default(arr)

def cache_result(ttl=None, key_func=None):
    """Cache function result with default settings."""
    return smart_cache(ttl=ttl, key_func=key_func)

def optimize_function(optimize_inputs=True, optimize_outputs=True):
    """Apply automatic optimization to function."""
    return auto_optimize(optimize_inputs=optimize_inputs, optimize_outputs=optimize_outputs)

def make_efficient():
    """Make function memory efficient."""
    return memory_efficient()

def track_perf():
    """Track function performance."""
    return performance_tracked()

# Advanced memory management functions
def optimize_with_gc(df):
    """Optimize DataFrame with garbage collection."""
    from .memory_optimized_decorators import optimize_dataframe_with_gc
    return optimize_dataframe_with_gc(df)

def optimize_array_with_gc(arr):
    """Optimize NumPy array with garbage collection."""
    from .memory_optimized_decorators import optimize_array_with_gc
    return optimize_array_with_gc(arr)

def memory_optimized_function(level='aggressive'):
    """Apply memory optimization to function."""
    return memory_optimized(optimization_level=MemoryOptimizationLevel[level.upper()])

def chunked_function(chunk_size_mb=50.0):
    """Apply chunked processing to function."""
    return chunked_processing_auto(chunk_size_mb=chunk_size_mb)

def gc_optimized_function():
    """Apply garbage collection optimization to function."""
    return gc_optimized()

def comprehensive_optimization():
    """Apply comprehensive memory optimization to function."""
    return comprehensive_memory_optimization()

def force_cleanup():
    """Force garbage collection and memory cleanup."""
    force_garbage_collection()
    cleanup_all_memory()

def get_memory_stats():
    """Get comprehensive memory statistics."""
    return get_memory_optimization_stats()

# Global initialization
def initialize_optimization_system():
    """Initialize the optimization system with default settings."""
    # Apply patches to existing code
    apply_optimization_patches()
    
    # Initialize integrated hardware manager
    manager = get_integrated_hardware_manager()
    
    # Get cache system
    cache = get_global_cache(DEFAULT_CACHE_CONFIG)
    
    return manager, cache

def get_optimization_status():
    """Get current optimization system status."""
    manager = get_integrated_hardware_manager()
    return manager.get_optimization_report()

def clear_all_caches():
    """Clear all optimization caches."""
    manager = get_integrated_hardware_manager()
    manager.clear_all_caches()

# Auto-initialize on import
try:
    _manager, _cache = initialize_optimization_system()
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to initialize optimization system: {e}")

# Export main classes and functions
__all__ = [
    # Core caching and optimization
    'EnhancedCacheSystem', 'CacheConfig', 'DataTypeOptimization', 'CacheStrategy',
    'get_global_cache', 'optimize_dataframe_default', 'optimize_numpy_array_default',
    
    # Advanced memory management
    'AdvancedMemoryManager', 'MemoryConfig', 'MemoryPressureLevel', 'ChunkingStrategy',
    'get_advanced_memory_manager', 'memory_efficient_processing', 'chunked_processing',
    'track_memory_usage', 'force_garbage_collection', 'cleanup_all_memory',
    
    # Memory-optimized decorators
    'memory_optimized', 'gc_optimized', 'chunked_processing_auto',
    'comprehensive_memory_optimization', 'MemoryOptimizationLevel',
    'optimize_large_dataframes', 'optimize_large_arrays', 'optimize_memory_intensive',
    'optimize_streaming_processing', 'get_memory_optimization_stats',
    
    # Decorators
    'smart_cache', 'auto_optimize', 'memory_efficient', 'performance_tracked',
    'cache_dataframe_result', 'cache_numpy_result', 'optimize_heavy_computation',
    'memory_aware', 'optimize_all_dataframes', 'optimize_all_arrays',
    
    # Integrated management
    'IntegratedHardwareManager', 'IntegratedHardwareConfig',
    'get_integrated_hardware_manager', 'process_market_data',
    'process_ml_training_data', 'process_backtesting_data',
    
    # Hardware utilities
    'UnifiedHardwareManager', 'HardwareConfig', 'UnifiedWorkloadType', 'OptimizationLevel',
    'M1MemoryOptimizer', 'M1CPUOptimizer', 'M1GPUManager', 'EnhancedM1GPUManager',
    
    # Comprehensive M1 enhancements
    'M1UnifiedMemoryManager', 'UnifiedMemoryConfig', 'MemoryTier', 'MemoryAllocationStrategy',
    'get_unified_memory_manager', 'optimize_for_unified_memory', 'allocate_unified_memory',
    'get_unified_memory_stats', 'unified_memory_optimized', 'memory_tier_aware',
    
    'M1AdvancedCPUOptimizer', 'CPUConfig', 'CoreType', 'CPUWorkloadType',
    'ThermalState', 'get_advanced_cpu_optimizer', 'optimize_cpu_execution',
    'parallel_cpu_execution', 'get_cpu_performance_metrics',
    
    'M1EnhancedGPUManager', 'GPUConfig', 'EnhancedGPUOperationType',
    'MemoryLayout', 'ComputePipeline', 'get_enhanced_gpu_manager', 'gpu_accelerated',
    'get_gpu_performance_metrics',
    
    'M1NeuralEngineManager', 'NeuralEngineConfig', 'NeuralEngineOperation',
    'ModelFormat', 'get_neural_engine_manager', 'neural_engine_optimized',
    'get_neural_engine_metrics',
    
    'M1ComprehensiveOptimizer', 'ComprehensiveConfig', 'OptimizationStrategy',
    'WorkloadCategory', 'OptimizationResult', 'get_comprehensive_optimizer',
    'm1_optimized', 'get_m1_comprehensive_metrics',
    
    # Enhanced hardware optimizations
    'VectorBTGPUAccelerator', 'VectorBTGPUConfig', 'VectorBTOperationType',
    'get_vectorbt_gpu_accelerator', 'gpu_vectorbt_optimization',
    'get_vectorbt_gpu_performance_metrics', 'gpu_accelerated_vectorbt',
    
    'EnhancedCPUOptimizer', 'EnhancedCPUConfig', 'CPUIntensity', 'PowerMode',
    'get_enhanced_cpu_optimizer', 'cpu_optimized_feature_correlation',
    'get_enhanced_cpu_performance_metrics', 'optimize_cpu_execution_enhanced',
    
    'EnhancedUnifiedMemoryManager', 'EnhancedUnifiedMemoryConfig', 'MemoryComponent',
    'MemoryAccessPattern', 'MemoryCompressionType', 'get_enhanced_unified_memory_manager',
    'unified_memory_feature_processing', 'get_enhanced_unified_memory_stats',
    
    'AdaptiveOptimizationEngine', 'get_adaptive_optimization_engine',
    'adaptive_feature_selection', 'get_adaptive_optimization_metrics',
    
    # Backward compatibility
    'get_hardware_optimization_manager', 'get_hardware_optimization_status',
    'clear_optimization_caches', 'initialize_optimization_system',
    
    # Convenience functions
    'optimize_dataframe', 'optimize_array', 'cache_result', 'optimize_function',
    'make_efficient', 'track_perf', 'get_optimization_status', 'clear_all_caches',
    'optimize_with_gc', 'optimize_array_with_gc', 'memory_optimized_function',
    'chunked_function', 'gc_optimized_function', 'comprehensive_optimization',
    'force_cleanup', 'get_memory_stats',
    
    # Utilities
    'apply_optimization_patches', 'remove_optimization_patches',
    'run_all_examples', 'initialize_optimization_system'
]