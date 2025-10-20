"""
Integrated Hardware Manager with Enhanced Caching and Optimization.

This module integrates the enhanced caching system with existing hardware utilities
to provide a unified, optimized hardware management solution.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

from .unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
)
from .enhanced_caching_system import (
    EnhancedCacheSystem, CacheConfig, DataTypeOptimization, CacheStrategy
)
from .optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    OptimizationConfig, OptimizationLevel as DecoratorOptimizationLevel
)
from .memory_optimized_decorators import (
    memory_optimized, gc_optimized, chunked_processing_auto,
    comprehensive_memory_optimization, MemoryOptimizationLevel,
    optimize_large_dataframes, optimize_large_arrays, optimize_memory_intensive
)
from .advanced_memory_manager import (
    get_advanced_memory_manager, MemoryConfig as AdvancedMemoryConfig
)
from .dynamic_memory_allocator import (
    get_dynamic_allocator, get_optimal_memory_allocation, WorkloadType,
    get_system_recommendations, update_memory_usage
)
from .m1_memory_optimizer import M1MemoryOptimizer
from .m1_cpu_optimizer import M1CPUOptimizer
from .enhanced_gpu_manager import EnhancedM1GPUManager

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_performance, LogLevel
)

logger = logging.getLogger(__name__)

@dataclass
class IntegratedHardwareConfig:
    """Configuration for integrated hardware management."""
    # Hardware configuration
    hardware_config: Optional[HardwareConfig] = None
    
    # Cache configuration
    cache_config: Optional[CacheConfig] = None
    
    # Optimization settings
    enable_automatic_optimization: bool = True
    enable_caching: bool = True
    enable_memory_monitoring: bool = True
    enable_performance_tracking: bool = True
    
    # Default optimization levels
    default_optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE
    default_cache_strategy: CacheStrategy = CacheStrategy.LRU
    
    # Memory management
    memory_limit_gb: float = 8.0
    cache_memory_limit_mb: float = 512.0
    
    # Performance thresholds
    performance_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.hardware_config is None:
            self.hardware_config = HardwareConfig(
                memory_limit_gb=self.memory_limit_gb,
                enable_adaptive_optimization=self.enable_automatic_optimization
            )
        
        if self.cache_config is None:
            self.cache_config = CacheConfig(
                max_memory_mb=self.cache_memory_limit_mb,
                strategy=self.default_cache_strategy,
                data_type_optimization=DataTypeOptimization.AGGRESSIVE,
                enable_compression=True,
                auto_optimize_dtypes=True
            )
        
        if self.performance_thresholds is None:
            self.performance_thresholds = {
                'cpu_usage': 85.0,
                'memory_usage': 90.0,
                'gpu_usage': 80.0,
                'cache_hit_rate': 0.7
            }

class IntegratedHardwareManager:
    """Integrated hardware manager with enhanced caching and optimization."""
    
    def __init__(self, config: Optional[IntegratedHardwareConfig] = None):
        self.config = config or IntegratedHardwareConfig()
        self.logger = logger.getChild('IntegratedHardwareManager')
        
        # Initialize hardware components
        self.hardware_manager = UnifiedHardwareManager(self.config.hardware_config)
        
        # Initialize enhanced caching system with dynamic allocation
        # Get optimal allocation based on system and workload
        allocation = get_optimal_memory_allocation(
            workload_type=WorkloadType.MODERATE,
            data_size_mb=None,
            user_preferences={'memory_usage_factor': 1.0}
        )
        
        # Create dynamic cache config
        dynamic_cache_config = CacheConfig(
            max_memory_mb=allocation.cache_memory_mb,
            strategy=CacheStrategy.LRU,
            data_type_optimization=DataTypeOptimization.AGGRESSIVE,
            enable_compression=True,
            auto_optimize_dtypes=True,
            prefer_int32=True,
            prefer_float32=True
        )
        
        self.cache_system = EnhancedCacheSystem(dynamic_cache_config)
        
        # Store allocation info for monitoring
        self.current_allocation = allocation
        self.dynamic_allocator = get_dynamic_allocator()
        
        # Initialize advanced memory manager
        memory_config = AdvancedMemoryConfig(
            enable_aggressive_gc=True,
            gc_threshold_mb=self.config.cache_memory_limit_mb * 0.8,
            enable_memory_pressure_detection=True,
            enable_chunking=True,
            default_chunk_size_mb=self.config.cache_memory_limit_mb * 0.1,
            enable_memory_pools=True,
            pool_size_mb=self.config.cache_memory_limit_mb * 0.2,
            enable_weak_references=True
        )
        self.advanced_memory_manager = get_advanced_memory_manager(memory_config)
        
        # Initialize individual optimizers
        self.memory_optimizer = M1MemoryOptimizer(self.config.memory_limit_gb)
        self.cpu_optimizer = M1CPUOptimizer()
        self.gpu_manager = EnhancedM1GPUManager()
        
        # Performance tracking
        self.performance_metrics = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'optimizations_applied': 0,
            'memory_savings_mb': 0.0,
            'average_execution_time': 0.0
        }
        
        # Initialize if not already done
        if not self.hardware_manager.is_initialized:
            self.hardware_manager.initialize()
        
        tprint_success("✅ Integrated Hardware Manager initialized")
        self.logger.info("Integrated hardware manager with caching initialized")
    
    @smart_cache(
        optimization_config=OptimizationConfig(
            enable_caching=True,
            enable_dtype_optimization=True,
            optimization_level=DecoratorOptimizationLevel.AGGRESSIVE,
            optimize_inputs=True,
            optimize_outputs=True
        )
    )
    def optimize_for_workload(self, workload_type: WorkloadType, 
                            optimization_level: Optional[OptimizationLevel] = None) -> bool:
        """Optimize hardware for specific workload with caching."""
        optimization_level = optimization_level or self.config.default_optimization_level
        
        # Check if we have cached optimization for this workload/level combination
        cache_key = f"workload_optimization:{workload_type.value}:{optimization_level.value}"
        
        # Use hardware manager's optimization
        result = self.hardware_manager.optimize_for_workload(workload_type, optimization_level)
        
        # Update performance metrics
        self.performance_metrics['total_operations'] += 1
        if result:
            self.performance_metrics['optimizations_applied'] += 1
        
        return result
    
    @comprehensive_memory_optimization(
        optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
        enable_caching=True,
        enable_chunking=True,
        enable_gc=True,
        enable_pools=True
    )
    def process_data_with_optimization(self, data: Any, 
                                     workload_type: WorkloadType = WorkloadType.GENERAL) -> Any:
        """Process data with automatic optimization and caching."""
        # Optimize for workload if not already done
        self.optimize_for_workload(workload_type)
        
        # Process data with appropriate optimizations
        if isinstance(data, dict):
            return self._process_dict_data(data, workload_type)
        elif hasattr(data, 'shape'):  # NumPy array or DataFrame
            return self._process_array_data(data, workload_type)
        else:
            return self._process_generic_data(data, workload_type)
    
    def _process_dict_data(self, data: Dict[str, Any], workload_type: WorkloadType) -> Dict[str, Any]:
        """Process dictionary data with optimization."""
        optimized_data = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                optimized_data[key] = self._process_dict_data(value, workload_type)
            else:
                optimized_data[key] = self._process_generic_data(value, workload_type)
        
        return optimized_data
    
    def _process_array_data(self, data: Any, workload_type: WorkloadType) -> Any:
        """Process array/DataFrame data with optimization."""
        import pandas as pd
        import numpy as np
        
        # Apply data type optimization
        if isinstance(data, pd.DataFrame):
            optimized_data, optimization_info = self.cache_system.data_type_optimizer.optimize_dataframe(data)
        elif isinstance(data, np.ndarray):
            optimized_data, optimization_info = self.cache_system.data_type_optimizer.optimize_numpy_array(data)
        else:
            optimized_data = data
            optimization_info = {}
        
        # Track memory savings
        if 'memory_saved_mb' in optimization_info:
            self.performance_metrics['memory_savings_mb'] += optimization_info['memory_saved_mb']
        
        return optimized_data
    
    def _process_generic_data(self, data: Any, workload_type: WorkloadType) -> Any:
        """Process generic data with optimization."""
        # For generic data, we mainly apply caching
        return data
    
    @performance_tracked(log_performance=True, track_memory=True)
    def execute_with_optimization(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with full optimization and performance tracking."""
        # Get workload type from function metadata if available
        workload_type = getattr(func, '_workload_type', WorkloadType.GENERAL)
        
        # Optimize for workload
        self.optimize_for_workload(workload_type)
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Update performance metrics
        self.performance_metrics['total_operations'] += 1
        
        return result
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        # Get hardware memory stats
        hardware_stats = self.hardware_manager.get_system_status()
        
        # Get cache stats
        cache_stats = self.cache_system.get_statistics()
        
        # Get memory optimizer stats
        memory_stats = self.memory_optimizer.get_memory_stats()
        
        # Get advanced memory manager stats
        advanced_memory_stats = self.advanced_memory_manager.get_detailed_memory_info()
        
        return {
            'hardware_memory': hardware_stats.get('memory_stats', {}),
            'cache_memory': cache_stats,
            'memory_optimizer': memory_stats,
            'advanced_memory': advanced_memory_stats,
            'performance_metrics': self.performance_metrics,
            'total_memory_usage_mb': (
                cache_stats.get('total_memory_used_mb', 0) +
                memory_stats.get('used_memory', 0) / (1024 * 1024) +
                advanced_memory_stats.get('memory_stats', {}).get('used_mb', 0)
            )
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            'hardware_status': self.hardware_manager.get_system_status(),
            'cache_statistics': self.cache_system.get_statistics(),
            'performance_metrics': self.performance_metrics,
            'optimization_config': {
                'hardware_config': self.config.hardware_config.__dict__,
                'cache_config': self.config.cache_config.__dict__
            }
        }
    
    def clear_all_caches(self):
        """Clear all caches and reset optimization state."""
        self.cache_system.clear()
        self.hardware_manager.reset_hardware_state()
        
        # Clear advanced memory manager
        self.advanced_memory_manager.cleanup_all()
        
        # Reset performance metrics
        self.performance_metrics = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'optimizations_applied': 0,
            'memory_savings_mb': 0.0,
            'average_execution_time': 0.0
        }
        
        tprint_info("All caches cleared and optimization state reset")
    
    def shutdown(self):
        """Shutdown all components."""
        self.cache_system.shutdown()
        self.hardware_manager.shutdown()
        tprint_info("Integrated hardware manager shutdown complete")

# Global instance
_global_integrated_manager: Optional[IntegratedHardwareManager] = None

def get_integrated_hardware_manager(
    config: Optional[IntegratedHardwareConfig] = None
) -> IntegratedHardwareManager:
    """Get or create the global integrated hardware manager."""
    global _global_integrated_manager
    
    if _global_integrated_manager is None:
        _global_integrated_manager = IntegratedHardwareManager(config)
    
    return _global_integrated_manager

def optimize_dataframe_default(df) -> Any:
    """Optimize DataFrame with default settings using integrated manager."""
    manager = get_integrated_hardware_manager()
    return manager.process_data_with_optimization(df, WorkloadType.DATA_PROCESSING)

def optimize_numpy_array_default(arr) -> Any:
    """Optimize NumPy array with default settings using integrated manager."""
    manager = get_integrated_hardware_manager()
    return manager.process_data_with_optimization(arr, WorkloadType.DATA_PROCESSING)

def cache_result_default(key_func: Optional[Callable] = None, ttl: Optional[float] = None):
    """Cache result with default settings using integrated manager."""
    manager = get_integrated_hardware_manager()
    return smart_cache(
        ttl=ttl,
        key_func=key_func,
        cache_config=manager.config.cache_config,
        optimization_config=OptimizationConfig(
            enable_caching=True,
            enable_dtype_optimization=True,
            optimization_level=DecoratorOptimizationLevel.AGGRESSIVE
        )
    )

def auto_optimize_default():
    """Auto-optimize with default settings using integrated manager."""
    return auto_optimize(
        optimization_level=DecoratorOptimizationLevel.AGGRESSIVE,
        optimize_inputs=True,
        optimize_outputs=True
    )

def memory_efficient_default():
    """Memory-efficient processing with default settings using integrated manager."""
    return memory_efficient(
        memory_threshold_mb=200.0,
        enable_compression=True,
        auto_cleanup=True
    )

# Convenience functions for common operations
def process_market_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process market data with full optimization."""
    manager = get_integrated_hardware_manager()
    return manager.process_data_with_optimization(data, WorkloadType.DATA_PROCESSING)

def process_ml_training_data(data: Any) -> Any:
    """Process ML training data with optimization."""
    manager = get_integrated_hardware_manager()
    return manager.process_data_with_optimization(data, WorkloadType.ML_TRAINING)

def process_backtesting_data(data: Any) -> Any:
    """Process backtesting data with optimization."""
    manager = get_integrated_hardware_manager()
    return manager.process_data_with_optimization(data, WorkloadType.BACKTESTING)

def get_system_optimization_status() -> Dict[str, Any]:
    """Get current system optimization status."""
    manager = get_integrated_hardware_manager()
    return manager.get_optimization_report()

def clear_optimization_caches():
    """Clear all optimization caches."""
    manager = get_integrated_hardware_manager()
    manager.clear_all_caches()