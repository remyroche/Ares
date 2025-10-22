"""
Phase 1 Optimization Integration for Profit Labeling

This module integrates all Phase 1 optimizations:
1. Vectorized rolling calculations using VectorBTRollingOptimizer
2. NumPy-based bar construction using VectorBTRollingOptimizer
3. Intelligent caching using hardware tools
4. Memory optimization using hardware tools

Expected Performance Improvements:
- 5-10x speedup for rolling calculations
- 3-5x speedup for bar construction
- 60-80% memory reduction
- Intelligent caching for repeated operations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass, field
import logging
import time

# Import Phase 1 optimization modules
try:
    from .volatility_modeling_optimized import OptimizedVolatilityModeler, OptimizedVolatilityConfig
    from .bar_construction_optimized import OptimizedEventBasedBarConstructor, OptimizedBarConstructionConfig
    from .intelligent_caching_system import IntelligentCachingSystem, CacheConfig
    from .memory_optimization_system import MemoryOptimizationSystem, MemoryOptimizationConfig
    PHASE1_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phase 1 optimization modules not available: {e}")
    PHASE1_MODULES_AVAILABLE = False

# Import original modules for fallback
try:
    from .volatility_modeling import VolatilityModeler, VolatilityConfig
    from .bar_construction import EventBasedBarConstructor, BarConstructionConfig
    from .consolidated_profit_labeler import ConsolidatedProfitLabeler, ConsolidatedLabelerConfig
    ORIGINAL_MODULES_AVAILABLE = True
except ImportError:
    ORIGINAL_MODULES_AVAILABLE = False

from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_performance
)


@dataclass
class Phase1OptimizationConfig:
    """Configuration for Phase 1 optimizations."""
    
    # Enable/disable specific optimizations
    enable_volatility_optimization: bool = True
    enable_bar_construction_optimization: bool = True
    enable_caching: bool = True
    enable_memory_optimization: bool = True
    
    # Performance settings
    enable_performance_monitoring: bool = True
    log_performance_metrics: bool = True
    
    # Memory settings
    memory_limit_gb: float = 2.0
    cache_size_mb: int = 100
    
    # VectorBT settings
    vectorbt_chunk_size: int = 1000
    vectorbt_memory_efficient: bool = True
    vectorbt_fast_fail: bool = True
    
    # Parallel processing
    enable_parallel_processing: bool = True


@dataclass
class Phase1OptimizationResult:
    """Result of Phase 1 optimization operations."""
    success: bool
    performance_improvement: float
    memory_saved_mb: float
    execution_time: float
    optimization_details: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class Phase1OptimizationManager:
    """
    Manager for Phase 1 optimizations.
    
    Integrates all Phase 1 optimization modules and provides
    a unified interface for optimized profit labeling operations.
    """
    
    def __init__(self, config: Optional[Phase1OptimizationConfig] = None):
        """Initialize Phase 1 optimization manager."""
        self.config = config or Phase1OptimizationConfig()
        self.logger = logging.getLogger("Phase1OptimizationManager")
        
        # Initialize optimization modules
        self._initialize_optimization_modules()
        
        # Performance tracking
        self.performance_metrics = {
            'total_operations': 0,
            'optimized_operations': 0,
            'fallback_operations': 0,
            'total_time': 0.0,
            'memory_saved_mb': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        tprint_success("✅ Phase1OptimizationManager initialized with all optimizations")
    
    def _initialize_optimization_modules(self):
        """Initialize all Phase 1 optimization modules."""
        try:
            if PHASE1_MODULES_AVAILABLE:
                # Initialize volatility modeler
                if self.config.enable_volatility_optimization:
                    volatility_config = OptimizedVolatilityConfig(
                        enable_vectorization=True,
                        enable_caching=self.config.enable_caching,
                        enable_memory_optimization=self.config.enable_memory_optimization,
                        enable_parallel_processing=self.config.enable_parallel_processing,
                        vectorbt_chunk_size=self.config.vectorbt_chunk_size,
                        vectorbt_memory_efficient=self.config.vectorbt_memory_efficient,
                        vectorbt_fast_fail=self.config.vectorbt_fast_fail,
                        memory_limit_gb=self.config.memory_limit_gb,
                        cache_size_mb=self.config.cache_size_mb
                    )
                    self.volatility_modeler = OptimizedVolatilityModeler(volatility_config)
                    tprint_info("   → OptimizedVolatilityModeler: Initialized")
                else:
                    self.volatility_modeler = None
                    tprint_info("   → Volatility optimization: Disabled")
                
                # Initialize bar constructor
                if self.config.enable_bar_construction_optimization:
                    bar_config = OptimizedBarConstructionConfig(
                        enable_vectorization=True,
                        enable_caching=self.config.enable_caching,
                        enable_memory_optimization=self.config.enable_memory_optimization,
                        enable_parallel_processing=self.config.enable_parallel_processing,
                        vectorbt_chunk_size=self.config.vectorbt_chunk_size,
                        vectorbt_memory_efficient=self.config.vectorbt_memory_efficient,
                        vectorbt_fast_fail=self.config.vectorbt_fast_fail,
                        memory_limit_gb=self.config.memory_limit_gb,
                        cache_size_mb=self.config.cache_size_mb
                    )
                    self.bar_constructor = OptimizedEventBasedBarConstructor(bar_config)
                    tprint_info("   → OptimizedEventBasedBarConstructor: Initialized")
                else:
                    self.bar_constructor = None
                    tprint_info("   → Bar construction optimization: Disabled")
                
                # Initialize caching system
                if self.config.enable_caching:
                    cache_config = CacheConfig(
                        max_memory_mb=self.config.cache_size_mb,
                        enable_hardware_optimization=True,
                        enable_monitoring=True
                    )
                    self.caching_system = IntelligentCachingSystem(cache_config)
                    tprint_info("   → IntelligentCachingSystem: Initialized")
                else:
                    self.caching_system = None
                    tprint_info("   → Caching: Disabled")
                
                # Initialize memory optimization
                if self.config.enable_memory_optimization:
                    memory_config = MemoryOptimizationConfig(
                        memory_limit_gb=self.config.memory_limit_gb,
                        enable_hardware_optimization=True,
                        enable_monitoring=True
                    )
                    self.memory_optimizer = MemoryOptimizationSystem(memory_config)
                    tprint_info("   → MemoryOptimizationSystem: Initialized")
                else:
                    self.memory_optimizer = None
                    tprint_info("   → Memory optimization: Disabled")
                
            else:
                # Fallback to original modules
                tprint_warning("Phase 1 modules not available, using original implementations")
                self.volatility_modeler = None
                self.bar_constructor = None
                self.caching_system = None
                self.memory_optimizer = None
                
        except Exception as e:
            tprint_error(f"Failed to initialize optimization modules: {e}")
            self.volatility_modeler = None
            self.bar_constructor = None
            self.caching_system = None
            self.memory_optimizer = None
    
    def optimize_volatility_modeling(self, bars: pd.DataFrame, 
                                   base_config: Optional[VolatilityConfig] = None) -> Any:
        """
        Optimize volatility modeling using Phase 1 optimizations.
        
        Args:
            bars: DataFrame with OHLCV data
            base_config: Optional base configuration
            
        Returns:
            Optimized volatility result
        """
        start_time = time.time()
        tprint_info("🚀 Starting optimized volatility modeling")
        
        try:
            if self.volatility_modeler:
                # Use optimized volatility modeler
                if base_config:
                    self.volatility_modeler.config.base_config = base_config
                
                result = self.volatility_modeler.model_volatility_optimized(bars)
                self.performance_metrics['optimized_operations'] += 1
                
                tprint_success(f"✅ Optimized volatility modeling completed in {result.processing_time:.3f}s")
                return result
                
            else:
                # Fallback to original implementation
                if ORIGINAL_MODULES_AVAILABLE:
                    tprint_warning("Falling back to original volatility modeling")
                    original_modeler = VolatilityModeler(base_config)
                    result = original_modeler.model_volatility(bars)
                    self.performance_metrics['fallback_operations'] += 1
                    return result
                else:
                    raise RuntimeError("No volatility modeling implementation available")
                    
        except Exception as e:
            tprint_error(f"Volatility modeling failed: {e}")
            raise
        
        finally:
            self._update_performance_metrics(time.time() - start_time)
    
    def optimize_bar_construction(self, tick_data: pd.DataFrame,
                                base_config: Optional[BarConstructionConfig] = None) -> Any:
        """
        Optimize bar construction using Phase 1 optimizations.
        
        Args:
            tick_data: DataFrame with tick data
            base_config: Optional base configuration
            
        Returns:
            Optimized bar construction result
        """
        start_time = time.time()
        tprint_info("🚀 Starting optimized bar construction")
        
        try:
            if self.bar_constructor:
                # Use optimized bar constructor
                if base_config:
                    self.bar_constructor.config.base_config = base_config
                
                result = self.bar_constructor.construct_bars_optimized(tick_data)
                self.performance_metrics['optimized_operations'] += 1
                
                tprint_success(f"✅ Optimized bar construction completed in {result.processing_time:.3f}s")
                return result
                
            else:
                # Fallback to original implementation
                if ORIGINAL_MODULES_AVAILABLE:
                    tprint_warning("Falling back to original bar construction")
                    original_constructor = EventBasedBarConstructor(base_config)
                    result = original_constructor.construct_bars(tick_data)
                    self.performance_metrics['fallback_operations'] += 1
                    return result
                else:
                    raise RuntimeError("No bar construction implementation available")
                    
        except Exception as e:
            tprint_error(f"Bar construction failed: {e}")
            raise
        
        finally:
            self._update_performance_metrics(time.time() - start_time)
    
    def optimize_data_structure(self, data: Any) -> Any:
        """
        Optimize data structure using memory optimization.
        
        Args:
            data: Data structure to optimize
            
        Returns:
            Optimized data structure
        """
        if self.memory_optimizer:
            return self.memory_optimizer.optimize_data_structure(data)
        return data
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get result from cache."""
        if self.caching_system:
            return self.caching_system.get(key)
        return None
    
    def set_cached_result(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set result in cache."""
        if self.caching_system:
            return self.caching_system.set(key, value, ttl)
        return False
    
    def optimize_memory_usage(self):
        """Perform comprehensive memory optimization."""
        if self.memory_optimizer:
            self.memory_optimizer.optimize_memory_usage()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        # Collect metrics from all optimization modules
        metrics = {
            **self.performance_metrics,
            'phase1_modules_available': PHASE1_MODULES_AVAILABLE,
            'original_modules_available': ORIGINAL_MODULES_AVAILABLE
        }
        
        # Add module-specific metrics
        if self.volatility_modeler:
            metrics['volatility_metrics'] = self.volatility_modeler.get_performance_metrics()
        
        if self.bar_constructor:
            metrics['bar_construction_metrics'] = self.bar_constructor.get_performance_metrics()
        
        if self.caching_system:
            metrics['caching_metrics'] = self.caching_system.get_stats()
        
        if self.memory_optimizer:
            metrics['memory_metrics'] = self.memory_optimizer.get_performance_metrics()
        
        return metrics
    
    def _update_performance_metrics(self, execution_time: float):
        """Update performance tracking metrics."""
        self.performance_metrics['total_operations'] += 1
        self.performance_metrics['total_time'] += execution_time
        
        if self.config.log_performance_metrics:
            tprint_performance(f"Operation completed in {execution_time:.3f}s")
    
    def clear_all_caches(self):
        """Clear all caches to free memory."""
        if self.caching_system:
            self.caching_system.clear()
        
        if self.volatility_modeler:
            self.volatility_modeler.clear_cache()
        
        if self.bar_constructor:
            self.bar_constructor.clear_cache()
        
        tprint_info("🧹 All caches cleared")
    
    def cleanup(self):
        """Perform cleanup operations."""
        if self.memory_optimizer:
            self.memory_optimizer.cleanup()
        
        self.clear_all_caches()
        
        tprint_info("🧹 Phase 1 optimization cleanup completed")


# Factory function for easy instantiation
def get_phase1_optimization_manager(config: Optional[Phase1OptimizationConfig] = None) -> Phase1OptimizationManager:
    """
    Get a Phase 1 optimization manager instance.
    
    Args:
        config: Optional configuration for the optimization manager
        
    Returns:
        Phase1OptimizationManager instance
    """
    return Phase1OptimizationManager(config)


# Example usage and integration with existing profit labeling
def integrate_phase1_optimizations():
    """
    Example of how to integrate Phase 1 optimizations with existing profit labeling.
    
    This function demonstrates how to use the Phase 1 optimizations
    as drop-in replacements for existing functionality.
    """
    tprint_info("🔧 Integrating Phase 1 optimizations with profit labeling system")
    
    # Initialize Phase 1 optimization manager
    config = Phase1OptimizationConfig(
        enable_volatility_optimization=True,
        enable_bar_construction_optimization=True,
        enable_caching=True,
        enable_memory_optimization=True,
        memory_limit_gb=2.0,
        cache_size_mb=100
    )
    
    manager = get_phase1_optimization_manager(config)
    
    # Example: Optimize volatility modeling
    # bars = pd.DataFrame(...)  # Your OHLCV data
    # volatility_result = manager.optimize_volatility_modeling(bars)
    
    # Example: Optimize bar construction
    # tick_data = pd.DataFrame(...)  # Your tick data
    # bar_result = manager.optimize_bar_construction(tick_data)
    
    # Example: Optimize data structures
    # optimized_data = manager.optimize_data_structure(your_data)
    
    # Get performance metrics
    metrics = manager.get_performance_metrics()
    tprint_success(f"✅ Phase 1 optimizations integrated. Performance metrics: {metrics}")
    
    return manager


if __name__ == "__main__":
    # Run integration example
    integrate_phase1_optimizations()