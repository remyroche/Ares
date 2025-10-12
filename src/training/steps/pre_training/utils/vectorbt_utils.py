#!/usr/bin/env python3
"""
VectorBT Utility Functions

Consolidated VectorBT initialization and configuration utilities
to reduce code duplication across the final feature selection pipeline.
"""

from typing import Optional, Dict, Any
from src.utils.tprint import tprint, tprint_warning, tprint_error

# Import VectorBT optimization utilities
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    from src.feature_generation.utils.unified_vectorization_manager import (
        UnifiedVectorizationManager, get_unified_vectorization_manager, VectorizationConfig
    )
    VECTORBT_UTILS_AVAILABLE = True
except ImportError as e:
    VECTORBT_UTILS_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
    VectorizationConfig = None


class VectorBTConfig:
    """Standardized VectorBT configuration for feature selection."""
    
    def __init__(
        self,
        enable_gpu: bool = False,
        enable_parallel: bool = True,
        memory_efficient: bool = True,
        chunk_size: int = 1000,
        enable_monitoring: bool = True,
        batch_size: int = 10000,
        enable_batch_processing: bool = True
    ):
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        self.enable_monitoring = enable_monitoring
        self.batch_size = batch_size
        self.enable_batch_processing = enable_batch_processing


def create_vectorbt_optimizer(config: Optional[VectorBTConfig] = None) -> Optional[VectorBTRollingOptimizer]:
    """
    Create a VectorBT rolling optimizer with standardized configuration.
    
    Args:
        config: VectorBT configuration, uses defaults if None
        
    Returns:
        VectorBTRollingOptimizer instance or None if not available
    """
    if not VECTORBT_UTILS_AVAILABLE:
        tprint_warning("⚠️ VectorBT utilities not available")
        return None
    
    if config is None:
        config = VectorBTConfig()
    
    try:
        tprint("🔧 Creating VectorBT rolling optimizer")
        optimizer = get_vectorbt_rolling_optimizer(
            enable_gpu=config.enable_gpu,
            enable_parallel=config.enable_parallel,
            memory_efficient=config.memory_efficient,
            chunk_size=config.chunk_size
        )
        tprint("✅ VectorBT rolling optimizer created successfully")
        return optimizer
    except Exception as e:
        tprint_error(f"❌ Failed to create VectorBT rolling optimizer: {e}")
        return None


def create_vectorization_manager(config: Optional[VectorBTConfig] = None) -> Optional[UnifiedVectorizationManager]:
    """
    Create a unified vectorization manager with standardized configuration.
    
    Args:
        config: VectorBT configuration, uses defaults if None
        
    Returns:
        UnifiedVectorizationManager instance or None if not available
    """
    if not VECTORBT_UTILS_AVAILABLE:
        tprint_warning("⚠️ VectorBT utilities not available")
        return None
    
    if config is None:
        config = VectorBTConfig()
    
    try:
        tprint("🔧 Creating VectorBT vectorization manager")
        vectorization_config = VectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=config.enable_gpu,
            enable_parallel=config.enable_parallel,
            memory_efficient=config.memory_efficient,
            chunk_size=config.chunk_size,
            enable_monitoring=config.enable_monitoring,
            batch_size=config.batch_size,
            enable_batch_processing=config.enable_batch_processing
        )
        manager = get_unified_vectorization_manager(vectorization_config)
        tprint("✅ VectorBT vectorization manager created successfully")
        return manager
    except Exception as e:
        tprint_error(f"❌ Failed to create VectorBT vectorization manager: {e}")
        return None


def create_vectorbt_tools(config: Optional[VectorBTConfig] = None) -> Dict[str, Any]:
    """
    Create both VectorBT optimizer and vectorization manager.
    
    Args:
        config: VectorBT configuration, uses defaults if None
        
    Returns:
        Dictionary with 'optimizer' and 'manager' keys
    """
    tprint("🚀 Initializing VectorBT tools for feature selection")
    
    result = {
        'optimizer': None,
        'manager': None,
        'available': VECTORBT_UTILS_AVAILABLE
    }
    
    if not VECTORBT_UTILS_AVAILABLE:
        tprint_warning("⚠️ VectorBT utilities not available")
        return result
    
    # Create optimizer
    result['optimizer'] = create_vectorbt_optimizer(config)
    
    # Create vectorization manager
    result['manager'] = create_vectorization_manager(config)
    
    if result['optimizer'] and result['manager']:
        tprint("✅ VectorBT tools initialized successfully")
    else:
        tprint_warning("⚠️ Some VectorBT tools failed to initialize")
    
    return result


def get_vectorbt_performance_stats(optimizer: Optional[VectorBTRollingOptimizer] = None,
                                 manager: Optional[UnifiedVectorizationManager] = None) -> Dict[str, Any]:
    """
    Get comprehensive VectorBT performance statistics.
    
    Args:
        optimizer: VectorBT rolling optimizer instance
        manager: Unified vectorization manager instance
        
    Returns:
        Dictionary with performance statistics
    """
    stats = {}
    
    # Get VectorBT rolling optimizer stats
    if optimizer:
        try:
            optimizer_stats = optimizer.get_performance_stats()
            stats.update({
                'vectorbt_rolling_operations': optimizer_stats.get('vectorbt_operations', 0),
                'pandas_fallbacks': optimizer_stats.get('pandas_fallbacks', 0),
                'gpu_operations': optimizer_stats.get('gpu_operations', 0),
                'memory_optimizations': optimizer_stats.get('memory_optimizations', 0),
                'chunk_operations': optimizer_stats.get('chunk_operations', 0),
                'avg_time_per_operation': optimizer_stats.get('avg_time_per_operation', 0.0),
                'vectorbt_usage_rate': optimizer_stats.get('vectorbt_usage_rate', 0.0)
            })
        except Exception as e:
            tprint_warning(f"⚠️ Could not retrieve VectorBT optimizer stats: {e}")
    
    # Get unified vectorization manager stats
    if manager:
        try:
            manager_stats = manager.get_performance_stats()
            stats.update({
                'total_operations': manager_stats.get('total_operations', 0),
                'strategy_usage': manager_stats.get('strategy_usage', {}),
                'average_speedup': manager_stats.get('average_speedup', 0.0),
                'total_computation_time': manager_stats.get('total_computation_time', 0.0)
            })
        except Exception as e:
            tprint_warning(f"⚠️ Could not retrieve VectorBT manager stats: {e}")
    
    return stats