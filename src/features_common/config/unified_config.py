"""
Unified configuration management.

This module provides a unified configuration system that combines
optimization and VectorBT configurations for seamless integration.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

from .optimization_config import OptimizationConfig, get_optimization_config
from .vectorbt_config import VectorBTConfig, get_vectorbt_config

logger = logging.getLogger(__name__)

@dataclass
class UnifiedConfig:
    """
    Unified configuration combining all features_common settings.

    This class provides a single interface for managing all configuration
    settings across the features_common system.
    """

    optimization: OptimizationConfig
    vectorbt: VectorBTConfig

    def __init__(self,
                 optimization: Optional[OptimizationConfig] = None,
                 vectorbt: Optional[VectorBTConfig] = None):
        """
        Initialize unified configuration.

        Args:
            optimization: Optimization configuration (uses default if None)
            vectorbt: VectorBT configuration (uses default if None)
        """
        self.optimization = optimization or get_optimization_config()
        self.vectorbt = vectorbt or get_vectorbt_config()

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary."""
        return {
            'optimization': self.optimization.to_dict(),
            'vectorbt': self.vectorbt.to_dict(),
        }

    def get_optimized_settings(self) -> Dict[str, Any]:
        """Get optimized settings for maximum performance."""
        return {
            'use_vectorbt': self.optimization.use_vectorbt and self.vectorbt.enable_vectorbt,
            'enable_gpu': self.optimization.enable_gpu and self.vectorbt.enable_gpu,
            'enable_parallel': self.optimization.enable_parallel and self.vectorbt.enable_parallel_processing,
            'memory_efficient': self.optimization.memory_efficient and self.vectorbt.enable_memory_efficient,
            'enable_caching': self.optimization.enable_caching and self.vectorbt.enable_caching,
            'enable_batch_processing': self.optimization.enable_batch_processing,
            'data_size_threshold': self.vectorbt.data_size_threshold,
            'chunk_size': self.vectorbt.chunk_size,
            'optimization_level': self.vectorbt.optimization_level,
        }

    def should_use_vectorbt(self, data_size: int) -> bool:
        """Determine if VectorBT should be used for given data size."""
        return (self.optimization.use_vectorbt and
                self.vectorbt.should_use_vectorbt(data_size))

    def get_performance_multiplier(self) -> float:
        """Get overall performance multiplier based on all settings."""
        multiplier = 1.0

        # VectorBT optimization multiplier
        if self.vectorbt.enable_vectorbt:
            multiplier *= self.vectorbt.get_optimization_level_multiplier()

        # Parallel processing multiplier
        if self.optimization.enable_parallel and self.vectorbt.enable_parallel_processing:
            multiplier *= 1.5

        # Memory efficiency multiplier
        if self.optimization.memory_efficient and self.vectorbt.enable_memory_efficient:
            multiplier *= 1.2

        # Caching multiplier
        if self.optimization.enable_caching and self.vectorbt.enable_caching:
            multiplier *= 1.1

        return multiplier

    def update_optimization(self, **kwargs) -> None:
        """Update optimization configuration."""
        self.optimization.update(**kwargs)

    def update_vectorbt(self, **kwargs) -> None:
        """Update VectorBT configuration."""
        self.vectorbt.update(**kwargs)

    def update(self, optimization: Optional[Dict[str, Any]] = None,
               vectorbt: Optional[Dict[str, Any]] = None) -> None:
        """Update both configurations."""
        if optimization:
            self.update_optimization(**optimization)
        if vectorbt:
            self.update_vectorbt(**vectorbt)

    def copy(self) -> 'UnifiedConfig':
        """Create a copy of the unified configuration."""
        return UnifiedConfig(
            optimization=self.optimization.copy(),
            vectorbt=self.vectorbt.copy()
        )

    def validate(self) -> bool:
        """Validate the unified configuration."""
        try:
            # Validate individual configurations
            self.optimization._validate_config()
            self.vectorbt._validate_config()

            # Check for conflicts
            if (self.optimization.use_vectorbt and
                not self.vectorbt.enable_vectorbt):
                logger.warning("Optimization config enables VectorBT but VectorBT config disables it")

            if (self.optimization.enable_gpu and
                not self.vectorbt.enable_gpu):
                logger.warning("Optimization config enables GPU but VectorBT config disables it")

            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

# Global unified configuration instance
_global_unified_config: Optional[UnifiedConfig] = None

def get_unified_config() -> UnifiedConfig:
    """Get the global unified configuration."""
    global _global_unified_config
    if _global_unified_config is None:
        _global_unified_config = UnifiedConfig()
    return _global_unified_config

def set_unified_config(config: UnifiedConfig) -> None:
    """Set the global unified configuration."""
    global _global_unified_config
    _global_unified_config = config

def reset_unified_config() -> None:
    """Reset the global unified configuration to defaults."""
    global _global_unified_config
    _global_unified_config = None

def create_optimized_config() -> UnifiedConfig:
    """Create an optimized configuration for maximum performance."""
    optimization = OptimizationConfig(
        use_vectorbt=True,
        enable_gpu=True,
        enable_parallel=True,
        memory_efficient=True,
        enable_caching=True,
        enable_batch_processing=True,
        enable_adaptive_optimization=True,
        enable_performance_monitoring=True,
        optimization_level='aggressive'
    )

    vectorbt = VectorBTConfig(
        enable_vectorbt=True,
        enable_parallel_processing=True,
        enable_gpu=True,
        enable_auto_optimization=True,
        optimization_level='aggressive',
        enable_caching=True,
        enable_memory_pooling=True,
        enable_performance_monitoring=True
    )

    return UnifiedConfig(optimization=optimization, vectorbt=vectorbt)

def create_balanced_config() -> UnifiedConfig:
    """Create a balanced configuration for good performance and stability."""
    optimization = OptimizationConfig(
        use_vectorbt=True,
        enable_gpu=True,
        enable_parallel=True,
        memory_efficient=True,
        enable_caching=True,
        enable_batch_processing=True,
        enable_adaptive_optimization=True,
        enable_performance_monitoring=True,
        optimization_level='balanced'
    )

    vectorbt = VectorBTConfig(
        enable_vectorbt=True,
        enable_parallel_processing=True,
        enable_gpu=True,
        enable_auto_optimization=True,
        optimization_level='balanced',
        enable_caching=True,
        enable_memory_pooling=True,
        enable_performance_monitoring=True
    )

    return UnifiedConfig(optimization=optimization, vectorbt=vectorbt)

def create_conservative_config() -> UnifiedConfig:
    """Create a conservative configuration for maximum stability."""
    optimization = OptimizationConfig(
        use_vectorbt=True,
        enable_gpu=True,
        enable_parallel=False,
        memory_efficient=False,
        enable_caching=False,
        enable_batch_processing=False,
        enable_adaptive_optimization=False,
        enable_performance_monitoring=True,
        optimization_level='conservative'
    )

    vectorbt = VectorBTConfig(
        enable_vectorbt=True,
        enable_parallel_processing=False,
        enable_gpu=True,
        enable_auto_optimization=False,
        optimization_level='conservative',
        enable_caching=False,
        enable_memory_pooling=False,
        enable_performance_monitoring=True
    )

    return UnifiedConfig(optimization=optimization, vectorbt=vectorbt)
