"""
Optimization configuration management.

This module provides centralized configuration for all optimization settings
across the features_common system.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """
    Centralized optimization configuration.

    This class manages all optimization settings for the features_common system,
    providing a single source of truth for performance parameters.
    """

    # VectorBT optimization settings
    use_vectorbt: bool = True
    vectorbt_threshold: int = 100  # Lower threshold to use VectorBT more often
    enable_gpu: bool = False
    enable_parallel: bool = True
    memory_efficient: bool = True
    prefer_vectorbt: bool = True  # Prefer VectorBT over pandas when available

    # Performance optimization settings
    enable_caching: bool = True
    cache_size: int = 1000
    enable_batch_processing: bool = True
    batch_size: int = 10000

    # Memory optimization settings
    optimize_data_types: bool = True
    enable_memory_pooling: bool = True
    max_memory_usage: float = 0.8  # 80% of available memory

    # Adaptive optimization settings
    enable_adaptive_optimization: bool = True
    performance_threshold: float = 0.1  # 10% improvement threshold
    auto_tune_parameters: bool = True

    # Monitoring and profiling settings
    enable_performance_monitoring: bool = True
    enable_profiling: bool = False
    profile_threshold: float = 1.0  # Profile operations taking >1 second

    # Fallback settings
    enable_fallbacks: bool = True
    fallback_timeout: float = 30.0  # 30 seconds timeout for fallbacks

    # Advanced settings
    enable_experimental_features: bool = False
    debug_mode: bool = False
    verbose_logging: bool = False

    # Environment-specific overrides
    _env_overrides: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Apply environment-specific overrides after initialization."""
        self._apply_env_overrides()
        self._validate_config()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            'FEATURES_COMMON_USE_VECTORBT': ('use_vectorbt', bool),
            'FEATURES_COMMON_ENABLE_GPU': ('enable_gpu', bool),
            'FEATURES_COMMON_ENABLE_PARALLEL': ('enable_parallel', bool),
            'FEATURES_COMMON_MEMORY_EFFICIENT': ('memory_efficient', bool),
            'FEATURES_COMMON_ENABLE_CACHING': ('enable_caching', bool),
            'FEATURES_COMMON_CACHE_SIZE': ('cache_size', int),
            'FEATURES_COMMON_BATCH_SIZE': ('batch_size', int),
            'FEATURES_COMMON_VECTORBT_THRESHOLD': ('vectorbt_threshold', int),
            'FEATURES_COMMON_MAX_MEMORY_USAGE': ('max_memory_usage', float),
            'FEATURES_COMMON_DEBUG_MODE': ('debug_mode', bool),
            'FEATURES_COMMON_VERBOSE_LOGGING': ('verbose_logging', bool),
        }

        for env_var, (attr_name, type_func) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if type_func == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = type_func(env_value)
                    setattr(self, attr_name, value)
                    self._env_overrides[attr_name] = value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}={env_value}: {e}")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.vectorbt_threshold < 1:
            raise ValueError("vectorbt_threshold must be >= 1")

        if self.cache_size < 0:
            raise ValueError("cache_size must be >= 0")

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if not 0 < self.max_memory_usage <= 1:
            raise ValueError("max_memory_usage must be between 0 and 1")

        if not 0 <= self.performance_threshold <= 1:
            raise ValueError("performance_threshold must be between 0 and 1")

        if self.fallback_timeout <= 0:
            raise ValueError("fallback_timeout must be > 0")

    def get_vectorbt_settings(self) -> Dict[str, Any]:
        """Get VectorBT-specific settings."""
        return {
            'use_vectorbt': self.use_vectorbt,
            'vectorbt_threshold': self.vectorbt_threshold,
            'enable_gpu': self.enable_gpu,
            'enable_parallel': self.enable_parallel,
            'memory_efficient': self.memory_efficient,
        }

    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance optimization settings."""
        return {
            'enable_caching': self.enable_caching,
            'cache_size': self.cache_size,
            'enable_batch_processing': self.enable_batch_processing,
            'batch_size': self.batch_size,
            'optimize_data_types': self.optimize_data_types,
            'enable_memory_pooling': self.enable_memory_pooling,
            'max_memory_usage': self.max_memory_usage,
        }

    def get_adaptive_settings(self) -> Dict[str, Any]:
        """Get adaptive optimization settings."""
        return {
            'enable_adaptive_optimization': self.enable_adaptive_optimization,
            'performance_threshold': self.performance_threshold,
            'auto_tune_parameters': self.auto_tune_parameters,
        }

    def get_monitoring_settings(self) -> Dict[str, Any]:
        """Get monitoring and profiling settings."""
        return {
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_profiling': self.enable_profiling,
            'profile_threshold': self.profile_threshold,
        }

    def get_fallback_settings(self) -> Dict[str, Any]:
        """Get fallback settings."""
        return {
            'enable_fallbacks': self.enable_fallbacks,
            'fallback_timeout': self.fallback_timeout,
        }

    def get_debug_settings(self) -> Dict[str, Any]:
        """Get debug and experimental settings."""
        return {
            'enable_experimental_features': self.enable_experimental_features,
            'debug_mode': self.debug_mode,
            'verbose_logging': self.verbose_logging,
        }

    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        self._validate_config()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'use_vectorbt': self.use_vectorbt,
            'vectorbt_threshold': self.vectorbt_threshold,
            'enable_gpu': self.enable_gpu,
            'enable_parallel': self.enable_parallel,
            'memory_efficient': self.memory_efficient,
            'enable_caching': self.enable_caching,
            'cache_size': self.cache_size,
            'enable_batch_processing': self.enable_batch_processing,
            'batch_size': self.batch_size,
            'optimize_data_types': self.optimize_data_types,
            'enable_memory_pooling': self.enable_memory_pooling,
            'max_memory_usage': self.max_memory_usage,
            'enable_adaptive_optimization': self.enable_adaptive_optimization,
            'performance_threshold': self.performance_threshold,
            'auto_tune_parameters': self.auto_tune_parameters,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_profiling': self.enable_profiling,
            'profile_threshold': self.profile_threshold,
            'enable_fallbacks': self.enable_fallbacks,
            'fallback_timeout': self.fallback_timeout,
            'enable_experimental_features': self.enable_experimental_features,
            'debug_mode': self.debug_mode,
            'verbose_logging': self.verbose_logging,
            'env_overrides': self._env_overrides,
        }

    def copy(self) -> 'OptimizationConfig':
        """Create a copy of the configuration."""
        return OptimizationConfig(**self.to_dict())

# Global configuration instance
_global_config: Optional[OptimizationConfig] = None

def get_optimization_config() -> OptimizationConfig:
    """Get the global optimization configuration."""
    global _global_config
    if _global_config is None:
        _global_config = OptimizationConfig()
    return _global_config

def set_optimization_config(config: OptimizationConfig) -> None:
    """Set the global optimization configuration."""
    global _global_config
    _global_config = config

def reset_optimization_config() -> None:
    """Reset the global optimization configuration to defaults."""
    global _global_config
    _global_config = None
