"""
VectorBT-specific configuration management.

This module provides centralized configuration for VectorBT optimization
settings and performance parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

@dataclass
class VectorBTConfig:
    """
    VectorBT-specific configuration.
    
    This class manages all VectorBT-related settings for optimal performance
    and compatibility across different environments.
    """
    
    # Core VectorBT settings
    enable_vectorbt: bool = True
    fallback_to_pandas: bool = True
    data_size_threshold: int = 100  # Lower threshold to use VectorBT more often
    
    # Performance settings
    enable_parallel_processing: bool = True
    num_threads: Optional[int] = None  # None = auto-detect
    enable_memory_efficient: bool = True
    chunk_size: int = 10000
    
    # GPU settings
    enable_gpu: bool = False
    gpu_memory_fraction: float = 0.8
    enable_gpu_fallback: bool = True
    
    # Optimization settings
    enable_auto_optimization: bool = True
    optimization_level: str = 'aggressive'  # 'conservative', 'balanced', 'aggressive'
    enable_vectorization: bool = True
    enable_compilation: bool = True
    prefer_vectorbt: bool = True  # Prefer VectorBT over pandas when available
    
    # Memory management
    enable_memory_pooling: bool = True
    memory_pool_size: int = 1000
    enable_garbage_collection: bool = True
    gc_frequency: int = 100  # Run GC every N operations
    
    # Caching settings
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: float = 3600.0  # 1 hour in seconds
    enable_disk_cache: bool = False
    disk_cache_path: Optional[str] = None
    
    # Error handling
    enable_error_recovery: bool = True
    max_retries: int = 3
    retry_delay: float = 0.1  # seconds
    enable_graceful_degradation: bool = True
    
    # Monitoring and profiling
    enable_performance_monitoring: bool = True
    enable_detailed_profiling: bool = False
    profile_memory_usage: bool = True
    profile_execution_time: bool = True
    
    # Advanced settings
    enable_experimental_features: bool = False
    enable_debug_mode: bool = False
    verbose_logging: bool = False
    
    # Method-specific settings
    rolling_window_optimization: bool = True
    scaling_optimization: bool = True
    batch_processing_optimization: bool = True
    
    # Environment-specific overrides
    _env_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Apply environment-specific overrides after initialization."""
        self._apply_env_overrides()
        self._validate_config()
        self._optimize_settings()
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            'VECTORBT_ENABLE': ('enable_vectorbt', bool),
            'VECTORBT_FALLBACK_PANDAS': ('fallback_to_pandas', bool),
            'VECTORBT_DATA_THRESHOLD': ('data_size_threshold', int),
            'VECTORBT_ENABLE_PARALLEL': ('enable_parallel_processing', bool),
            'VECTORBT_NUM_THREADS': ('num_threads', lambda x: int(x) if x else None),
            'VECTORBT_ENABLE_GPU': ('enable_gpu', bool),
            'VECTORBT_GPU_MEMORY_FRACTION': ('gpu_memory_fraction', float),
            'VECTORBT_OPTIMIZATION_LEVEL': ('optimization_level', str),
            'VECTORBT_ENABLE_CACHING': ('enable_caching', bool),
            'VECTORBT_CACHE_SIZE': ('cache_size', int),
            'VECTORBT_ENABLE_DEBUG': ('enable_debug_mode', bool),
            'VECTORBT_VERBOSE_LOGGING': ('verbose_logging', bool),
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
        if self.data_size_threshold < 1:
            raise ValueError("data_size_threshold must be >= 1")
        
        if self.num_threads is not None and self.num_threads < 1:
            raise ValueError("num_threads must be >= 1 or None")
        
        if not 0 < self.gpu_memory_fraction <= 1:
            raise ValueError("gpu_memory_fraction must be between 0 and 1")
        
        if self.optimization_level not in ['conservative', 'balanced', 'aggressive']:
            raise ValueError("optimization_level must be 'conservative', 'balanced', or 'aggressive'")
        
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        
        if self.cache_size < 0:
            raise ValueError("cache_size must be >= 0")
        
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be > 0")
        
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        
        if self.retry_delay <= 0:
            raise ValueError("retry_delay must be > 0")
    
    def _optimize_settings(self) -> None:
        """Optimize settings based on system capabilities."""
        import multiprocessing
        
        # Auto-detect optimal number of threads
        if self.num_threads is None:
            self.num_threads = min(multiprocessing.cpu_count(), 8)
        
        # Adjust chunk size based on available memory
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            # Use 1% of available memory for chunk size estimation
            estimated_chunk_size = max(1000, int(available_memory * 0.01 / 8))  # 8 bytes per float64
            self.chunk_size = min(self.chunk_size, estimated_chunk_size)
        except ImportError:
            pass  # psutil not available, use default
        
        # Set disk cache path if not specified
        if self.enable_disk_cache and self.disk_cache_path is None:
            import tempfile
            self.disk_cache_path = os.path.join(tempfile.gettempdir(), 'vectorbt_cache')
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related settings."""
        return {
            'enable_parallel_processing': self.enable_parallel_processing,
            'num_threads': self.num_threads,
            'enable_memory_efficient': self.enable_memory_efficient,
            'chunk_size': self.chunk_size,
            'enable_auto_optimization': self.enable_auto_optimization,
            'optimization_level': self.optimization_level,
            'enable_vectorization': self.enable_vectorization,
            'enable_compilation': self.enable_compilation,
        }
    
    def get_gpu_settings(self) -> Dict[str, Any]:
        """Get GPU-related settings."""
        return {
            'enable_gpu': self.enable_gpu,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'enable_gpu_fallback': self.enable_gpu_fallback,
        }
    
    def get_memory_settings(self) -> Dict[str, Any]:
        """Get memory management settings."""
        return {
            'enable_memory_pooling': self.enable_memory_pooling,
            'memory_pool_size': self.memory_pool_size,
            'enable_garbage_collection': self.enable_garbage_collection,
            'gc_frequency': self.gc_frequency,
        }
    
    def get_caching_settings(self) -> Dict[str, Any]:
        """Get caching settings."""
        return {
            'enable_caching': self.enable_caching,
            'cache_size': self.cache_size,
            'cache_ttl': self.cache_ttl,
            'enable_disk_cache': self.enable_disk_cache,
            'disk_cache_path': self.disk_cache_path,
        }
    
    def get_error_handling_settings(self) -> Dict[str, Any]:
        """Get error handling settings."""
        return {
            'enable_error_recovery': self.enable_error_recovery,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'enable_graceful_degradation': self.enable_graceful_degradation,
        }
    
    def get_monitoring_settings(self) -> Dict[str, Any]:
        """Get monitoring and profiling settings."""
        return {
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_detailed_profiling': self.enable_detailed_profiling,
            'profile_memory_usage': self.profile_memory_usage,
            'profile_execution_time': self.profile_execution_time,
        }
    
    def get_method_settings(self) -> Dict[str, Any]:
        """Get method-specific optimization settings."""
        return {
            'rolling_window_optimization': self.rolling_window_optimization,
            'scaling_optimization': self.scaling_optimization,
            'batch_processing_optimization': self.batch_processing_optimization,
        }
    
    def should_use_vectorbt(self, data_size: int) -> bool:
        """Determine if VectorBT should be used for given data size."""
        return (self.enable_vectorbt and 
                data_size >= self.data_size_threshold)
    
    def get_optimization_level_multiplier(self) -> float:
        """Get optimization level multiplier for performance tuning."""
        multipliers = {
            'conservative': 0.5,
            'balanced': 1.0,
            'aggressive': 2.0
        }
        return multipliers.get(self.optimization_level, 1.0)
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown VectorBT configuration parameter: {key}")
        
        self._validate_config()
        self._optimize_settings()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enable_vectorbt': self.enable_vectorbt,
            'fallback_to_pandas': self.fallback_to_pandas,
            'data_size_threshold': self.data_size_threshold,
            'enable_parallel_processing': self.enable_parallel_processing,
            'num_threads': self.num_threads,
            'enable_memory_efficient': self.enable_memory_efficient,
            'chunk_size': self.chunk_size,
            'enable_gpu': self.enable_gpu,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'enable_gpu_fallback': self.enable_gpu_fallback,
            'enable_auto_optimization': self.enable_auto_optimization,
            'optimization_level': self.optimization_level,
            'enable_vectorization': self.enable_vectorization,
            'enable_compilation': self.enable_compilation,
            'enable_memory_pooling': self.enable_memory_pooling,
            'memory_pool_size': self.memory_pool_size,
            'enable_garbage_collection': self.enable_garbage_collection,
            'gc_frequency': self.gc_frequency,
            'enable_caching': self.enable_caching,
            'cache_size': self.cache_size,
            'cache_ttl': self.cache_ttl,
            'enable_disk_cache': self.enable_disk_cache,
            'disk_cache_path': self.disk_cache_path,
            'enable_error_recovery': self.enable_error_recovery,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'enable_graceful_degradation': self.enable_graceful_degradation,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_detailed_profiling': self.enable_detailed_profiling,
            'profile_memory_usage': self.profile_memory_usage,
            'profile_execution_time': self.profile_execution_time,
            'enable_experimental_features': self.enable_experimental_features,
            'enable_debug_mode': self.enable_debug_mode,
            'verbose_logging': self.verbose_logging,
            'rolling_window_optimization': self.rolling_window_optimization,
            'scaling_optimization': self.scaling_optimization,
            'batch_processing_optimization': self.batch_processing_optimization,
            'env_overrides': self._env_overrides,
        }
    
    def copy(self) -> 'VectorBTConfig':
        """Create a copy of the configuration."""
        return VectorBTConfig(**self.to_dict())


# Global configuration instance
_global_config: Optional[VectorBTConfig] = None

def get_vectorbt_config() -> VectorBTConfig:
    """Get the global VectorBT configuration."""
    global _global_config
    if _global_config is None:
        _global_config = VectorBTConfig()
    return _global_config

def set_vectorbt_config(config: VectorBTConfig) -> None:
    """Set the global VectorBT configuration."""
    global _global_config
    _global_config = config

def reset_vectorbt_config() -> None:
    """Reset the global VectorBT configuration to defaults."""
    global _global_config
    _global_config = None