"""
VectorBT configuration module for production use.

This module provides centralized configuration for VectorBT settings
optimized for the Ares trading system.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class VectorBTConfig:
    """Configuration for VectorBT production settings."""
    
    # Performance settings
    memory_efficient: bool = True
    parallel: bool = True
    validate_data: bool = True
    raise_on_warning: bool = False
    
    # Caching settings
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    
    # Memory management
    max_memory_usage: float = 0.8  # 80% of available memory
    chunk_size: int = 10000
    enable_gc: bool = True
    
    # Parallel processing
    num_threads: Optional[int] = None  # None = auto-detect
    enable_gpu: bool = False  # GPU acceleration if available
    
    # Data validation
    strict_validation: bool = True
    allow_nan: bool = False
    allow_inf: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_operations: bool = True
    log_performance: bool = True
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    fail_fast: bool = True
    
    # Production settings
    production_mode: bool = True
    debug_mode: bool = False
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.num_threads is not None and self.num_threads <= 0:
            raise ValueError("num_threads must be positive or None")
        
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        
        if not 0 < self.max_memory_usage <= 1:
            raise ValueError("max_memory_usage must be between 0 and 1")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

def get_vectorbt_config() -> VectorBTConfig:
    """
    Get VectorBT configuration from environment variables and defaults.
    
    Returns:
        VectorBTConfig: Configured settings
    """
    config = VectorBTConfig()
    
    # Override with environment variables
    if os.getenv("VECTORBT_MEMORY_EFFICIENT"):
        config.memory_efficient = os.getenv("VECTORBT_MEMORY_EFFICIENT").lower() == "true"
    
    if os.getenv("VECTORBT_PARALLEL"):
        config.parallel = os.getenv("VECTORBT_PARALLEL").lower() == "true"
    
    if os.getenv("VECTORBT_NUM_THREADS"):
        config.num_threads = int(os.getenv("VECTORBT_NUM_THREADS"))
    
    if os.getenv("VECTORBT_MAX_MEMORY"):
        config.max_memory_usage = float(os.getenv("VECTORBT_MAX_MEMORY"))
    
    if os.getenv("VECTORBT_CHUNK_SIZE"):
        config.chunk_size = int(os.getenv("VECTORBT_CHUNK_SIZE"))
    
    if os.getenv("VECTORBT_DEBUG"):
        config.debug_mode = os.getenv("VECTORBT_DEBUG").lower() == "true"
        if config.debug_mode:
            config.log_level = "DEBUG"
            config.log_operations = True
            config.log_performance = True
    
    if os.getenv("VECTORBT_PRODUCTION"):
        config.production_mode = os.getenv("VECTORBT_PRODUCTION").lower() == "true"
    
    return config

def configure_vectorbt(config: Optional[VectorBTConfig] = None) -> None:
    """
    Configure VectorBT with the given settings.
    
    Args:
        config: VectorBT configuration. If None, uses default config.
    """
    if config is None:
        config = get_vectorbt_config()
    
    try:
        import vectorbt as vbt
        from vectorbt.utils.config import configure
        
        # Configure VectorBT
        configure(
            array_wrapper=vbt.ArrayWrapper,
            caching=vbt.cached_method if config.enable_caching else None,
            memory_efficient=config.memory_efficient,
            parallel=config.parallel,
            validate_data=config.validate_data,
            raise_on_warning=config.raise_on_warning,
        )
        
        # Set additional settings
        vbt.settings['array_wrapper'] = vbt.ArrayWrapper
        vbt.settings['caching'] = vbt.cached_method if config.enable_caching else None
        vbt.settings['memory_efficient'] = config.memory_efficient
        vbt.settings['parallel'] = config.parallel
        vbt.settings['validate_data'] = config.validate_data
        vbt.settings['raise_on_warning'] = config.raise_on_warning
        
        # Configure logging
        if config.log_operations or config.log_performance:
            logging.basicConfig(level=getattr(logging, config.log_level.upper()))
        
        logger.info(f"VectorBT configured: memory_efficient={config.memory_efficient}, "
                   f"parallel={config.parallel}, validate_data={config.validate_data}")
        
    except ImportError as e:
        logger.error(f"Failed to configure VectorBT: {e}")
        raise
    except Exception as e:
        logger.error(f"VectorBT configuration error: {e}")
        raise

def get_optimal_chunk_size(data_size: int, available_memory: float = None) -> int:
    """
    Calculate optimal chunk size based on data size and available memory.
    
    Args:
        data_size: Size of the dataset
        available_memory: Available memory in GB. If None, auto-detects.
        
    Returns:
        int: Optimal chunk size
    """
    if available_memory is None:
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        except ImportError:
            available_memory = 8.0  # Default assumption
    
    # Use 50% of available memory for chunking
    memory_for_chunking = available_memory * 0.5
    
    # Estimate memory per data point (rough estimate)
    bytes_per_point = 8  # Assuming float64
    
    # Calculate optimal chunk size
    optimal_chunk_size = int(memory_for_chunking * 1024**3 / bytes_per_point)
    
    # Ensure reasonable bounds
    optimal_chunk_size = max(1000, min(optimal_chunk_size, data_size))
    
    return optimal_chunk_size

def validate_vectorbt_config(config: VectorBTConfig) -> bool:
    """
    Validate VectorBT configuration.
    
    Args:
        config: VectorBT configuration to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    errors = []
    
    if config.num_threads is not None and config.num_threads <= 0:
        errors.append("num_threads must be positive or None")
    
    if config.cache_size <= 0:
        errors.append("cache_size must be positive")
    
    if not 0 < config.max_memory_usage <= 1:
        errors.append("max_memory_usage must be between 0 and 1")
    
    if config.chunk_size <= 0:
        errors.append("chunk_size must be positive")
    
    if config.max_retries < 0:
        errors.append("max_retries must be non-negative")
    
    if config.retry_delay < 0:
        errors.append("retry_delay must be non-negative")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True

# Default configuration instance
DEFAULT_CONFIG = VectorBTConfig()

# Production configuration
PRODUCTION_CONFIG = VectorBTConfig(
    memory_efficient=True,
    parallel=True,
    validate_data=True,
    raise_on_warning=False,
    enable_caching=True,
    strict_validation=True,
    production_mode=True,
    debug_mode=False
)

# Development configuration
DEVELOPMENT_CONFIG = VectorBTConfig(
    memory_efficient=False,
    parallel=False,
    validate_data=True,
    raise_on_warning=True,
    enable_caching=False,
    strict_validation=False,
    production_mode=False,
    debug_mode=True,
    log_level="DEBUG",
    log_operations=True,
    log_performance=True
)