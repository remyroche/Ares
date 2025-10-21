"""
Unified Vectorization Manager

Provides vectorization configuration and management.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorizationConfig:
    """Configuration for vectorization operations."""
    
    # Basic settings
    enable_vectorization: bool = True
    vectorization_method: str = "numpy"
    batch_size: int = 1000
    memory_limit_mb: int = 1000
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    enable_caching: bool = True
    cache_size_mb: int = 100
    
    # Optimization settings
    enable_optimization: bool = True
    optimization_level: str = "balanced"
    enable_compression: bool = False
    
    # Hardware settings
    enable_gpu: bool = False
    gpu_memory_limit_mb: int = 500
    enable_m1_optimizations: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")
        if self.vectorization_method not in ["numpy", "pandas", "custom"]:
            raise ValueError("vectorization_method must be one of: numpy, pandas, custom")


class UnifiedVectorizationManager:
    """Manager for unified vectorization operations."""
    
    def __init__(self, config: Optional[VectorizationConfig] = None):
        """Initialize the vectorization manager."""
        self.config = config or VectorizationConfig()
        self.logger = logger
        
    def vectorize_data(self, data: Any, **kwargs) -> Any:
        """
        Vectorize data using the configured method.
        
        Args:
            data: Data to vectorize
            **kwargs: Additional parameters
            
        Returns:
            Vectorized data
        """
        try:
            self.logger.info(f"Vectorizing data with method: {self.config.vectorization_method}")
            
            # Placeholder implementation
            if hasattr(data, 'values'):
                return data.values
            return data
            
        except Exception as e:
            self.logger.error(f"Vectorization failed: {e}")
            return data
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'vectorization_method': self.config.vectorization_method,
            'batch_size': self.config.batch_size,
            'memory_limit_mb': self.config.memory_limit_mb,
            'enable_parallel_processing': self.config.enable_parallel_processing
        }


def get_unified_vectorization_manager(config: Optional[VectorizationConfig] = None) -> UnifiedVectorizationManager:
    """
    Get a unified vectorization manager instance.
    
    Args:
        config: Optional configuration for the manager
        
    Returns:
        UnifiedVectorizationManager instance
    """
    return UnifiedVectorizationManager(config)


# Export the main classes and functions
__all__ = ['VectorizationConfig', 'UnifiedVectorizationManager', 'get_unified_vectorization_manager']