"""
Unified Matrix Operations for Ares Trading System

This module provides a single, optimized interface for all matrix and vector
operations, consolidating scattered implementations across the codebase while
retaining ALL existing capabilities.

Key Features:
- Single entry point for all matrix/vector operations
- GPU acceleration (MPS/CUDA) with automatic fallback
- Memory optimization and chunked processing
- Parallel processing and vectorization
- Cross-validation with matrix optimization
- M1 hardware optimization
- 100% backwards compatibility

Usage:
    from src.utils.ml_common.matrix_operations import AresOptimizer
    
    # Initialize once with automatic optimization
    optimizer = AresOptimizer()
    
    # All operations through single interface
    result = optimizer.matrix_multiply(A, B)
    correlation = optimizer.correlation_matrix(data)
    cv_results = optimizer.cross_validate(X, y, model_class)
    vectorized_data = optimizer.vectorize_features(data)
    
    # Legacy imports still work (with deprecation warnings)
    from src.utils.ml_common.matrix_operations import get_unified_matrix_operations
    ops = get_unified_matrix_operations()  # Returns AresOptimizer instance
"""

from .core_engine import AresOptimizer
from .configuration import UnifiedConfiguration
from .backwards_compatibility import *

# Version info
__version__ = "1.0.0"
__author__ = "Ares Trading System"

# Main exports
__all__ = [
    'AresOptimizer',
    'UnifiedConfiguration',
    'get_optimizer',
    # Legacy compatibility exports
    'get_unified_matrix_operations',
    'get_vectorized_processing_core',
    'get_enhanced_matrix_operations',
    'get_batch_matrix_processor',
    'matrix_cross_validate',
    'get_vectorized_processing_core',
]

# Convenience function for quick initialization
def get_optimizer(config: dict = None, optimization_target: str = "balanced") -> AresOptimizer:
    """
    Get optimized AresOptimizer instance with automatic configuration.
    
    Args:
        config: Optional configuration dictionary
        optimization_target: Optimization target ("performance", "memory", "accuracy", "balanced")
    
    Returns:
        Configured AresOptimizer instance
    """
    if config is None:
        config = UnifiedConfiguration.create_optimal_config(optimization_target)
    
    return AresOptimizer(config)

# Legacy compatibility functions (with deprecation warnings)
def get_unified_matrix_operations(**kwargs):
    """
    Legacy compatibility function for get_unified_matrix_operations().
    
    This function is deprecated but maintained for backwards compatibility.
    New code should use AresOptimizer() instead.
    """
    import warnings
    warnings.warn(
        "get_unified_matrix_operations() is deprecated and will be removed in a future version. "
        "Use AresOptimizer() instead for better performance and features.",
        DeprecationWarning,
        stacklevel=2
    )
    return AresOptimizer(kwargs)

def get_enhanced_matrix_operations(**kwargs):
    """
    Legacy compatibility function for get_enhanced_matrix_operations().
    
    This function is deprecated but maintained for backwards compatibility.
    New code should use AresOptimizer() instead.
    """
    import warnings
    warnings.warn(
        "get_enhanced_matrix_operations() is deprecated and will be removed in a future version. "
        "Use AresOptimizer() instead for better performance and features.",
        DeprecationWarning,
        stacklevel=2
    )
    return AresOptimizer(kwargs)

# Quick start examples
def quick_start_example():
    """Example showing how to use the unified matrix operations system."""
    
    import numpy as np
    
    # Initialize optimizer
    optimizer = AresOptimizer()
    
    # Create sample data
    A = np.random.randn(1000, 1000)
    B = np.random.randn(1000, 1000)
    data = np.random.randn(1000, 50)
    
    # Matrix operations
    result = optimizer.matrix_multiply(A, B)
    print(f"Matrix multiplication result shape: {result.shape}")
    
    # Correlation matrix
    corr = optimizer.correlation_matrix(data)
    print(f"Correlation matrix shape: {corr.shape}")
    
    # SVD decomposition
    U, s, V = optimizer.svd_decomposition(A, k=100)
    print(f"SVD components: U={U.shape}, s={s.shape}, V={V.shape}")
    
    # Get performance stats
    stats = optimizer.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    return optimizer

if __name__ == "__main__":
    # Run quick start example
    optimizer = quick_start_example()
    print("✅ Unified Matrix Operations system ready!")