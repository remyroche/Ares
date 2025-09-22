"""
ML Common - Utils Module

This module contains all utility functionality including:
- Logging utilities
- Memory optimization
- Parallel processing
- Caching
- Error handling
- Safeguards
"""

def __getattr__(name: str):
    """Lazily import heavy submodules to avoid circular imports at import time."""
    if name == 'setup_logger':
        from ..logger import setup_logger
        return setup_logger
    elif name == 'get_logger':
        from ..logger import get_logger
        return get_logger
    elif name == 'MemoryOptimizer':
        from .memory_optimization import MemoryEfficientTraining
        return MemoryEfficientTraining
    elif name == 'MemoryIntegrator':
        from .memory_integration import MemoryIntegrator
        return MemoryIntegrator
    elif name == 'UnifiedCache':
        from src.utils.unified_cache import UnifiedCache
        return UnifiedCache
    elif name == 'get_unified_cache':
        from src.utils.unified_cache import get_unified_cache
        return get_unified_cache
    elif name == 'cached':
        from src.utils.unified_cache import cached
        return cached
    elif name == 'limit_blas_threads':
        from .thread_guard import limit_blas_threads
        return limit_blas_threads
    elif name == 'get_thread_info':
        from .thread_guard import get_thread_info
        return get_thread_info
    elif name == 'validate_thread_environment':
        from .thread_guard import validate_thread_environment
        return validate_thread_environment
    elif name == 'LookaheadProtection':
        from .lookahead_protection import LookaheadProtection
        return LookaheadProtection
    elif name == 'MLTrainingSafeguards':
        from .base_safeguards import MLTrainingSafeguards
        return MLTrainingSafeguards
    elif name == 'RobustErrorHandler':
        from .enhanced_error_handling import RobustErrorHandler
        return RobustErrorHandler
    elif name == 'ParallelProcessor':
        from src.utils.parallel_processing_optimizer import ParallelProcessor
        return ParallelProcessor

    raise AttributeError(f"module 'utils.ml_common.utils' has no attribute {name!r}")

__all__ = [
    # Logging
    'setup_logger', 'get_logger',

    # Memory Management
    'MemoryOptimizer', 'MemoryIntegrator',

    # Parallel Processing
    'ParallelProcessor',

    # Caching
    'UnifiedCache', 'get_unified_cache', 'cached',

    # Threading
    'limit_blas_threads', 'get_thread_info', 'validate_thread_environment',

    # Protection
    'LookaheadProtection', 'MLTrainingSafeguards',

    # Error Handling
    'RobustErrorHandler'
]