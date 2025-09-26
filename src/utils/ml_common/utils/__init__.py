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

from ..logger import setup_logger, get_logger
from .memory_optimization import MemoryEfficientTraining as MemoryOptimizer
from .memory_integration import MemoryIntegrator
from src.utils.parallel_processing_optimizer import ParallelProcessor
from src.utils.unified_cache import UnifiedCache, get_unified_cache, cached
from .thread_guard import limit_blas_threads, get_thread_info, validate_thread_environment
from .lookahead_protection import LookaheadProtection
from .base_safeguards import MLTrainingSafeguards
from .enhanced_error_handling import RobustErrorHandler
from .dependency_management import manager as dependency_manager, OptionalDependencyError
from .error_handling import (
    log_and_raise,
    log_and_return_default,
    suppress_with_logging,
    guard_execution,
)
from .performance import bounded_sleep

__all__ = [
    # Logging
    'setup_logger', 'get_logger',
    
    # Memory Management
    'MemoryOptimizer', 'MemoryIntegrator',
    
    # Parallel Processing
    'ParallelProcessor',
    
    # Caching
    'UnifiedCache', 'get_unified_cache', 'cached', 'SharedMLCache',
    
    # Threading
    'limit_blas_threads', 'get_thread_info', 'validate_thread_environment',
    
    # Protection
    'LookaheadProtection', 'MLTrainingSafeguards',

    # Error Handling
    'RobustErrorHandler', 'log_and_raise', 'log_and_return_default',
    'suppress_with_logging', 'guard_execution',

    # Optional dependencies
    'dependency_manager', 'OptionalDependencyError',

    # Performance safeguards
    'bounded_sleep'
]


def __getattr__(name: str):
    """Lazily import heavy submodules to avoid circular imports at import time."""
    if name == 'MemoryOptimizer':
        from .memory_optimization import MemoryEfficientTraining as _MemoryOptimizer
        return _MemoryOptimizer
    if name == 'MemoryIntegrator':
        # Provided by memory_integration for backward compatibility
        from .memory_integration import MemoryIntegrator as _MemoryIntegrator
        return _MemoryIntegrator
    if name == 'ParallelProcessor':
        # Resides in top-level utils
        from src.utils.parallel_processing_optimizer import ParallelProcessor as _ParallelProcessor
        return _ParallelProcessor
    if name == 'SharedMLCache':
        from .shared_cache import SharedMLCache as _SharedMLCache
        return _SharedMLCache
    if name in ('limit_blas_threads', 'get_thread_info', 'validate_thread_environment'):
        from .thread_guard import limit_blas_threads as _limit, get_thread_info as _info, validate_thread_environment as _validate
        return {
            'limit_blas_threads': _limit,
            'get_thread_info': _info,
            'validate_thread_environment': _validate,
        }[name]
    if name == 'LookaheadProtection':
        from .lookahead_protection import LookaheadProtection as _LookaheadProtection
        return _LookaheadProtection
    if name == 'MLTrainingSafeguards':
        from .base_safeguards import MLTrainingSafeguards as _MLTrainingSafeguards
        return _MLTrainingSafeguards
    if name == 'RobustErrorHandler':
        from .enhanced_error_handling import RobustErrorHandler as _RobustErrorHandler
        return _RobustErrorHandler
    if name == 'dependency_manager':
        from .dependency_management import manager as _manager
        return _manager
    if name == 'OptionalDependencyError':
        from .dependency_management import OptionalDependencyError as _OptionalDependencyError
        return _OptionalDependencyError
    if name in ('log_and_raise', 'log_and_return_default', 'suppress_with_logging', 'guard_execution'):
        from .error_handling import (
            log_and_raise as _log_and_raise,
            log_and_return_default as _log_and_return_default,
            suppress_with_logging as _suppress_with_logging,
            guard_execution as _guard_execution,
        )
        return {
            'log_and_raise': _log_and_raise,
            'log_and_return_default': _log_and_return_default,
            'suppress_with_logging': _suppress_with_logging,
            'guard_execution': _guard_execution,
        }[name]
    if name == 'bounded_sleep':
        from .performance import bounded_sleep as _bounded_sleep
        return _bounded_sleep
    raise AttributeError(f"module 'utils.ml_common.utils' has no attribute {name!r}")
