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
from .parallel_processing import ParallelProcessor
from src.utils.unified_cache import UnifiedCache, get_unified_cache, cached
from .thread_guard import limit_blas_threads, get_thread_info, validate_thread_environment
from .lookahead_protection import LookaheadProtection
from .base_safeguards import MLTrainingSafeguards
from .enhanced_error_handling import RobustErrorHandler

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