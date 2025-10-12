"""
Centralized tprint utilities for feature lookback optimization.

This module provides a single source of truth for tprint functionality
across the entire feature lookback optimization module.
"""

# Import tprint for enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, 
        tprint_error, tprint_success, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    
    def tprint(*args, **kwargs):
        print("TPRINT:", *args, **kwargs)
    
    def tprint_debug(*args, **kwargs):
        print("DEBUG:", *args, **kwargs)
    
    def tprint_info(*args, **kwargs):
        print("INFO:", *args, **kwargs)
    
    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)
    
    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)
    
    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)
    
    def tprint_performance(*args, **kwargs):
        print("PERF:", *args, **kwargs)

__all__ = [
    'tprint', 'tprint_debug', 'tprint_info', 'tprint_warning',
    'tprint_error', 'tprint_success', 'tprint_performance',
    'TPRINT_AVAILABLE'
]