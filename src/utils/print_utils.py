"""
Print utilities for the Ares pipeline.

This module provides a centralized way to import timestamped print functions
throughout the codebase. Simply import from this module to get consistent
timestamped printing across all scripts.
"""

# Import all tprint functions
from .tprint import (
    tprint,
    tprint_debug,
    tprint_info, 
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_progress,
    tprint_performance,
    timestamped_print
)

# Re-export for easy importing
__all__ = [
    'tprint',
    'tprint_debug',
    'tprint_info',
    'tprint_warning', 
    'tprint_error',
    'tprint_success',
    'tprint_progress',
    'tprint_performance',
    'timestamped_print'
]

# Usage examples:
# from src.utils.print_utils import tprint, tprint_info, tprint_error
# 
# tprint("Regular message")           # [2025-09-11 06:30:15] Regular message
# tprint_info("Info message")         # [2025-09-11 06:30:15] INFO: Info message  
# tprint_error("Error message")       # [2025-09-11 06:30:15] ERROR: Error message
# tprint_progress(3, 10, "Processing") # [2025-09-11 06:30:15] PROGRESS: 3/10 (30.0%) Processing