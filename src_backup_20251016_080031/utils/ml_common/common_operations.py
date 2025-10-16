"""
ML Common - Common Operations Module

This module provides common operation utilities for the ML Common package.
"""

# Re-export from parent common_operations
from ...utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory,
    safe_file_exists, create_fallback_logger
)

def get_ml_common_operations():
    """Get ML common operations for compatibility."""
    return {
        'safe_json_dump': safe_json_dump,
        'safe_json_load': safe_json_load,
        'ensure_directory': ensure_directory,
        'safe_file_exists': safe_file_exists,
        'create_fallback_logger': create_fallback_logger
    }

__all__ = [
    'safe_json_dump', 'safe_json_load', 'ensure_directory',
    'safe_file_exists', 'create_fallback_logger', 'get_ml_common_operations'
]
