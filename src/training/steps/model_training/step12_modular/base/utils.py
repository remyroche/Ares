from ..standardized_parquet_handler import standardized_parquet_handler
"""
Step 12 Modular: Base Utilities

This module contains shared utility functions for Step 12.
"""

import os
import json

from typing import Any, Dict, List, Union

from .imports import system_logger

def ensure_directory(directory_path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

def safe_json_dump(data: Dict[str, Any], file_path: str) -> bool:
    """Safely dump data to JSON file."""
    try:
        ensure_directory(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        system_logger.error(f"Failed to save JSON to {file_path}: {e}")
        return False

def error(msg: str) -> str:
    """Format error message."""
    return f'❌ {msg}'

def failed(msg: str) -> str:
    """Format failed message."""
    return f'💥 {msg}'

def timeout(msg: str) -> str:
    """Format timeout message."""
    return f'⏰ {msg}'

def warning(msg: str) -> str:
    """Format warning message."""
    return f'⚠️ {msg}'

def get_unified_data_loader(config: Dict[str, Any]) -> Union[object, Dict[str, Any]]:
    """Get a unified data loader instance."""
    class SimpleDataLoader:
        def __init__(self, config: Dict[str, Any]) -> None:
            self.config = config

        def get_performance_metrics(self) -> Dict[str, Any]:
            return {
                'memory_usage': {'percent': 50.0},
                'cache_stats': {'cache_size': 0, 'max_cache_size': 1000}
            }

    return SimpleDataLoader(config)

__all__ = [
    'ensure_directory',
    'safe_json_dump',
    'error',
    'failed',
    'timeout',
    'warning',
    'get_unified_data_loader'
]
