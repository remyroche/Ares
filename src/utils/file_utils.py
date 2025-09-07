"""
File utilities with passthrough functions for common operations.
"""

import os
from pathlib import Path
from typing import Any, Optional

def ensure_directory(path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_json_dump(data: Any, file_path: str, **kwargs) -> None:
    """Safely dump data to JSON file."""
    import json
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(data, f, **kwargs)

def safe_json_load(file_path: str, **kwargs) -> Any:
    """Safely load data from JSON file."""
    import json
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        return json.load(f, **kwargs)

def file_exists(file_path: str) -> bool:
    """Check if file exists."""
    return os.path.exists(file_path)

def directory_exists(dir_path: str) -> bool:
    """Check if directory exists."""
    return os.path.isdir(dir_path)

# Export all functions
__all__ = [
    'ensure_directory',
    'safe_json_dump',
    'safe_json_load',
    'file_exists',
    'directory_exists'
]
