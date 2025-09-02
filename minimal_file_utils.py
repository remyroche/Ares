"""
Minimal file utilities for code quality analysis.
"""

import os
from pathlib import Path
from typing import List, Pattern, Dict
from code_quality.utils.file_utils import (
    get_file_dependencies as _cq_get_file_dependencies,
)
import fnmatch


def find_python_files(directory: str, exclude_patterns: List[str] = None) -> List[Path]:
    """
    Find all Python files in a directory, excluding specified patterns.
    
    Args:
        directory: Directory to search
        exclude_patterns: List of patterns to exclude
        
    Returns:
        List of Python file paths
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    python_files = []
    directory_path = Path(directory).resolve()
    
    for root, dirs, files in os.walk(directory_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not _should_exclude(d, exclude_patterns)]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                if not _should_exclude(str(file_path), exclude_patterns):
                    python_files.append(file_path)
    
    return python_files


def _should_exclude(path: str, exclude_patterns: List[str]) -> bool:
    """Check if a path should be excluded based on patterns."""
    path_str = str(path)
    
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(path_str, pattern):
            return True
        if pattern in path_str:
            return True
    
    return False


def get_file_dependencies(file_path: str) -> Dict[str, List[str]]:
    """Proxy to the full implementation used by code_quality.

    Some analyzers import this symbol from minimal_file_utils. Provide a thin
    wrapper that delegates to the canonical implementation to satisfy imports.
    """
    return _cq_get_file_dependencies(file_path)