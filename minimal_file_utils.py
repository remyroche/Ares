"""
Minimal file utilities for code quality analysis.
"""

import os
from pathlib import Path
from typing import List, Pattern
import fnmatch
import re


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


def get_file_dependencies(file_path: str) -> List[str]:
    """
    Get the dependencies (imports) from a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        List of imported module names
    """
    dependencies = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple regex-based import extraction
        import re
        
        # Match import statements
        import_pattern = r'^\s*(?:from\s+(\S+)\s+)?import\s+([^#\n]+)'
        
        for line in content.split('\n'):
            match = re.match(import_pattern, line)
            if match:
                if match.group(1):  # from X import Y
                    dependencies.append(match.group(1).split('.')[0])
                else:  # import X
                    imports = match.group(2).split(',')
                    for imp in imports:
                        imp = imp.strip().split(' as ')[0].split('.')[0]
                        if imp:
                            dependencies.append(imp)
        
        # Remove duplicates
        dependencies = list(set(dependencies))
        
    except Exception:
        pass
    
    return dependencies