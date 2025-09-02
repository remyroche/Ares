"""
Minimal file utilities for code quality analysis.
"""

import os
import ast
from pathlib import Path
from typing import List, Pattern, Dict
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
    """Extract import dependencies from a Python file.
    
    Returns a dictionary with keys:
    - 'imports': list of top-level modules imported via `import x`
    - 'from_imports': list of fully qualified names imported via `from x import y`
    - 'relative_imports': list of relative imports like `.module.name`
    """
    dependencies: Dict[str, List[str]] = {
        "imports": [],
        "from_imports": [],
        "relative_imports": []
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # store the module root (e.g., numpy from numpy.linalg)
                    dependencies["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if module.startswith("."):
                        dependencies["relative_imports"].append(f"{module}.{alias.name}")
                    else:
                        full_name = f"{module}.{alias.name}" if module else alias.name
                        dependencies["from_imports"].append(full_name)
    except Exception:
        # Fail silently; analyzers will handle empty deps
        pass
    return dependencies