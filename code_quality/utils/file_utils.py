"""
File utility functions for code quality tools.
"""

import os
import ast
import tokenize
from pathlib import Path
from typing import List, Set, Dict, Any, Optional, Tuple
from fnmatch import fnmatch


def find_python_files(directory: str, exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """
    Find all Python files in a directory recursively.
    
    Args:
        directory: Root directory to search
        exclude_patterns: Patterns to exclude (e.g., ['__pycache__', '*.pyc'])
    
    Returns:
        List of Python file paths
    """
    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", "*.pyc", ".git", "venv", "env"]
    
    python_files = []
    directory_path = Path(directory)
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not any(fnmatch(d, pattern) for pattern in exclude_patterns)]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if not any(fnmatch(file_path, pattern) for pattern in exclude_patterns):
                    python_files.append(file_path)
    
    return python_files


def is_valid_python_file(file_path: str) -> bool:
    """
    Check if a Python file has valid syntax.
    
    Args:
        file_path: Path to the Python file
    
    Returns:
        True if the file has valid Python syntax, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return False
    except Exception:
        return False


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a Python file.
    
    Args:
        file_path: Path to the Python file
    
    Returns:
        Dictionary containing file information
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic file stats
        file_info = {
            'path': file_path,
            'size': len(content),
            'lines': len(content.splitlines()),
            'valid_syntax': True,
            'encoding': 'utf-8'
        }
        
        # Parse AST for more detailed info
        try:
            tree = ast.parse(content)
            
            # Count different types of nodes
            node_counts = {
                'functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'imports': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
                'assignments': len([n for n in ast.walk(tree) if isinstance(n, ast.Assign)]),
                'calls': len([n for n in ast.walk(tree) if isinstance(n, ast.Call)])
            }
            
            file_info.update(node_counts)
            
        except SyntaxError as e:
            file_info['valid_syntax'] = False
            file_info['syntax_error'] = str(e)
        
        return file_info
        
    except Exception as e:
        return {
            'path': file_path,
            'error': str(e),
            'valid_syntax': False
        }


def get_directory_stats(directory: str, exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get statistics for all Python files in a directory.
    
    Args:
        directory: Directory to analyze
        exclude_patterns: Patterns to exclude
    
    Returns:
        Dictionary containing directory statistics
    """
    python_files = find_python_files(directory, exclude_patterns)
    
    stats = {
        'total_files': len(python_files),
        'valid_files': 0,
        'invalid_files': 0,
        'total_lines': 0,
        'total_size': 0,
        'total_functions': 0,
        'total_classes': 0,
        'total_imports': 0,
        'file_details': []
    }
    
    for file_path in python_files:
        file_info = get_file_info(file_path)
        stats['file_details'].append(file_info)
        
        if file_info.get('valid_syntax', False):
            stats['valid_files'] += 1
            stats['total_lines'] += file_info.get('lines', 0)
            stats['total_size'] += file_info.get('size', 0)
            stats['total_functions'] += file_info.get('functions', 0)
            stats['total_classes'] += file_info.get('classes', 0)
            stats['total_imports'] += file_info.get('imports', 0)
        else:
            stats['invalid_files'] += 1
    
    return stats


def backup_file(file_path: str, backup_suffix: str = ".backup") -> str:
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to the file to backup
        backup_suffix: Suffix for the backup file
    
    Returns:
        Path to the backup file
    """
    backup_path = file_path + backup_suffix
    try:
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        return backup_path
    except Exception as e:
        raise RuntimeError(f"Failed to create backup of {file_path}: {e}")


def restore_file(backup_path: str, original_path: str) -> None:
    """
    Restore a file from its backup.
    
    Args:
        backup_path: Path to the backup file
        original_path: Path where to restore the file
    """
    try:
        with open(backup_path, 'r', encoding='utf-8') as src:
            with open(original_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
    except Exception as e:
        raise RuntimeError(f"Failed to restore {original_path} from backup: {e}")


def get_file_dependencies(file_path: str) -> Dict[str, List[str]]:
    """
    Extract import dependencies from a Python file.
    
    Args:
        file_path: Path to the Python file
    
    Returns:
        Dictionary with 'imports' and 'from_imports' lists
    """
    dependencies = {
        'imports': [],
        'from_imports': [],
        'relative_imports': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies['imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    if module.startswith('.'):
                        dependencies['relative_imports'].append(f"{module}.{alias.name}")
                    else:
                        dependencies['from_imports'].append(f"{module}.{alias.name}")
        
    except Exception:
        pass
    
    return dependencies


def find_unused_imports(file_path: str) -> List[str]:
    """
    Find potentially unused imports in a Python file.
    
    Args:
        file_path: Path to the Python file
    
    Returns:
        List of potentially unused import names
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Get all imported names
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
        
        # Get all used names
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Handle attribute access (e.g., module.function)
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # Find unused imports
        unused = imported_names - used_names
        
        return list(unused)
        
    except Exception:
        return []