#!/usr/bin/env python3
"""File utility functions for code analysis."""

import ast
import re
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any


def find_python_files(directory: str, exclude_dirs: List[str] = None) -> List[Path]:
    """Find all Python files in directory, excluding specified directories."""
    if exclude_dirs is None:
        exclude_dirs = ["venv", "__pycache__", ".git", "node_modules", ".pytest_cache"]
    
    project_root = Path(directory)
    python_files = []
    
    for py_file in project_root.rglob("*.py"):
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue
        python_files.append(py_file)
    
    return python_files


def read_file_safely(file_path: Path) -> Optional[str]:
    """Read a file with encoding detection and error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception:
            return None
    except Exception:
        return None


def parse_ast_safely(content: str, file_path: Path) -> Optional[ast.AST]:
    """Parse AST with error handling."""
    try:
        return ast.parse(content, filename=str(file_path))
    except SyntaxError:
        return None
    except Exception:
        return None


def extract_function_name_from_issue(issue) -> Optional[str]:
    """Extract function name from an issue object."""
    if hasattr(issue, 'description'):
        patterns = [
            r"unused function '([^']+)'",
            r"function '([^']+)'",
            r"deprecated ([a-zA-Z_][a-zA-Z0-9_]*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, issue.description.lower())
            if match:
                return match.group(1)
    return None


def get_module_from_file_path(file_path: str) -> Optional[str]:
    """Extract module name from file path."""
    try:
        path_parts = Path(file_path).parts
        if 'workspace' in path_parts:
            workspace_idx = path_parts.index('workspace')
            module_parts = path_parts[workspace_idx + 1:]
            if module_parts[-1].endswith('.py'):
                module_parts[-1] = module_parts[-1][:-3]
            return '.'.join(module_parts)
    except Exception:
        pass
    return None


def is_documentation_file(file_path: str) -> bool:
    """Check if a file is a documentation or config file."""
    doc_extensions = {'.md', '.rst', '.txt', '.yaml', '.yml', '.json', '.toml', '.ini', '.cfg'}
    config_keywords = ['config', 'settings', 'example', 'demo', 'test']
    
    file_path_str = str(file_path)
    
    # Check file extension
    if any(file_path_str.endswith(ext) for ext in doc_extensions):
        return True
    
    # Check if it's in a config-related directory
    if any(keyword in file_path_str.lower() for keyword in config_keywords):
        return True
    
    return False


def is_valid_python_file(file_path: Path) -> bool:
    """Check if a file is a valid Python file."""
    try:
        if not file_path.suffix == '.py':
            return False
        
        content = read_file_safely(file_path)
        if content is None:
            return False
        
        # Try to parse the AST to check for syntax errors
        ast.parse(content, filename=str(file_path))
        return True
    except (SyntaxError, UnicodeDecodeError, Exception):
        return False


def backup_file(file_path: Path) -> Optional[Path]:
    """Create a backup of a file."""
    try:
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception:
        return None


def restore_file(file_path: Path, backup_path: Path) -> bool:
    """Restore a file from backup."""
    try:
        shutil.copy2(backup_path, file_path)
        return True
    except Exception:
        return False


def find_unused_imports(file_path: Path) -> List[str]:
    """Find unused imports in a Python file."""
    try:
        content = read_file_safely(file_path)
        if content is None:
            return []
        
        tree = ast.parse(content, filename=str(file_path))
        imports = []
        used_names = set()
        
        # Collect all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        # Collect all used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Handle attribute access like module.function
                if isinstance(node.value, ast.Name):
                    used_names.add(f"{node.value.id}.{node.attr}")
        
        # Find unused imports
        unused = []
        for imp in imports:
            if imp not in used_names and imp.split('.')[-1] not in used_names:
                unused.append(imp)
        
        return unused
    except Exception:
        return []


class FileUtils:
    """Utility functions for file operations."""
    
    @staticmethod
    def find_python_files_static(directory: str, exclude_dirs: List[str] = None) -> List[Path]:
        """Find all Python files in directory, excluding specified directories."""
        if exclude_dirs is None:
            exclude_dirs = ["venv", "__pycache__", ".git", "node_modules", ".pytest_cache"]
        
        project_root = Path(directory)
        python_files = []
        
        for py_file in project_root.rglob("*.py"):
            if any(excluded in py_file.parts for excluded in exclude_dirs):
                continue
            python_files.append(py_file)
        
        return python_files
    
    @staticmethod
    def read_file_safely(file_path: Path) -> Optional[str]:
        """Read a file with encoding detection and error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception:
                return None
        except Exception:
            return None
    
    @staticmethod
    def parse_ast_safely(content: str, file_path: Path) -> Optional[ast.AST]:
        """Parse AST with error handling."""
        try:
            return ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return None
        except Exception:
            return None
    
    @staticmethod
    def extract_function_name_from_issue(issue) -> Optional[str]:
        """Extract function name from an issue object."""
        if hasattr(issue, 'description'):
            patterns = [
                r"unused function '([^']+)'",
                r"function '([^']+)'",
                r"deprecated ([a-zA-Z_][a-zA-Z0-9_]*)",
            ]
            for pattern in patterns:
                match = re.search(pattern, issue.description.lower())
                if match:
                    return match.group(1)
        return None
    
    @staticmethod
    def get_module_from_file_path(file_path: str) -> Optional[str]:
        """Extract module name from file path."""
        try:
            path_parts = Path(file_path).parts
            if 'workspace' in path_parts:
                workspace_idx = path_parts.index('workspace')
                module_parts = path_parts[workspace_idx + 1:]
                if module_parts[-1].endswith('.py'):
                    module_parts[-1] = module_parts[-1][:-3]
                return '.'.join(module_parts)
        except Exception:
            pass
        return None
    
    @staticmethod
    def is_documentation_file(file_path: str) -> bool:
        """Check if a file is a documentation or config file."""
        doc_extensions = {'.md', '.rst', '.txt', '.yaml', '.yml', '.json', '.toml', '.ini', '.cfg'}
        config_keywords = ['config', 'settings', 'example', 'demo', 'test']
        
        file_path_str = str(file_path)
        
        # Check file extension
        if any(file_path_str.endswith(ext) for ext in doc_extensions):
            return True
        
        # Check if it's in a config-related directory
        if any(keyword in file_path_str.lower() for keyword in config_keywords):
            return True
        
        return False
