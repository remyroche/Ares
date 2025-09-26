#!/usr/bin/env python3
"""
Base analyzer class for code analysis operations.

This module provides the base class that all specific analyzers inherit from,
ensuring consistent interface and common functionality.
"""

import ast
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.tprint import tprint_error, tprint_warning


class BaseAnalyzer(ABC):
    """Base class for all code analyzers."""
    
    def __init__(self, config):
        """Initialize the analyzer with configuration."""
        self.config = config
        self.results = {}
        self.stats = {
            "files_analyzed": 0,
            "files_failed": 0,
            "total_items": 0
        }
    
    @abstractmethod
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze a directory and return results."""
        message = (
            f"{self.__class__.__name__} does not implement 'analyze_directory'. "
            "All analyzers must provide a concrete implementation so audits "
            "can fail fast instead of silently skipping work."
        )
        tprint_error(message)
        raise NotImplementedError(message)
    
    def _find_python_files(self, directory_path: str) -> List[Path]:
        """Find all Python files in the directory, excluding problematic ones."""
        project_root = Path(directory_path)
        python_files = []
        
        for py_file in project_root.rglob("*.py"):
            # Skip files in excluded directories
            if any(excluded in py_file.parts for excluded in ["venv", "__pycache__", ".git", "node_modules", ".pytest_cache"]):
                continue
            python_files.append(py_file)
        
        return python_files
    
    def _read_file_safely(self, file_path: Path) -> Optional[str]:
        """Read a file with encoding detection and error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as latin_error:
                tprint_error(
                    f"Failed to read {file_path} with latin-1 encoding: {latin_error}"
                )
                self.stats["files_failed"] += 1
                return None
        except Exception as utf_error:
            tprint_error(f"Failed to read {file_path}: {utf_error}")
            self.stats["files_failed"] += 1
            return None
    
    def _parse_ast_safely(self, content: str, file_path: Path) -> Optional[ast.AST]:
        """Parse AST with error handling."""
        try:
            return ast.parse(content, filename=str(file_path))
        except SyntaxError:
            self.stats["files_failed"] += 1
            return None
        except Exception as parse_error:
            tprint_error(f"Failed to parse AST for {file_path}: {parse_error}")
            self.stats["files_failed"] += 1
            return None
    
    def _extract_function_name_from_issue(self, issue) -> Optional[str]:
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
    
    def _get_module_from_file_path(self, file_path: str) -> Optional[str]:
        """Extract module name from file path."""
        try:
            path_parts = Path(file_path).parts
            if 'workspace' in path_parts:
                workspace_idx = path_parts.index('workspace')
                module_parts = path_parts[workspace_idx + 1:]
                if module_parts[-1].endswith('.py'):
                    module_parts[-1] = module_parts[-1][:-3]
                return '.'.join(module_parts)
        except Exception as path_error:
            tprint_warning(
                f"Could not derive module name from {file_path}: {path_error}"
            )
        return None
