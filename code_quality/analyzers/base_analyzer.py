#!/usr/bin/env python3
"""
Base analyzer class for code analysis operations.

This module provides the base class that all specific analyzers inherit from,
ensuring consistent interface and common functionality.
"""

import ast
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


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
        pass
    
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
            except Exception:
                return None
        except Exception:
            return None
    
    def _parse_ast_safely(self, content: str, file_path: Path) -> Optional[ast.AST]:
        """Parse AST with error handling."""
        try:
            return ast.parse(content, filename=str(file_path))
        except SyntaxError:
            self.stats["files_failed"] += 1
            return None
        except Exception:
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
        except Exception:
            pass
        return None