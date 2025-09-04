#!/usr/bin/env python3
"""
Dependency Manager - Handles optional dependencies gracefully.

This module provides fallback mechanisms for missing dependencies.
"""

import importlib
import sys
from typing import Any, Dict, List, Optional, Tuple


class DependencyManager:
    """Manages optional dependencies with fallback mechanisms."""
    
    def __init__(self):
        self.available_dependencies = {}
        self.fallback_implementations = {}
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check which dependencies are available."""
        optional_deps = [
            "rich", "astroid", "pylint", "flake8", "black", "isort", 
            "mypy", "autopep8", "yapf", "docformatter", "flynt"
        ]
        
        for dep in optional_deps:
            try:
                importlib.import_module(dep)
                self.available_dependencies[dep] = True
            except ImportError:
                self.available_dependencies[dep] = False
    
    def is_available(self, dependency: str) -> bool:
        """Check if a dependency is available."""
        return self.available_dependencies.get(dependency, False)
    
    def get_available_dependencies(self) -> List[str]:
        """Get list of available dependencies."""
        return [dep for dep, available in self.available_dependencies.items() if available]
    
    def get_missing_dependencies(self) -> List[str]:
        """Get list of missing dependencies."""
        return [dep for dep, available in self.available_dependencies.items() if not available]
    
    def safe_import(self, module_name: str, fallback: Any = None) -> Tuple[Any, bool]:
        """Safely import a module with fallback."""
        try:
            module = importlib.import_module(module_name)
            return module, True
        except ImportError:
            return fallback, False
    
    def get_rich_console(self):
        """Get Rich console with fallback to basic print."""
        if self.is_available("rich"):
            from rich.console import Console
            return Console()
        else:
            return None
    
    def get_rich_progress(self):
        """Get Rich progress with fallback to basic progress."""
        if self.is_available("rich"):
            from rich.progress import Progress
            return Progress
        else:
            return None
    
    def get_linter_tools(self) -> Dict[str, Any]:
        """Get available linter tools."""
        linters = {}
        
        if self.is_available("pylint"):
            try:
                import pylint
                linters["pylint"] = pylint
            except ImportError:
                pass
        
        if self.is_available("flake8"):
            try:
                import flake8
                linters["flake8"] = flake8
            except ImportError:
                pass
        
        if self.is_available("mypy"):
            try:
                import mypy
                linters["mypy"] = mypy
            except ImportError:
                pass
        
        return linters
    
    def get_formatter_tools(self) -> Dict[str, Any]:
        """Get available formatter tools."""
        formatters = {}
        
        if self.is_available("black"):
            try:
                import black
                formatters["black"] = black
            except ImportError:
                pass
        
        if self.is_available("isort"):
            try:
                import isort
                formatters["isort"] = isort
            except ImportError:
                pass
        
        if self.is_available("autopep8"):
            try:
                import autopep8
                formatters["autopep8"] = autopep8
            except ImportError:
                pass
        
        return formatters
    
    def get_astroid_parser(self):
        """Get ASTroid parser with fallback to standard ast."""
        if self.is_available("astroid"):
            try:
                import astroid
                return astroid
            except ImportError:
                pass
        
        # Fallback to standard ast
        import ast
        return ast
    
    def print_dependency_status(self):
        """Print the status of all dependencies."""
        print("Dependency Status:")
        print("=" * 50)
        
        available = self.get_available_dependencies()
        missing = self.get_missing_dependencies()
        
        print(f"Available ({len(available)}): {', '.join(available) if available else 'None'}")
        print(f"Missing ({len(missing)}): {', '.join(missing) if missing else 'None'}")
        
        if missing:
            print(f"\nTo install missing dependencies:")
            print(f"pip install {' '.join(missing)}")
    
    def create_fallback_config(self) -> Dict[str, Any]:
        """Create a configuration that works with available dependencies."""
        config = {
            "auto_fix": {
                "enabled": True,
                "tools": [],
                "aggressive": False,
                "max_line_length": 120
            },
            "analysis": {
                "linters": [],
                "exclude_patterns": ["__pycache__", "*.pyc", ".git", "venv", "env"]
            }
        }
        
        # Add available formatters
        formatters = self.get_formatter_tools()
        if formatters:
            config["auto_fix"]["tools"].extend(formatters.keys())
        
        # Add available linters
        linters = self.get_linter_tools()
        if linters:
            config["analysis"]["linters"].extend(linters.keys())
        
        return config


# Global instance
dependency_manager = DependencyManager()


def get_dependency_manager() -> DependencyManager:
    """Get the global dependency manager instance."""
    return dependency_manager


def safe_import(module_name: str, fallback: Any = None) -> Tuple[Any, bool]:
    """Convenience function for safe imports."""
    return dependency_manager.safe_import(module_name, fallback)


def is_dependency_available(dependency: str) -> bool:
    """Convenience function to check dependency availability."""
    return dependency_manager.is_available(dependency)