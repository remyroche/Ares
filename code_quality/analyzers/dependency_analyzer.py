#!/usr/bin/env python3
"""Dependency analyzer for mapping module dependencies."""

import ast
from pathlib import Path
from typing import Dict, List, Any
from .base_analyzer import BaseAnalyzer


class DependencyAnalyzer(BaseAnalyzer):
    """Analyzes module dependencies and import relationships."""
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze dependencies in the directory."""
        python_files = self._find_python_files(directory_path)
        modules = {}
        
        for file_path in python_files:
            content = self._read_file_safely(file_path)
            if not content:
                continue
            
            tree = self._parse_ast_safely(content, file_path)
            if not tree:
                continue
            
            module_name = self._get_module_name(file_path, directory_path)
            dependencies = self._extract_dependencies(tree)
            
            modules[module_name] = {
                "file_path": str(file_path),
                "dependencies": dependencies,
                "import_count": len(dependencies)
            }
            
            self.stats["files_analyzed"] += 1
            self.stats["total_items"] += len(dependencies)
        
        return {
            "modules": modules,
            "total_modules": len(modules),
            "total_dependencies": sum(len(m["dependencies"]) for m in modules.values()),
            "stats": self.stats
        }
    
    def _get_module_name(self, file_path: Path, root_path: str) -> str:
        """Convert file path to module name."""
        try:
            relative_path = file_path.relative_to(Path(root_path))
            module_parts = list(relative_path.parts)
            if module_parts[-1].endswith('.py'):
                module_parts[-1] = module_parts[-1][:-3]
            return '.'.join(module_parts)
        except Exception:
            return str(file_path.name)
    
    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extract import dependencies from AST."""
        dependencies = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module)
        return dependencies
