#!/usr/bin/env python3
"""Enhanced dependency analyzer for mapping module dependencies."""

import ast
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
from .base_analyzer import BaseAnalyzer
import numpy as np


class DependencyAnalyzer(BaseAnalyzer):
    """Enhanced analyzer for module dependencies and import relationships."""
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze dependencies in the directory with enhanced features."""
        python_files = self._find_python_files(directory_path)
        modules = {}
        dependency_graph = defaultdict(set)
        circular_dependencies = []
        external_dependencies = set()
        internal_dependencies = set()
        
        # First pass: collect all modules and their dependencies
        for file_path in python_files:
            content = self._read_file_safely(file_path)
            if not content:
                continue
            
            tree = self._parse_ast_safely(content, file_path)
            if not tree:
                continue
            
            module_name = self._get_module_name(file_path, directory_path)
            dependencies = self._extract_dependencies(tree)
            
            # Categorize dependencies
            internal_deps = []
            external_deps = []
            
            for dep in dependencies:
                if self._is_internal_dependency(dep, python_files, directory_path):
                    internal_deps.append(dep)
                    internal_dependencies.add(dep)
                    dependency_graph[module_name].add(dep)
                else:
                    external_deps.append(dep)
                    external_dependencies.add(dep)
            
            modules[module_name] = {
                "file_path": str(file_path),
                "dependencies": dependencies,
                "internal_dependencies": internal_deps,
                "external_dependencies": external_deps,
                "import_count": len(dependencies),
                "internal_import_count": len(internal_deps),
                "external_import_count": len(external_deps),
                "dependents": [],  # Will be filled in second pass
                "dependency_level": 0  # Will be calculated
            }
            
            self.stats["files_analyzed"] += 1
            self.stats["total_items"] += len(dependencies)
        
        # Second pass: find dependents and calculate dependency levels
        for module_name, module_data in modules.items():
            dependents = []
            for other_module, other_data in modules.items():
                if module_name in other_data["dependencies"]:
                    dependents.append(other_module)
            module_data["dependents"] = dependents
        
        # Calculate dependency levels (how deep in the dependency chain)
        dependency_levels = self._calculate_dependency_levels(modules)
        for module_name, level in dependency_levels.items():
            if module_name in modules:
                modules[module_name]["dependency_level"] = level
        
        # Detect circular dependencies
        circular_dependencies = self._detect_circular_dependencies(dependency_graph)
        
        # Calculate dependency metrics
        total_dependencies = sum(len(m["dependencies"]) for m in modules.values())
        total_internal = sum(len(m["internal_dependencies"]) for m in modules.values())
        total_external = sum(len(m["external_dependencies"]) for m in modules.values())
        
        # Convert sets to lists for JSON serialization
        serializable_dependency_graph = {}
        for module, deps in dependency_graph.items():
            serializable_dependency_graph[module] = list(deps)
        
        return {
            "modules": modules,
            "total_modules": len(modules),
            "total_dependencies": total_dependencies,
            "internal_dependencies": total_internal,
            "external_dependencies": total_external,
            "circular_dependencies": circular_dependencies,
            "dependency_graph": serializable_dependency_graph,
            "external_modules": list(external_dependencies),
            "internal_modules": list(internal_dependencies),
            "dependency_metrics": {
                "max_dependency_level": max(dependency_levels.values()) if dependency_levels else 0,
                "avg_dependencies_per_module": total_dependencies / len(modules) if modules else 0,
                "modules_with_circular_deps": len(set(module for cycle in circular_dependencies for module in cycle)),
                "most_dependent_module": max(modules.keys(), key=lambda m: len(modules[m]["dependents"])) if modules else None,
                "least_dependent_module": min(modules.keys(), key=lambda m: len(modules[m]["dependents"])) if modules else None
            },
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
        """Extract import dependencies from AST with enhanced parsing."""
        dependencies = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module)
                # Also track specific imports
                for alias in node.names:
                    if alias.name != '*':  # Skip wildcard imports
                        full_name = f"{node.module}.{alias.name}" if node.module else alias.name
                        dependencies.append(full_name)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _is_internal_dependency(self, dependency: str, python_files: List[Path], root_path: str) -> bool:
        """Check if a dependency is internal to the project."""
        # Convert dependency to potential file path
        dep_parts = dependency.split('.')
        
        # Check if it matches any internal module
        for file_path in python_files:
            module_name = self._get_module_name(file_path, root_path)
            if module_name == dependency or module_name.startswith(dependency + '.'):
                return True
        
        # Check for relative imports (internal)
        if dependency.startswith('.'):
            return True
        
        return False
    
    def _calculate_dependency_levels(self, modules: Dict[str, Any]) -> Dict[str, int]:
        """Calculate dependency levels for each module."""
        levels = {}
        visited = set()
        
        def calculate_level(module_name: str, current_path: List[str]) -> int:
            if module_name in current_path:
                # Circular dependency - return current level
                return len(current_path)
            
            if module_name in levels:
                return levels[module_name]
            
            if module_name not in modules:
                return 0
            
            current_path.append(module_name)
            max_dep_level = 0
            
            for dep in modules[module_name]["dependencies"]:
                if dep in modules:  # Only internal dependencies
                    dep_level = calculate_level(dep, current_path.copy())
                    max_dep_level = max(max_dep_level, dep_level)
            
            current_path.pop()
            level = max_dep_level + 1
            levels[module_name] = level
            return level
        
        for module_name in modules:
            if module_name not in visited:
                calculate_level(module_name, [])
                visited.add(module_name)
        
        return levels
    
    def _detect_circular_dependencies(self, dependency_graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Detect circular dependencies using DFS."""
        circular_deps = []
        visited = set()
        rec_stack = set()
        
        def dfs(module: str, path: List[str]) -> None:
            if module in rec_stack:
                # Found a cycle
                cycle_start = path.index(module)
                cycle = path[cycle_start:] + [module]
                circular_deps.append(cycle)
                return
            
            if module in visited:
                return
            
            visited.add(module)
            rec_stack.add(module)
            path.append(module)
            
            for neighbor in dependency_graph.get(module, []):
                dfs(neighbor, path.copy())
            
            rec_stack.remove(module)
            path.pop()
        
        for module in dependency_graph:
            if module not in visited:
                dfs(module, [])
        
        return circular_deps
