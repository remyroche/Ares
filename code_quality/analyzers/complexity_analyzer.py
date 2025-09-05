#!/usr/bin/env python3
"""Complexity analyzer for code complexity analysis."""

import ast
from typing import Dict, List, Any
from .base_analyzer import BaseAnalyzer


class ComplexityAnalyzer(BaseAnalyzer):
    """Analyzes code complexity."""
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze complexity in the directory."""
        python_files = self._find_python_files(directory_path)
        files = {}
        total_complexity = 0
        file_count = 0
        
        for file_path in python_files:
            content = self._read_file_safely(file_path)
            if not content:
                continue
            
            tree = self._parse_ast_safely(content, file_path)
            if not tree:
                continue
            
            complexity = self._calculate_complexity(tree)
            files[str(file_path)] = {
                "complexity": complexity,
                "line_count": len(content.split('\n'))
            }
            
            total_complexity += complexity
            file_count += 1
            self.stats["files_analyzed"] += 1
        
        return {
            "files": files,
            "average_complexity": total_complexity / file_count if file_count > 0 else 0,
            "total_files": file_count,
            "stats": self.stats
        }
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity
