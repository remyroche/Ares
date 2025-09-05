#!/usr/bin/env python3
"""Call graph analyzer for mapping function call relationships."""

import ast
from pathlib import Path
from typing import Dict, List, Any
from .base_analyzer import BaseAnalyzer


class CallGraphAnalyzer(BaseAnalyzer):
    """Analyzes function call relationships."""
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze call graph in the directory."""
        python_files = self._find_python_files(directory_path)
        functions = {}
        call_relationships = []
        
        for file_path in python_files:
            content = self._read_file_safely(file_path)
            if not content:
                continue
            
            tree = self._parse_ast_safely(content, file_path)
            if not tree:
                continue
            
            file_functions, file_calls = self._extract_calls(tree, file_path)
            functions.update(file_functions)
            call_relationships.extend(file_calls)
            
            self.stats["files_analyzed"] += 1
            self.stats["total_items"] += len(file_calls)
        
        return {
            "functions": functions,
            "call_relationships": call_relationships,
            "total_functions": len(functions),
            "total_calls": len(call_relationships),
            "stats": self.stats
        }
    
    def _extract_calls(self, tree: ast.AST, file_path: Path) -> tuple:
        """Extract function definitions and calls from AST."""
        functions = {}
        calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = {
                    "file_path": str(file_path),
                    "line_number": node.lineno,
                    "calls": []
                }
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append({
                        "caller": {"name": "unknown", "file_path": str(file_path), "line_number": node.lineno},
                        "callee": {"name": node.func.id, "file_path": str(file_path), "line_number": node.lineno}
                    })
        
        return functions, calls
