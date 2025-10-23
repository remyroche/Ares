#!/usr/bin/env python3
"""Call graph analyzer for mapping function call relationships."""

import ast
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, deque
from .base_analyzer import BaseAnalyzer
import numpy as np


class CallGraphAnalyzer(BaseAnalyzer):
    """Analyzes function call relationships with proper call depth calculation."""
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze call graph in the directory."""
        python_files = self._find_python_files(directory_path)
        functions = {}
        call_relationships = []
        function_calls = defaultdict(list)  # function -> list of called functions
        
        # First pass: collect all function definitions and calls
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
            
            # Build call graph
            for call in file_calls:
                caller = call["caller"]["name"]
                callee = call["callee"]["name"]
                if caller != "unknown" and caller in functions:
                    function_calls[caller].append(callee)
            
            self.stats["files_analyzed"] += 1
            self.stats["total_items"] += len(file_calls)
        
        # Calculate call depths and circular calls
        max_call_depth, circular_calls = self._calculate_call_depths(functions, function_calls)
        
        # Update function data with call information
        for func_name, func_data in functions.items():
            func_data["calls"] = function_calls.get(func_name, [])
            func_data["call_count"] = len(function_calls.get(func_name, []))
        
        return {
            "functions": functions,
            "call_relationships": call_relationships,
            "total_functions": len(functions),
            "total_calls": len(call_relationships),
            "max_call_depth": max_call_depth,
            "circular_calls": circular_calls,
            "call_graph": dict(function_calls),
            "stats": self.stats
        }
    
    def _extract_calls(self, tree: ast.AST, file_path: Path) -> tuple:
        """Extract function definitions and calls from AST with proper context."""
        functions = {}
        calls = []
        current_function = None
        
        class CallExtractor(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None
                self.functions = {}
                self.calls = []
            
            def visit_FunctionDef(self, node):
                # Store current function context
                old_function = self.current_function
                self.current_function = node.name
                
                # Record function definition
                self.functions[node.name] = {
                    "file_path": str(file_path),
                    "line_number": node.lineno,
                    "calls": []
                }
                
                # Visit function body
                self.generic_visit(node)
                
                # Restore previous function context
                self.current_function = old_function
            
            def visit_ClassDef(self, node):
                # For class methods, we'll track them separately
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Extract function call information
                callee_name = self._get_call_name(node)
                if callee_name:
                    call_info = {
                        "caller": {
                            "name": self.current_function or "module_level",
                            "file_path": str(file_path),
                            "line_number": node.lineno
                        },
                        "callee": {
                            "name": callee_name,
                            "file_path": str(file_path),
                            "line_number": node.lineno
                        }
                    }
                    self.calls.append(call_info)
                
                self.generic_visit(node)
            
            def _get_call_name(self, node):
                """Extract the name of the called function."""
                if isinstance(node.func, ast.Name):
                    return node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # Handle method calls like obj.method()
                    return f"{self._get_attr_name(node.func)}"
                elif isinstance(node.func, ast.Call):
                    # Handle chained calls
                    return "chained_call"
                return None
            
            def _get_attr_name(self, node):
                """Get attribute name for method calls."""
                if isinstance(node.value, ast.Name):
                    return f"{node.value.id}.{node.attr}"
                elif isinstance(node.value, ast.Attribute):
                    return f"{self._get_attr_name(node.value)}.{node.attr}"
                else:
                    return f"*.{node.attr}"
        
        extractor = CallExtractor()
        extractor.visit(tree)
        
        return extractor.functions, extractor.calls
    
    def _calculate_call_depths(self, functions: Dict, function_calls: Dict[str, List[str]]) -> Tuple[int, List[str]]:
        """Calculate maximum call depth and detect circular calls."""
        max_depth = 0
        circular_calls = []
        
        def calculate_depth(func_name: str, visited: Set[str], current_path: List[str]) -> int:
            """Calculate call depth for a function using DFS."""
            if func_name in current_path:
                # Circular call detected
                cycle_start = current_path.index(func_name)
                cycle = current_path[cycle_start:] + [func_name]
                circular_calls.append(" -> ".join(cycle))
                return 0
            
            if func_name in visited:
                return 0
            
            visited.add(func_name)
            current_path.append(func_name)
            
            max_child_depth = 0
            for callee in function_calls.get(func_name, []):
                if callee in functions:  # Only count calls to defined functions
                    child_depth = calculate_depth(callee, visited, current_path.copy())
                    max_child_depth = max(max_child_depth, child_depth)
            
            current_path.pop()
            return 1 + max_child_depth
        
        # Calculate depth for each function
        for func_name in functions:
            if func_name in function_calls:
                depth = calculate_depth(func_name, set(), [])
                max_depth = max(max_depth, depth)
        
        return max_depth, circular_calls
