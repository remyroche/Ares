from src.utils.tprint import tprint

"""
Simple Complexity Analyzer

A basic complexity analyzer that works without external dependencies.
Uses only Python's built-in ast module for analysis.
"""

import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
import logging


@dataclass
class FunctionComplexity:
    name: str
    line_number: int
    complexity: int
    signature: str


@dataclass
class ClassComplexity:
    name: str
    line_number: int
    complexity: int
    methods: List[FunctionComplexity]


@dataclass
class ModuleComplexity:
    path: str
    functions: List[FunctionComplexity]
    classes: List[ClassComplexity]
    overall_metrics: Dict[str, Any]
    complexity_score: float


class SimpleComplexityAnalyzer:
    """Simple complexity analyzer using only built-in Python modules."""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def analyze_file(self, file_path: str) -> ModuleComplexity:
        """Analyze a single Python file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix != ".py":
            raise ValueError(f"File must be a Python file: {file_path}")
        
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
        
        return self._analyze_source(source, str(file_path))
    
    def analyze_directory(self, directory: str) -> Dict[str, ModuleComplexity]:
        """Analyze all Python files in a directory."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        python_files = list(directory.rglob("*.py"))
        results = {}
        
        for file_path in python_files:
            try:
                results[str(file_path)] = self.analyze_file(file_path)
            except Exception as e:
                tprint(f"Warning: Could not analyze {file_path}: {e}")
        
        return results
    
    def _analyze_source(self, source: str, file_path: str) -> ModuleComplexity:
        """Analyze source code complexity."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}")
        
        # Analyze functions and classes
        functions = self._analyze_functions(tree, source)
        classes = self._analyze_classes(tree, source)
        
        # Calculate overall metrics
        total_complexity = sum(func.complexity for func in functions)
        total_complexity += sum(cls.complexity for cls in classes)
        
        overall_metrics = {
            "cyclomatic_complexity": total_complexity,
            "maintainability_index": max(0, 100 - total_complexity * 2),  # Simple calculation
            "halstead_volume": len(source.split()),  # Rough approximation
            "loc": len(source.splitlines()),
            "functions_count": len(functions),
            "classes_count": len(classes)
        }
        
        # Calculate complexity score (0-1, higher is more complex)
        complexity_score = min(1.0, total_complexity / 100.0)
        
        return ModuleComplexity(
            path=file_path,
            functions=functions,
            classes=classes,
            overall_metrics=overall_metrics,
            complexity_score=complexity_score
        )
    
    def _analyze_functions(self, tree: ast.AST, source: str) -> List[FunctionComplexity]:
        """Analyze function complexity."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                signature = self._get_function_signature(node, source)
                
                functions.append(FunctionComplexity(
                    name=node.name,
                    line_number=node.lineno,
                    complexity=complexity,
                    signature=signature
                ))
        
        return functions
    
    def _analyze_classes(self, tree: ast.AST, source: str) -> List[ClassComplexity]:
        """Analyze class complexity."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                class_complexity = 0
                
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        method_complexity = self._calculate_complexity(method)
                        class_complexity += method_complexity
                        
                        methods.append(FunctionComplexity(
                            name=method.name,
                            line_number=method.lineno,
                            complexity=method_complexity,
                            signature=self._get_function_signature(method, source)
                        ))
                
                classes.append(ClassComplexity(
                    name=node.name,
                    line_number=node.lineno,
                    complexity=class_complexity,
                    methods=methods
                ))
        
        return classes
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _get_function_signature(self, node: ast.FunctionDef, source: str) -> str:
        """Extract function signature."""
        lines = source.splitlines()
        if node.lineno <= len(lines):
            return lines[node.lineno - 1].strip()
        return f"def {node.name}(...)"