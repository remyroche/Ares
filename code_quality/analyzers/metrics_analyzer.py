#!/usr/bin/env python3
"""
Code Metrics Analyzer

Calculates various code quality metrics including:
- Cyclomatic Complexity
- Cognitive Complexity
- Halstead Metrics
- Maintainability Index
- Lines of Code metrics (LOC, SLOC, CLOC)
"""

import ast
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass
class FunctionMetrics:
    """Metrics for a single function."""
    name: str
    line_number: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    halstead_metrics: Dict[str, float]
    lines_of_code: int
    source_lines_of_code: int
    comment_lines: int
    maintainability_index: float
    parameter_count: int
    return_points: int
    nesting_depth: int


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    name: str
    line_number: int
    methods_count: int
    lines_of_code: int
    weighted_methods_per_class: int  # Sum of complexities
    depth_of_inheritance: int
    coupling_between_objects: int
    lack_of_cohesion: float


@dataclass
class FileMetrics:
    """Metrics for a single file."""
    file_path: str
    total_lines: int
    source_lines: int
    comment_lines: int
    blank_lines: int
    functions: List[FunctionMetrics]
    classes: List[ClassMetrics]
    average_complexity: float
    max_complexity: int
    maintainability_index: float


class MetricsAnalyzer:
    """Analyzes code to calculate various quality metrics."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.file_metrics: Dict[str, FileMetrics] = {}
        
    def analyze_file(self, file_path: Path) -> FileMetrics:
        """Analyze a single Python file for metrics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Calculate line metrics
            lines = content.split('\n')
            line_metrics = self._calculate_line_metrics(lines)
            
            # Analyze functions and classes
            analyzer = MetricsVisitor(content, lines)
            analyzer.visit(tree)
            
            # Calculate file-level metrics
            all_complexities = [f.cyclomatic_complexity for f in analyzer.functions]
            avg_complexity = sum(all_complexities) / len(all_complexities) if all_complexities else 0
            max_complexity = max(all_complexities) if all_complexities else 0
            
            # Calculate file maintainability index
            file_mi = self._calculate_file_maintainability_index(
                analyzer.functions,
                line_metrics['source_lines']
            )
            
            metrics = FileMetrics(
                file_path=str(file_path),
                total_lines=line_metrics['total_lines'],
                source_lines=line_metrics['source_lines'],
                comment_lines=line_metrics['comment_lines'],
                blank_lines=line_metrics['blank_lines'],
                functions=analyzer.functions,
                classes=analyzer.classes,
                average_complexity=avg_complexity,
                max_complexity=max_complexity,
                maintainability_index=file_mi
            )
            
            self.file_metrics[str(file_path)] = metrics
            return metrics
            
        except Exception as e:
            # Return empty metrics on error
            return FileMetrics(
                file_path=str(file_path),
                total_lines=0,
                source_lines=0,
                comment_lines=0,
                blank_lines=0,
                functions=[],
                classes=[],
                average_complexity=0,
                max_complexity=0,
                maintainability_index=0
            )
            
    def _calculate_line_metrics(self, lines: List[str]) -> Dict[str, int]:
        """Calculate line-based metrics."""
        total_lines = len(lines)
        blank_lines = 0
        comment_lines = 0
        source_lines = 0
        
        in_multiline_string = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check for multiline strings
            if '"""' in line or "'''" in line:
                in_multiline_string = not in_multiline_string
                comment_lines += 1
                continue
                
            if in_multiline_string:
                comment_lines += 1
            elif not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            else:
                source_lines += 1
                
        return {
            'total_lines': total_lines,
            'source_lines': source_lines,
            'comment_lines': comment_lines,
            'blank_lines': blank_lines
        }
        
    def _calculate_file_maintainability_index(self, 
                                             functions: List[FunctionMetrics],
                                             source_lines: int) -> float:
        """Calculate maintainability index for entire file."""
        if not functions or source_lines == 0:
            return 0
            
        # Average of function maintainability indices
        avg_mi = sum(f.maintainability_index for f in functions) / len(functions)
        
        # Adjust based on file size
        size_penalty = max(0, (source_lines - 1000) / 1000) * 10
        
        return max(0, avg_mi - size_penalty)
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report."""
        total_files = len(self.file_metrics)
        total_functions = sum(len(m.functions) for m in self.file_metrics.values())
        total_classes = sum(len(m.classes) for m in self.file_metrics.values())
        
        # Aggregate metrics
        all_complexities = []
        all_mi_scores = []
        total_loc = 0
        total_sloc = 0
        
        for metrics in self.file_metrics.values():
            all_complexities.extend([f.cyclomatic_complexity for f in metrics.functions])
            all_mi_scores.extend([f.maintainability_index for f in metrics.functions])
            total_loc += metrics.total_lines
            total_sloc += metrics.source_lines
            
        return {
            'summary': {
                'total_files': total_files,
                'total_functions': total_functions,
                'total_classes': total_classes,
                'total_lines_of_code': total_loc,
                'total_source_lines': total_sloc,
                'average_cyclomatic_complexity': sum(all_complexities) / len(all_complexities) if all_complexities else 0,
                'max_cyclomatic_complexity': max(all_complexities) if all_complexities else 0,
                'average_maintainability_index': sum(all_mi_scores) / len(all_mi_scores) if all_mi_scores else 0
            },
            'files': {
                path: self._file_metrics_to_dict(metrics)
                for path, metrics in self.file_metrics.items()
            },
            'high_complexity_functions': self._get_high_complexity_functions(),
            'low_maintainability_functions': self._get_low_maintainability_functions()
        }
        
    def _file_metrics_to_dict(self, metrics: FileMetrics) -> Dict[str, Any]:
        """Convert FileMetrics to dictionary."""
        return {
            'total_lines': metrics.total_lines,
            'source_lines': metrics.source_lines,
            'comment_lines': metrics.comment_lines,
            'blank_lines': metrics.blank_lines,
            'average_complexity': metrics.average_complexity,
            'max_complexity': metrics.max_complexity,
            'maintainability_index': metrics.maintainability_index,
            'functions': [
                {
                    'name': f.name,
                    'line_number': f.line_number,
                    'cyclomatic_complexity': f.cyclomatic_complexity,
                    'cognitive_complexity': f.cognitive_complexity,
                    'maintainability_index': f.maintainability_index,
                    'lines_of_code': f.lines_of_code,
                    'parameter_count': f.parameter_count
                }
                for f in metrics.functions
            ]
        }
        
    def _get_high_complexity_functions(self, threshold: int = 10) -> List[Dict[str, Any]]:
        """Get functions with high cyclomatic complexity."""
        high_complexity = []
        
        for file_path, metrics in self.file_metrics.items():
            for func in metrics.functions:
                if func.cyclomatic_complexity > threshold:
                    high_complexity.append({
                        'file': file_path,
                        'function': func.name,
                        'line': func.line_number,
                        'complexity': func.cyclomatic_complexity
                    })
                    
        return sorted(high_complexity, key=lambda x: x['complexity'], reverse=True)
        
    def _get_low_maintainability_functions(self, threshold: float = 20) -> List[Dict[str, Any]]:
        """Get functions with low maintainability index."""
        low_maintainability = []
        
        for file_path, metrics in self.file_metrics.items():
            for func in metrics.functions:
                if func.maintainability_index < threshold:
                    low_maintainability.append({
                        'file': file_path,
                        'function': func.name,
                        'line': func.line_number,
                        'maintainability_index': func.maintainability_index
                    })
                    
        return sorted(low_maintainability, key=lambda x: x['maintainability_index'])


class MetricsVisitor(ast.NodeVisitor):
    """AST visitor for calculating code metrics."""
    
    def __init__(self, source_code: str, lines: List[str]):
        self.source_code = source_code
        self.lines = lines
        self.functions: List[FunctionMetrics] = []
        self.classes: List[ClassMetrics] = []
        self.current_class = None
        self.current_function = None
        self.nesting_level = 0
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        self._visit_function(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self._visit_function(node)
        
    def _visit_function(self, node: Any) -> None:
        """Process function definition."""
        # Calculate metrics
        cyclomatic = self._calculate_cyclomatic_complexity(node)
        cognitive = self._calculate_cognitive_complexity(node)
        halstead = self._calculate_halstead_metrics(node)
        loc = self._calculate_function_lines(node)
        
        # Calculate maintainability index
        volume = halstead['volume']
        cyclo = cyclomatic
        sloc = loc['source_lines']
        
        # Microsoft's Maintainability Index formula
        mi = 171 - 5.2 * math.log(volume + 1) - 0.23 * cyclo - 16.2 * math.log(sloc + 1)
        mi = max(0, mi * 100 / 171)  # Normalize to 0-100
        
        metrics = FunctionMetrics(
            name=node.name,
            line_number=node.lineno,
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cognitive,
            halstead_metrics=halstead,
            lines_of_code=loc['total_lines'],
            source_lines_of_code=loc['source_lines'],
            comment_lines=loc['comment_lines'],
            maintainability_index=mi,
            parameter_count=len(node.args.args),
            return_points=self._count_return_points(node),
            nesting_depth=self._calculate_max_nesting(node)
        )
        
        self.functions.append(metrics)
        
        # Continue visiting
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        
        # Calculate class metrics
        metrics = ClassMetrics(
            name=node.name,
            line_number=node.lineno,
            methods_count=len(methods),
            lines_of_code=node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0,
            weighted_methods_per_class=sum(self._calculate_cyclomatic_complexity(m) for m in methods),
            depth_of_inheritance=len(node.bases),
            coupling_between_objects=self._calculate_coupling(node),
            lack_of_cohesion=self._calculate_cohesion(node)
        )
        
        self.classes.append(metrics)
        
        # Continue visiting
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += sum(1 for _ in child.ifs)
                
        return complexity
        
    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """Calculate cognitive complexity (simplified version)."""
        cognitive = 0
        nesting = 0
        
        def visit_with_nesting(n: ast.AST, level: int) -> None:
            nonlocal cognitive
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                cognitive += 1 + level  # Increase for nesting
                for child in ast.iter_child_nodes(n):
                    visit_with_nesting(child, level + 1)
            elif isinstance(n, ast.BoolOp):
                cognitive += len(n.values) - 1
            else:
                for child in ast.iter_child_nodes(n):
                    visit_with_nesting(child, level)
                    
        visit_with_nesting(node, 0)
        return cognitive
        
    def _calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """Calculate Halstead metrics."""
        operators = set()
        operands = set()
        total_operators = 0
        total_operands = 0
        
        for child in ast.walk(node):
            # Count operators
            if isinstance(child, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
                                ast.Pow, ast.LShift, ast.RShift, ast.BitOr,
                                ast.BitXor, ast.BitAnd, ast.FloorDiv)):
                operators.add(type(child).__name__)
                total_operators += 1
            elif isinstance(child, (ast.And, ast.Or, ast.Not)):
                operators.add(type(child).__name__)
                total_operators += 1
            elif isinstance(child, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                                  ast.Gt, ast.GtE, ast.Is, ast.IsNot,
                                  ast.In, ast.NotIn)):
                operators.add(type(child).__name__)
                total_operators += 1
                
            # Count operands
            elif isinstance(child, ast.Name):
                operands.add(child.id)
                total_operands += 1
            elif isinstance(child, (ast.Num, ast.Str, ast.Constant)):
                operands.add(str(child.n if isinstance(child, ast.Num) else child.value))
                total_operands += 1
                
        n1 = len(operators)  # Unique operators
        n2 = len(operands)   # Unique operands
        N1 = total_operators # Total operators
        N2 = total_operands  # Total operands
        
        # Halstead metrics
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary + 1) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = volume * difficulty
        
        return {
            'vocabulary': vocabulary,
            'length': length,
            'volume': volume,
            'difficulty': difficulty,
            'effort': effort,
            'unique_operators': n1,
            'unique_operands': n2,
            'total_operators': N1,
            'total_operands': N2
        }
        
    def _calculate_function_lines(self, node: ast.AST) -> Dict[str, int]:
        """Calculate line metrics for a function."""
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        total_lines = end_line - start_line + 1
        
        # Count source and comment lines
        source_lines = 0
        comment_lines = 0
        
        for i in range(start_line - 1, min(end_line, len(self.lines))):
            line = self.lines[i].strip()
            if line and not line.startswith('#'):
                source_lines += 1
            elif line.startswith('#'):
                comment_lines += 1
                
        return {
            'total_lines': total_lines,
            'source_lines': source_lines,
            'comment_lines': comment_lines
        }
        
    def _count_return_points(self, node: ast.AST) -> int:
        """Count number of return statements."""
        return sum(1 for child in ast.walk(node) if isinstance(child, ast.Return))
        
    def _calculate_max_nesting(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        
        def visit_with_depth(n: ast.AST, depth: int) -> None:
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                for child in ast.iter_child_nodes(n):
                    visit_with_depth(child, depth + 1)
            else:
                for child in ast.iter_child_nodes(n):
                    visit_with_depth(child, depth)
                    
        visit_with_depth(node, 0)
        return max_depth
        
    def _calculate_coupling(self, node: ast.ClassDef) -> int:
        """Calculate coupling between objects (simplified)."""
        # Count unique external references
        external_refs = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name) and child.value.id != 'self':
                    external_refs.add(child.value.id)
                    
        return len(external_refs)
        
    def _calculate_cohesion(self, node: ast.ClassDef) -> float:
        """Calculate lack of cohesion in methods (simplified)."""
        # This is a simplified LCOM calculation
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(methods) <= 1:
            return 0.0
            
        # Count instance variable usage per method
        method_vars = defaultdict(set)
        
        for method in methods:
            for child in ast.walk(method):
                if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id == 'self':
                    method_vars[method.name].add(child.attr)
                    
        # Calculate cohesion
        total_pairs = len(methods) * (len(methods) - 1) / 2
        disjoint_pairs = 0
        
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                if not (method_vars[m1.name] & method_vars[m2.name]):
                    disjoint_pairs += 1
                    
        return disjoint_pairs / total_pairs if total_pairs > 0 else 0.0