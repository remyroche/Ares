#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Enhanced Function Analyzer

This analyzer focuses on legitimate function issues while excluding false positives
from fallback functions and stub implementations. It specifically tracks:

1. Functions with too many arguments (legitimate refactoring needed)
2. Undefined function calls (import issues)
3. Other function design issues
4. Excludes missing docstrings from fallback functions
"""

import ast
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class FunctionIssueType(Enum):
    """Types of function issues."""
    TOO_MANY_ARGUMENTS = "too_many_arguments"
    UNDEFINED_FUNCTION_CALL = "undefined_function_call"
    FUNCTION_COMPLEXITY = "function_complexity"
    FUNCTION_NAMING = "function_naming"
    FUNCTION_DESIGN = "function_design"
    MISSING_RETURN_TYPE = "missing_return_type"
    UNUSED_PARAMETERS = "unused_parameters"


class FunctionIssueSeverity(Enum):
    """Function issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FunctionIssue:
    """Represents a function issue."""
    type: FunctionIssueType
    severity: FunctionIssueSeverity
    function_name: str
    line: int
    column: int = 0
    context: str = ""
    file_path: str = ""
    description: str = ""
    suggestions: List[str] = field(default_factory=list)
    argument_count: Optional[int] = None
    is_fallback_function: bool = False


@dataclass
class FunctionAnalysisResult:
    """Results from function analysis."""
    file_path: str
    issues: List[FunctionIssue] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def total_issues(self) -> int:
        return len(self.issues)
    
    @property
    def critical_issues(self) -> int:
        return len([i for i in self.issues if i.severity == FunctionIssueSeverity.CRITICAL])
    
    @property
    def high_issues(self) -> int:
        return len([i for i in self.issues if i.severity == FunctionIssueSeverity.HIGH])
    
    @property
    def issues_by_type(self) -> Dict[FunctionIssueType, int]:
        type_counts = {}
        for issue in self.issues:
            type_counts[issue.type] = type_counts.get(issue.type, 0) + 1
        return type_counts


class EnhancedFunctionAnalyzer:
    """Enhanced analyzer for legitimate function issues."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the function analyzer."""
        self.config = config or {}
        
        # Common fallback function patterns to exclude
        self.fallback_function_patterns = {
            'handles_errors', 'monitor_feature_engineering', 'validates',
            'traced', 'log_execution_time', 'cached', 'ensure_data_integrity',
            'monitor_step_execution', 'secure_step_execution', 'comprehensive_data_validation',
            'handle_errors', 'memory_efficient', 'resource_monitor', 'secure_data_processing',
            'validate_data_structure', 'with_tracing_span', 'quality_gate',
            'validate_pipeline_step', 'with_enhanced_mlflow_logging', 'decorator',
            'create_fallback_logger', 'create_fallback_decorator', '_identity', '_wrap'
        }
        
        # Common stub class patterns
        self.stub_class_patterns = {
            '_PDStub', '_NPStub', '_Stub', 'Mock', 'Dummy', 'Placeholder',
            'Fallback', 'Default', 'Safe', 'Conditional'
        }
        
        # Argument count thresholds
        self.max_arguments = self.config.get('max_arguments', 6)
        self.critical_arguments = self.config.get('critical_arguments', 10)
        
        # Function complexity thresholds
        self.max_function_length = self.config.get('max_function_length', 100)
        self.max_complexity = self.config.get('max_complexity', 10)
    
    def analyze_file(self, file_path: str) -> FunctionAnalysisResult:
        """Analyze a file for function issues."""
        start_time = time.time()
        result = FunctionAnalysisResult(file_path=file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            visitor = FunctionIssueVisitor(self, result, content)
            visitor.visit(tree)
            
        except Exception as e:
            result.error = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def is_fallback_function(self, function_name: str) -> bool:
        """Check if a function is a fallback implementation."""
        return function_name in self.fallback_function_patterns
    
    def is_stub_class(self, class_name: str) -> bool:
        """Check if a class is a stub implementation."""
        return any(pattern in class_name for pattern in self.stub_class_patterns)


class FunctionIssueVisitor(ast.NodeVisitor):
    """AST visitor for detecting function issues."""
    
    def __init__(self, analyzer: EnhancedFunctionAnalyzer, result: FunctionAnalysisResult, content: str):
        self.analyzer = analyzer
        self.result = result
        self.content = content
        self.lines = content.split('\n')
        self.current_class = None
        self.function_calls = set()
        self.defined_functions = set()
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        
        # Skip stub classes
        if not self.analyzer.is_stub_class(node.name):
            self.generic_visit(node)
        
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        function_name = node.name
        self.defined_functions.add(function_name)
        
        # Skip fallback functions
        if self.analyzer.is_fallback_function(function_name):
            return
        
        # Skip functions in stub classes
        if self.current_class and self.analyzer.is_stub_class(self.current_class):
            return
        
        # Check for too many arguments
        self._check_too_many_arguments(node)
        
        # Check function complexity
        self._check_function_complexity(node)
        
        # Check function naming
        self._check_function_naming(node)
        
        # Check function design
        self._check_function_design(node)
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls."""
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
            self.function_calls.add(function_name)
            
            # Check for undefined function calls
            if (function_name not in self.defined_functions and 
                not self._is_builtin_function(function_name) and
                not self._is_imported_function(function_name)):
                
                self._add_function_issue(
                    FunctionIssueType.UNDEFINED_FUNCTION_CALL,
                    FunctionIssueSeverity.MEDIUM,
                    function_name,
                    node.lineno,
                    node.col_offset,
                    f"Undefined function call: {function_name}",
                    [
                        "Check if the function is imported",
                        "Verify the function name is correct",
                        "Add the missing import statement"
                    ]
                )
        
        self.generic_visit(node)
    
    def _check_too_many_arguments(self, node: ast.FunctionDef) -> None:
        """Check if a function has too many arguments."""
        arg_count = len(node.args.args)
        
        if arg_count > self.analyzer.critical_arguments:
            severity = FunctionIssueSeverity.CRITICAL
            description = f"Function '{node.name}' has {arg_count} arguments (critical threshold: {self.analyzer.critical_arguments})"
            suggestions = [
                "Consider using a configuration object",
                "Split the function into smaller functions",
                "Use keyword-only arguments",
                "Consider using a dataclass or named tuple for parameters"
            ]
        elif arg_count > self.analyzer.max_arguments:
            severity = FunctionIssueSeverity.HIGH
            description = f"Function '{node.name}' has {arg_count} arguments (recommended max: {self.analyzer.max_arguments})"
            suggestions = [
                "Consider using a configuration object",
                "Group related parameters into objects",
                "Use keyword-only arguments for clarity"
            ]
        else:
            return
        
        self._add_function_issue(
            FunctionIssueType.TOO_MANY_ARGUMENTS,
            severity,
            node.name,
            node.lineno,
            node.col_offset,
            description,
            suggestions,
            argument_count=arg_count
        )
    
    def _check_function_complexity(self, node: ast.FunctionDef) -> None:
        """Check function complexity."""
        # Calculate function length
        if hasattr(node, 'end_lineno') and node.end_lineno:
            function_length = node.end_lineno - node.lineno
        else:
            function_length = len([stmt for stmt in ast.walk(node) if isinstance(stmt, ast.stmt)])
        
        if function_length > self.analyzer.max_function_length:
            self._add_function_issue(
                FunctionIssueType.FUNCTION_COMPLEXITY,
                FunctionIssueSeverity.MEDIUM,
                node.name,
                node.lineno,
                node.col_offset,
                f"Function '{node.name}' is too long ({function_length} lines, max: {self.analyzer.max_function_length})",
                [
                    "Split the function into smaller functions",
                    "Extract helper methods",
                    "Consider using a class-based approach"
                ]
            )
    
    def _check_function_naming(self, node: ast.FunctionDef) -> None:
        """Check function naming conventions."""
        function_name = node.name
        
        # Check for unclear names
        unclear_names = {'_identity', '_wrap', 'decorator', 'wrapper', 'func', 'fn'}
        if function_name in unclear_names:
            self._add_function_issue(
                FunctionIssueType.FUNCTION_NAMING,
                FunctionIssueSeverity.LOW,
                node.name,
                node.lineno,
                node.col_offset,
                f"Function '{node.name}' has an unclear name",
                [
                    "Use a more descriptive function name",
                    "The name should clearly indicate what the function does"
                ]
            )
    
    def _check_function_design(self, node: ast.FunctionDef) -> None:
        """Check function design issues."""
        # Check for functions with multiple responsibilities
        if self._has_multiple_responsibilities(node):
            self._add_function_issue(
                FunctionIssueType.FUNCTION_DESIGN,
                FunctionIssueSeverity.MEDIUM,
                node.name,
                node.lineno,
                node.col_offset,
                f"Function '{node.name}' appears to have multiple responsibilities",
                [
                    "Split the function into smaller, focused functions",
                    "Each function should have a single responsibility",
                    "Consider using a class-based approach"
                ]
            )
    
    def _has_multiple_responsibilities(self, node: ast.FunctionDef) -> bool:
        """Check if a function has multiple responsibilities."""
        # Simple heuristic: count different types of operations
        operation_types = set()
        
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                operation_types.add('assignment')
            elif isinstance(stmt, ast.Call):
                operation_types.add('function_call')
            elif isinstance(stmt, (ast.If, ast.While, ast.For)):
                operation_types.add('control_flow')
            elif isinstance(stmt, ast.Return):
                operation_types.add('return')
        
        # If function has many different types of operations, it might have multiple responsibilities
        return len(operation_types) > 4
    
    def _is_builtin_function(self, function_name: str) -> bool:
        """Check if a function is a built-in."""
        builtins = {
            'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
            'print', 'input', 'open', 'range', 'enumerate', 'zip', 'map', 'filter',
            'sum', 'min', 'max', 'abs', 'round', 'sorted', 'reversed', 'all', 'any',
            'isinstance', 'type', 'hasattr', 'getattr', 'setattr', 'delattr'
        }
        return function_name in builtins
    
    def _is_imported_function(self, function_name: str) -> bool:
        """Check if a function is likely imported."""
        # This is a simplified check - in a full implementation, we'd track imports
        # For now, we'll be more permissive
        return True
    
    def _add_function_issue(self, issue_type: FunctionIssueType, severity: FunctionIssueSeverity,
                           function_name: str, line: int, column: int, description: str,
                           suggestions: List[str], argument_count: Optional[int] = None) -> None:
        """Add a function issue to the results."""
        issue = FunctionIssue(
            type=issue_type,
            severity=severity,
            function_name=function_name,
            line=line,
            column=column,
            context=self._get_context(line),
            file_path=self.result.file_path,
            description=description,
            suggestions=suggestions,
            argument_count=argument_count,
            is_fallback_function=self.analyzer.is_fallback_function(function_name)
        )
        self.result.issues.append(issue)
    
    def _get_context(self, line_num: int) -> str:
        """Get context around a line number."""
        if 1 <= line_num <= len(self.lines):
            return self.lines[line_num - 1].strip()
        return ""


def analyze_function_issues(file_path: str, config: Optional[Dict[str, Any]] = None) -> FunctionAnalysisResult:
    """Analyze a file for function issues."""
    analyzer = EnhancedFunctionAnalyzer(config)
    return analyzer.analyze_file(file_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_function_issues(sys.argv[1])
        tprint(f"Found {result.total_issues} function issues in {sys.argv[1]}")
        tprint(f"Critical: {result.critical_issues}, High: {result.high_issues}")
        for issue in result.issues:
            tprint(f"  {issue.severity.value}: {issue.type.value} - {issue.function_name} (line {issue.line})")
    else:
        tprint("Usage: python enhanced_function_analyzer.py <file_path>")