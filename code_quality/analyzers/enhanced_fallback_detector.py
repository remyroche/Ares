#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Enhanced Fallback Pattern Detector

This analyzer specifically identifies and categorizes fallback patterns to reduce false positives
in code quality analysis. It recognizes:

1. Fallback decorator implementations
2. Stub classes and mock objects
3. Conditional import patterns
4. Graceful degradation patterns
5. Dynamic import fallbacks

This helps distinguish between real issues and intentional fallback code.
"""

import ast
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class FallbackType(Enum):
    """Types of fallback patterns."""
    DECORATOR_FALLBACK = "decorator_fallback"
    STUB_CLASS = "stub_class"
    MOCK_OBJECT = "mock_object"
    CONDITIONAL_IMPORT = "conditional_import"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    LAMBDA_FALLBACK = "lambda_fallback"
    TRY_EXCEPT_IMPORT = "try_except_import"


class FallbackSeverity(Enum):
    """Severity levels for fallback patterns."""
    INFO = "info"  # Normal fallback pattern
    WARNING = "warning"  # Fallback that could be improved
    ERROR = "error"  # Fallback that indicates a real problem


@dataclass
class FallbackPattern:
    """Represents a detected fallback pattern."""
    type: FallbackType
    severity: FallbackSeverity
    name: str
    line: int
    column: int = 0
    context: str = ""
    file_path: str = ""
    description: str = ""
    suggestions: List[str] = field(default_factory=list)
    is_intentional: bool = True  # Most fallbacks are intentional


@dataclass
class FallbackAnalysisResult:
    """Results from fallback pattern analysis."""
    file_path: str
    patterns: List[FallbackPattern] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def total_patterns(self) -> int:
        return len(self.patterns)
    
    @property
    def patterns_by_type(self) -> Dict[FallbackType, int]:
        type_counts = {}
        for pattern in self.patterns:
            type_counts[pattern.type] = type_counts.get(pattern.type, 0) + 1
        return type_counts


class EnhancedFallbackDetector:
    """Enhanced detector for fallback patterns and graceful degradation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the fallback detector."""
        self.config = config or {}
        self.fallback_indicators = {
            'fallback', 'stub', 'mock', 'dummy', 'placeholder', 'default',
            'backup', 'alternative', 'graceful', 'safe', 'conditional'
        }
        
        # Common fallback decorator patterns
        self.fallback_decorator_patterns = {
            'handles_errors', 'monitor_feature_engineering', 'validates',
            'traced', 'log_execution_time', 'cached', 'ensure_data_integrity',
            'monitor_step_execution', 'secure_step_execution', 'comprehensive_data_validation',
            'handle_errors', 'memory_efficient', 'resource_monitor', 'secure_data_processing',
            'validate_data_structure', 'with_tracing_span', 'quality_gate',
            'validate_pipeline_step', 'with_enhanced_mlflow_logging'
        }
        
        # Common stub class patterns
        self.stub_class_patterns = {
            '_PDStub', '_NPStub', '_Stub', 'Mock', 'Dummy', 'Placeholder',
            'Fallback', 'Default', 'Safe', 'Conditional'
        }
    
    def analyze_file(self, file_path: str) -> FallbackAnalysisResult:
        """Analyze a file for fallback patterns."""
        start_time = time.time()
        result = FallbackAnalysisResult(file_path=file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            visitor = FallbackPatternVisitor(self, result, content)
            visitor.visit(tree)
            
        except Exception as e:
            result.error = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def is_fallback_decorator(self, name: str) -> bool:
        """Check if a name is likely a fallback decorator."""
        return name in self.fallback_decorator_patterns
    
    def is_stub_class(self, name: str) -> bool:
        """Check if a name is likely a stub class."""
        return any(pattern in name for pattern in self.stub_class_patterns)
    
    def is_fallback_indicator(self, name: str) -> bool:
        """Check if a name contains fallback indicators."""
        name_lower = name.lower()
        return any(indicator in name_lower for indicator in self.fallback_indicators)


class FallbackPatternVisitor(ast.NodeVisitor):
    """AST visitor for detecting fallback patterns."""
    
    def __init__(self, detector: EnhancedFallbackDetector, result: FallbackAnalysisResult, content: str):
        self.detector = detector
        self.result = result
        self.content = content
        self.lines = content.split('\n')
        self.in_try_except_import = False
        self.import_error_handlers = set()
        self.fallback_functions = set()
        self.stub_classes = set()
    
    def visit_Try(self, node: ast.Try) -> None:
        """Visit try-except blocks to detect import fallbacks."""
        # Check if this is a try-except ImportError pattern
        has_import_error = False
        for handler in node.handlers:
            if isinstance(handler.type, ast.Name) and handler.type.id == 'ImportError':
                has_import_error = True
                self.in_try_except_import = True
                break
        
        if has_import_error:
            # Analyze the except block for fallback patterns
            for handler in node.handlers:
                if isinstance(handler.type, ast.Name) and handler.type.id == 'ImportError':
                    self._analyze_import_error_handler(handler)
        
        self.generic_visit(node)
        self.in_try_except_import = False
    
    def _analyze_import_error_handler(self, handler: ast.ExceptHandler) -> None:
        """Analyze an ImportError handler for fallback patterns."""
        for stmt in handler.body:
            if isinstance(stmt, ast.FunctionDef):
                self._analyze_fallback_function(stmt)
            elif isinstance(stmt, ast.ClassDef):
                self._analyze_stub_class(stmt)
            elif isinstance(stmt, ast.Assign):
                self._analyze_fallback_assignment(stmt)
    
    def _analyze_fallback_function(self, node: ast.FunctionDef) -> None:
        """Analyze a function that might be a fallback implementation."""
        func_name = node.name
        
        # Check if it's a known fallback decorator
        if self.detector.is_fallback_decorator(func_name):
            self.result.patterns.append(FallbackPattern(
                type=FallbackType.DECORATOR_FALLBACK,
                severity=FallbackSeverity.INFO,
                name=func_name,
                line=node.lineno,
                column=node.col_offset,
                context=self._get_context(node.lineno),
                file_path=self.result.file_path,
                description=f"Fallback decorator implementation for '{func_name}'",
                suggestions=[
                    "This is an intentional fallback when the real decorator is unavailable",
                    "Consider documenting why this fallback is needed",
                    "Ensure the fallback behavior matches the expected interface"
                ]
            ))
            self.fallback_functions.add(func_name)
        
        # Check for minimal implementations (common in fallbacks)
        elif self._is_minimal_implementation(node):
            self.result.patterns.append(FallbackPattern(
                type=FallbackType.DECORATOR_FALLBACK,
                severity=FallbackSeverity.INFO,
                name=func_name,
                line=node.lineno,
                column=node.col_offset,
                context=self._get_context(node.lineno),
                file_path=self.result.file_path,
                description=f"Minimal fallback implementation for '{func_name}'",
                suggestions=[
                    "This appears to be a fallback implementation",
                    "Consider adding a comment explaining the fallback purpose"
                ]
            ))
    
    def _analyze_stub_class(self, node: ast.ClassDef) -> None:
        """Analyze a class that might be a stub implementation."""
        class_name = node.name
        
        if self.detector.is_stub_class(class_name):
            self.result.patterns.append(FallbackPattern(
                type=FallbackType.STUB_CLASS,
                severity=FallbackSeverity.INFO,
                name=class_name,
                line=node.lineno,
                column=node.col_offset,
                context=self._get_context(node.lineno),
                file_path=self.result.file_path,
                description=f"Stub class implementation for '{class_name}'",
                suggestions=[
                    "This is an intentional stub when the real class is unavailable",
                    "Consider documenting the expected interface",
                    "Ensure stub methods match the real class interface"
                ]
            ))
            self.stub_classes.add(class_name)
        
        # Check for classes with minimal implementations
        elif self._is_minimal_class(node):
            self.result.patterns.append(FallbackPattern(
                type=FallbackType.STUB_CLASS,
                severity=FallbackSeverity.INFO,
                name=class_name,
                line=node.lineno,
                column=node.col_offset,
                context=self._get_context(node.lineno),
                file_path=self.result.file_path,
                description=f"Minimal class implementation for '{class_name}'",
                suggestions=[
                    "This appears to be a stub or fallback class",
                    "Consider adding documentation for the expected interface"
                ]
            ))
    
    def _analyze_fallback_assignment(self, node: ast.Assign) -> None:
        """Analyze assignments that might be fallback patterns."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                target_name = target.name
                
                # Check for lambda fallbacks
                if isinstance(node.value, ast.Lambda):
                    if self.detector.is_fallback_indicator(target_name):
                        self.result.patterns.append(FallbackPattern(
                            type=FallbackType.LAMBDA_FALLBACK,
                            severity=FallbackSeverity.INFO,
                            name=target_name,
                            line=node.lineno,
                            column=node.col_offset,
                            context=self._get_context(node.lineno),
                            file_path=self.result.file_path,
                            description=f"Lambda fallback for '{target_name}'",
                            suggestions=[
                                "This is an intentional lambda fallback",
                                "Consider documenting the expected behavior"
                            ]
                        ))
                
                # Check for function fallbacks
                elif isinstance(node.value, ast.Call):
                    if self.detector.is_fallback_indicator(target_name):
                        self.result.patterns.append(FallbackPattern(
                            type=FallbackType.GRACEFUL_DEGRADATION,
                            severity=FallbackSeverity.INFO,
                            name=target_name,
                            line=node.lineno,
                            column=node.col_offset,
                            context=self._get_context(node.lineno),
                            file_path=self.result.file_path,
                            description=f"Graceful degradation for '{target_name}'",
                            suggestions=[
                                "This is an intentional fallback assignment",
                                "Consider documenting the fallback behavior"
                            ]
                        ))
    
    def _is_minimal_implementation(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a minimal implementation (likely fallback)."""
        # Check for functions with very simple bodies
        if len(node.body) <= 2:
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    # Simple return statements are common in fallbacks
                    return True
                elif isinstance(stmt, ast.Pass):
                    return True
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    # Simple constant expressions
                    return True
        
        # Check for functions that just return the input (common in decorator fallbacks)
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Name):
                    # Return the first parameter (common in decorator fallbacks)
                    if node.args.args and stmt.value.id == node.args.args[0].arg:
                        return True
        
        return False
    
    def _is_minimal_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is a minimal implementation (likely stub)."""
        # Classes with only pass statements or minimal methods
        if len(node.body) <= 3:
            for stmt in node.body:
                if isinstance(stmt, ast.Pass):
                    continue
                elif isinstance(stmt, ast.FunctionDef):
                    if self._is_minimal_implementation(stmt):
                        continue
                    else:
                        return False
                else:
                    return False
            return True
        
        return False
    
    def _get_context(self, line_num: int) -> str:
        """Get context around a line number."""
        if 1 <= line_num <= len(self.lines):
            return self.lines[line_num - 1].strip()
        return ""


def analyze_fallback_patterns(file_path: str, config: Optional[Dict[str, Any]] = None) -> FallbackAnalysisResult:
    """Analyze a file for fallback patterns."""
    detector = EnhancedFallbackDetector(config)
    return detector.analyze_file(file_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_fallback_patterns(sys.argv[1])
        tprint(f"Found {result.total_patterns} fallback patterns in {sys.argv[1]}")
        for pattern in result.patterns:
            tprint(f"  {pattern.type.value}: {pattern.name} (line {pattern.line})")
    else:
        tprint("Usage: python enhanced_fallback_detector.py <file_path>")