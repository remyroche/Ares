#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Stub Object and Mock Object Analyzer

This analyzer specifically identifies and categorizes stub classes, mock objects, and placeholder implementations.
These should be flagged as issues but in a separate category since they indicate incomplete implementations
or missing dependencies rather than code quality problems.

Categories:
1. Stub Classes - Incomplete class implementations
2. Mock Objects - Test doubles and mocks
3. Placeholder Implementations - Temporary implementations
4. Fallback Classes - Graceful degradation classes
5. Interface Stubs - Interface implementations with minimal functionality
"""

import ast
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd


class StubObjectType(Enum):
    """Types of stub objects."""
    STUB_CLASS = "stub_class"
    MOCK_OBJECT = "mock_object"
    PLACEHOLDER_IMPLEMENTATION = "placeholder_implementation"
    FALLBACK_CLASS = "fallback_class"
    INTERFACE_STUB = "interface_stub"
    DUMMY_CLASS = "dummy_class"
    TEST_DOUBLE = "test_double"
    MINIMAL_IMPLEMENTATION = "minimal_implementation"


class StubObjectSeverity(Enum):
    """Severity levels for stub objects."""
    INFO = "info"  # Normal stub (expected)
    WARNING = "warning"  # Stub that should be replaced
    ERROR = "error"  # Stub that indicates a real problem


class StubObjectCategory(Enum):
    """Categories for stub objects."""
    DEPENDENCY_MISSING = "dependency_missing"
    INCOMPLETE_IMPLEMENTATION = "incomplete_implementation"
    TEST_ARTIFACT = "test_artifact"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    INTERFACE_PLACEHOLDER = "interface_placeholder"
    TEMPORARY_IMPLEMENTATION = "temporary_implementation"


@dataclass
class StubObject:
    """Represents a detected stub object."""
    type: StubObjectType
    severity: StubObjectSeverity
    category: StubObjectCategory
    name: str
    line: int
    column: int = 0
    context: str = ""
    file_path: str = ""
    description: str = ""
    suggestions: List[str] = field(default_factory=list)
    is_expected: bool = False  # Whether this stub is expected/acceptable
    missing_dependency: Optional[str] = None
    expected_interface: Optional[str] = None


@dataclass
class StubObjectAnalysisResult:
    """Results from stub object analysis."""
    file_path: str
    stub_objects: List[StubObject] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def total_stubs(self) -> int:
        return len(self.stub_objects)
    
    @property
    def stubs_by_type(self) -> Dict[StubObjectType, int]:
        type_counts = {}
        for stub in self.stub_objects:
            type_counts[stub.type] = type_counts.get(stub.type, 0) + 1
        return type_counts
    
    @property
    def stubs_by_category(self) -> Dict[StubObjectCategory, int]:
        category_counts = {}
        for stub in self.stub_objects:
            category_counts[stub.category] = category_counts.get(stub.category, 0) + 1
        return category_counts
    
    @property
    def expected_stubs(self) -> int:
        return len([stub for stub in self.stub_objects if stub.is_expected])
    
    @property
    def unexpected_stubs(self) -> int:
        return len([stub for stub in self.stub_objects if not stub.is_expected])


class StubObjectAnalyzer:
    """Analyzer for stub objects and mock implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the stub object analyzer."""
        self.config = config or {}
        
        # Common stub class patterns
        self.stub_class_patterns = {
            '_PDStub', '_NPStub', '_Stub', 'Mock', 'Dummy', 'Placeholder',
            'Fallback', 'Default', 'Safe', 'Conditional', 'Stub', 'Fake'
        }
        
        # Common mock object patterns
        self.mock_object_patterns = {
            'Mock', 'MagicMock', 'patch', 'mocker', 'fixture', 'spy',
            'stub', 'fake', 'double', 'test_double'
        }
        
        # Common placeholder patterns
        self.placeholder_patterns = {
            'placeholder', 'todo', 'fixme', 'hack', 'temp', 'temporary',
            'not_implemented', 'raise_not_implemented'
        }
        
        # Common fallback patterns
        self.fallback_patterns = {
            'fallback', 'backup', 'alternative', 'default', 'safe',
            'graceful', 'conditional', 'optional'
        }
        
        # Common interface patterns
        self.interface_patterns = {
            'interface', 'abstract', 'base', 'protocol', 'contract'
        }
    
    def analyze_file(self, file_path: str) -> StubObjectAnalysisResult:
        """Analyze a file for stub objects."""
        start_time = time.time()
        result = StubObjectAnalysisResult(file_path=file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            visitor = StubObjectVisitor(self, result, content)
            visitor.visit(tree)
            
        except Exception as e:
            result.error = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def is_stub_class(self, name: str) -> bool:
        """Check if a name is likely a stub class."""
        return any(pattern in name for pattern in self.stub_class_patterns)
    
    def is_mock_object(self, name: str) -> bool:
        """Check if a name is likely a mock object."""
        return any(pattern in name.lower() for pattern in self.mock_object_patterns)
    
    def is_placeholder(self, name: str) -> bool:
        """Check if a name is likely a placeholder."""
        return any(pattern in name.lower() for pattern in self.placeholder_patterns)
    
    def is_fallback_class(self, name: str) -> bool:
        """Check if a name is likely a fallback class."""
        return any(pattern in name.lower() for pattern in self.fallback_patterns)
    
    def is_interface_stub(self, name: str) -> bool:
        """Check if a name is likely an interface stub."""
        return any(pattern in name.lower() for pattern in self.interface_patterns)


class StubObjectVisitor(ast.NodeVisitor):
    """AST visitor for detecting stub objects."""
    
    def __init__(self, analyzer: StubObjectAnalyzer, result: StubObjectAnalysisResult, content: str):
        self.analyzer = analyzer
        self.result = result
        self.content = content
        self.lines = content.split('\n')
        self.in_try_except_import = False
        self.import_context = {}
    
    def visit_Try(self, node: ast.Try) -> None:
        """Visit try-except blocks to detect import-related stubs."""
        # Check if this is a try-except ImportError pattern
        has_import_error = False
        for handler in node.handlers:
            if isinstance(handler.type, ast.Name) and handler.type.id == 'ImportError':
                has_import_error = True
                self.in_try_except_import = True
                break
        
        if has_import_error:
            # Analyze the except block for stub implementations
            for handler in node.handlers:
                if isinstance(handler.type, ast.Name) and handler.type.id == 'ImportError':
                    self._analyze_import_error_stubs(handler)
        
        self.generic_visit(node)
        self.in_try_except_import = False
    
    def _analyze_import_error_stubs(self, handler: ast.ExceptHandler) -> None:
        """Analyze an ImportError handler for stub implementations."""
        for stmt in handler.body:
            if isinstance(stmt, ast.ClassDef):
                self._analyze_stub_class(stmt, is_import_fallback=True)
            elif isinstance(stmt, ast.FunctionDef):
                self._analyze_stub_function(stmt, is_import_fallback=True)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to detect stub classes."""
        self._analyze_stub_class(node, is_import_fallback=False)
        self.generic_visit(node)
    
    def _analyze_stub_class(self, node: ast.ClassDef, is_import_fallback: bool = False) -> None:
        """Analyze a class to determine if it's a stub."""
        class_name = node.name
        
        # Determine the type of stub
        stub_type = None
        category = None
        severity = StubObjectSeverity.INFO
        is_expected = False
        missing_dependency = None
        
        if self.analyzer.is_stub_class(class_name):
            stub_type = StubObjectType.STUB_CLASS
            if is_import_fallback:
                category = StubObjectCategory.DEPENDENCY_MISSING
                severity = StubObjectSeverity.INFO
                is_expected = True
                missing_dependency = self._infer_missing_dependency(class_name)
            else:
                category = StubObjectCategory.INCOMPLETE_IMPLEMENTATION
                severity = StubObjectSeverity.WARNING
                is_expected = False
        
        elif self.analyzer.is_mock_object(class_name):
            stub_type = StubObjectType.MOCK_OBJECT
            category = StubObjectCategory.TEST_ARTIFACT
            severity = StubObjectSeverity.INFO
            is_expected = True
        
        elif self.analyzer.is_placeholder(class_name):
            stub_type = StubObjectType.PLACEHOLDER_IMPLEMENTATION
            category = StubObjectCategory.TEMPORARY_IMPLEMENTATION
            severity = StubObjectSeverity.WARNING
            is_expected = False
        
        elif self.analyzer.is_fallback_class(class_name):
            stub_type = StubObjectType.FALLBACK_CLASS
            category = StubObjectCategory.GRACEFUL_DEGRADATION
            severity = StubObjectSeverity.INFO
            is_expected = True
        
        elif self.analyzer.is_interface_stub(class_name):
            stub_type = StubObjectType.INTERFACE_STUB
            category = StubObjectCategory.INTERFACE_PLACEHOLDER
            severity = StubObjectSeverity.INFO
            is_expected = True
        
        # Check for minimal implementations
        elif self._is_minimal_class_implementation(node):
            stub_type = StubObjectType.MINIMAL_IMPLEMENTATION
            category = StubObjectCategory.INCOMPLETE_IMPLEMENTATION
            severity = StubObjectSeverity.WARNING
            is_expected = False
        
        if stub_type:
            self._add_stub_object(
                stub_type=stub_type,
                category=category,
                severity=severity,
                name=class_name,
                line=node.lineno,
                column=node.col_offset,
                description=self._generate_stub_description(stub_type, class_name, category),
                suggestions=self._generate_stub_suggestions(stub_type, category, is_expected),
                is_expected=is_expected,
                missing_dependency=missing_dependency
            )
    
    def _analyze_stub_function(self, node: ast.FunctionDef, is_import_fallback: bool = False) -> None:
        """Analyze a function to determine if it's a stub."""
        func_name = node.name
        
        # Check for minimal implementations that might be stubs
        if self._is_minimal_function_implementation(node):
            stub_type = StubObjectType.MINIMAL_IMPLEMENTATION
            category = StubObjectCategory.INCOMPLETE_IMPLEMENTATION
            severity = StubObjectSeverity.WARNING
            is_expected = False
            
            if is_import_fallback:
                category = StubObjectCategory.DEPENDENCY_MISSING
                severity = StubObjectSeverity.INFO
                is_expected = True
            
            self._add_stub_object(
                stub_type=stub_type,
                category=category,
                severity=severity,
                name=func_name,
                line=node.lineno,
                column=node.col_offset,
                description=f"Minimal function implementation: {func_name}",
                suggestions=self._generate_stub_suggestions(stub_type, category, is_expected),
                is_expected=is_expected
            )
    
    def _is_minimal_class_implementation(self, node: ast.ClassDef) -> bool:
        """Check if a class is a minimal implementation (likely stub)."""
        # Classes with only pass statements or minimal methods
        if len(node.body) <= 3:
            for stmt in node.body:
                if isinstance(stmt, ast.Pass):
                    continue
                elif isinstance(stmt, ast.FunctionDef):
                    if self._is_minimal_function_implementation(stmt):
                        continue
                    else:
                        return False
                else:
                    return False
            return True
        
        # Check for classes with only docstrings and pass
        if len(node.body) <= 2:
            for stmt in node.body:
                if isinstance(stmt, (ast.Pass, ast.Expr)):
                    continue
                else:
                    return False
            return True
        
        return False
    
    def _is_minimal_function_implementation(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a minimal implementation (likely stub)."""
        # Functions with only pass
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            return True
        
        # Functions with only return None
        if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
            if isinstance(node.body[0].value, ast.Constant) and node.body[0].value.value is None:
                return True
        
        # Functions that just return the first parameter (common in decorator stubs)
        if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
            if isinstance(node.body[0].value, ast.Name):
                if node.args.args and node.body[0].value.id == node.args.args[0].arg:
                    return True
        
        # Functions with only raise NotImplementedError
        if len(node.body) == 1 and isinstance(node.body[0], ast.Raise):
            if isinstance(node.body[0].exc, ast.Name) and node.body[0].exc.id == 'NotImplementedError':
                return True
        
        return False
    
    def _infer_missing_dependency(self, class_name: str) -> Optional[str]:
        """Infer the missing dependency from the class name."""
        if '_PDStub' in class_name or 'DataFrame' in class_name or 'Series' in class_name:
            return 'pandas'
        elif '_NPStub' in class_name or 'ndarray' in class_name:
            return 'numpy'
        elif 'MLflow' in class_name or 'mlflow' in class_name:
            return 'mlflow'
        elif 'PSUtil' in class_name or 'psutil' in class_name:
            return 'psutil'
        return None
    
    def _generate_stub_description(self, stub_type: StubObjectType, name: str, category: StubObjectCategory) -> str:
        """Generate a description for a stub object."""
        descriptions = {
            StubObjectType.STUB_CLASS: f"Stub class implementation: {name}",
            StubObjectType.MOCK_OBJECT: f"Mock object: {name}",
            StubObjectType.PLACEHOLDER_IMPLEMENTATION: f"Placeholder implementation: {name}",
            StubObjectType.FALLBACK_CLASS: f"Fallback class: {name}",
            StubObjectType.INTERFACE_STUB: f"Interface stub: {name}",
            StubObjectType.DUMMY_CLASS: f"Dummy class: {name}",
            StubObjectType.TEST_DOUBLE: f"Test double: {name}",
            StubObjectType.MINIMAL_IMPLEMENTATION: f"Minimal implementation: {name}"
        }
        return descriptions.get(stub_type, f"Stub object: {name}")
    
    def _generate_stub_suggestions(self, stub_type: StubObjectType, category: StubObjectCategory, is_expected: bool) -> List[str]:
        """Generate suggestions for a stub object."""
        if is_expected:
            return [
                "This stub is expected and acceptable",
                "Consider documenting why this stub is needed",
                "Ensure the stub interface matches the expected interface"
            ]
        
        suggestions = []
        
        if category == StubObjectCategory.DEPENDENCY_MISSING:
            suggestions.extend([
                "Install the missing dependency",
                "Consider making the dependency optional",
                "Document the dependency requirement"
            ])
        elif category == StubObjectCategory.INCOMPLETE_IMPLEMENTATION:
            suggestions.extend([
                "Complete the implementation",
                "Add proper error handling",
                "Add comprehensive tests"
            ])
        elif category == StubObjectCategory.TEMPORARY_IMPLEMENTATION:
            suggestions.extend([
                "Replace with proper implementation",
                "Add TODO comment with timeline",
                "Consider if this is still needed"
            ])
        elif category == StubObjectCategory.INTERFACE_PLACEHOLDER:
            suggestions.extend([
                "Implement the full interface",
                "Add proper documentation",
                "Consider using abstract base classes"
            ])
        
        return suggestions
    
    def _add_stub_object(self, stub_type: StubObjectType, category: StubObjectCategory,
                        severity: StubObjectSeverity, name: str, line: int, column: int,
                        description: str, suggestions: List[str], is_expected: bool = False,
                        missing_dependency: Optional[str] = None) -> None:
        """Add a stub object to the results."""
        stub = StubObject(
            type=stub_type,
            severity=severity,
            category=category,
            name=name,
            line=line,
            column=column,
            context=self._get_context(line),
            file_path=self.result.file_path,
            description=description,
            suggestions=suggestions,
            is_expected=is_expected,
            missing_dependency=missing_dependency
        )
        self.result.stub_objects.append(stub)
    
    def _get_context(self, line_num: int) -> str:
        """Get context around a line number."""
        if 1 <= line_num <= len(self.lines):
            return self.lines[line_num - 1].strip()
        return ""


def analyze_stub_objects(file_path: str, config: Optional[Dict[str, Any]] = None) -> StubObjectAnalysisResult:
    """Analyze a file for stub objects."""
    analyzer = StubObjectAnalyzer(config)
    return analyzer.analyze_file(file_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_stub_objects(sys.argv[1])
        tprint(f"Found {result.total_stubs} stub objects in {sys.argv[1]}")
        tprint(f"Expected stubs: {result.expected_stubs}")
        tprint(f"Unexpected stubs: {result.unexpected_stubs}")
        
        for stub in result.stub_objects:
            status = "EXPECTED" if stub.is_expected else "UNEXPECTED"
            tprint(f"  {status} {stub.type.value}: {stub.name} (line {stub.line}) - {stub.category.value}")
    else:
        tprint("Usage: python stub_object_analyzer.py <file_path>")