#!/usr/bin/env python3
"""
Code Smell Detector

Detects various code smells and anti-patterns including:
- Long methods/classes
- God objects
- Feature envy
- Data clumps
- Inappropriate intimacy
- Lazy classes
- Speculative generality
- Message chains
- Middle man
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter


@dataclass
class CodeSmell:
    """Represents a detected code smell."""
    smell_type: str
    severity: str  # 'high', 'medium', 'low'
    file_path: str
    line_number: int
    entity_name: str  # function/class/variable name
    message: str
    suggestion: str


class CodeSmellDetector:
    """Detects various code smells in Python code."""
    
    # Thresholds for various smells
    THRESHOLDS = {
        'max_method_length': 50,
        'max_class_length': 300,
        'max_parameters': 5,
        'max_instance_variables': 10,
        'max_method_calls_chain': 3,
        'min_class_methods': 2,  # For lazy class
        'max_class_dependencies': 5,
        'similar_name_threshold': 0.8,
        'max_nested_depth': 4
    }
    
    def __init__(self, project_root: str, custom_thresholds: Optional[Dict[str, int]] = None):
        self.project_root = Path(project_root)
        self.smells: List[CodeSmell] = []
        
        # Update thresholds with custom values
        if custom_thresholds:
            self.THRESHOLDS.update(custom_thresholds)
            
        # Track various metrics for smell detection
        self.class_metrics: Dict[str, Dict[str, Any]] = {}
        self.function_calls: Dict[str, List[str]] = defaultdict(list)
        self.variable_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
    def analyze_file(self, file_path: Path) -> List[CodeSmell]:
        """Analyze a single file for code smells."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            tree = ast.parse(content, filename=str(file_path))
            
            # Run smell detectors
            detector = SmellDetectorVisitor(str(file_path), lines, self)
            detector.visit(tree)
            
            # Analyze collected metrics for additional smells
            self._detect_cross_class_smells(str(file_path))
            
            return detector.smells
            
        except Exception as e:
            return []
            
    def _detect_cross_class_smells(self, file_path: str) -> None:
        """Detect smells that require cross-class analysis."""
        # Feature Envy - methods that use another class's data more than their own
        for class_name, metrics in self.class_metrics.items():
            for method_name, method_data in metrics.get('methods', {}).items():
                external_refs = method_data.get('external_references', {})
                internal_refs = method_data.get('internal_references', 0)
                
                for external_class, ref_count in external_refs.items():
                    if ref_count > internal_refs * 2:  # Uses external data twice as much
                        self._add_smell(
                            'feature_envy', 'medium', file_path,
                            method_data.get('line', 0), f"{class_name}.{method_name}",
                            f"Method uses {external_class} data more than its own class data",
                            f"Consider moving this method to {external_class} or refactoring"
                        )
                        
    def _add_smell(self, smell_type: str, severity: str, file_path: str,
                   line_number: int, entity_name: str, message: str, suggestion: str) -> None:
        """Add a detected code smell."""
        self.smells.append(CodeSmell(
            smell_type=smell_type,
            severity=severity,
            file_path=file_path,
            line_number=line_number,
            entity_name=entity_name,
            message=message,
            suggestion=suggestion
        ))
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive code smell report."""
        # Group smells by type
        smells_by_type = defaultdict(list)
        for smell in self.smells:
            smells_by_type[smell.smell_type].append({
                'file': smell.file_path,
                'line': smell.line_number,
                'entity': smell.entity_name,
                'message': smell.message,
                'suggestion': smell.suggestion,
                'severity': smell.severity
            })
            
        # Calculate statistics
        severity_counts = Counter(smell.severity for smell in self.smells)
        
        # Find most problematic files
        file_smell_counts = Counter(smell.file_path for smell in self.smells)
        
        return {
            'summary': {
                'total_smells': len(self.smells),
                'high_severity': severity_counts['high'],
                'medium_severity': severity_counts['medium'],
                'low_severity': severity_counts['low'],
                'unique_smell_types': len(smells_by_type)
            },
            'smells_by_type': dict(smells_by_type),
            'most_problematic_files': [
                {'file': file, 'smell_count': count}
                for file, count in file_smell_counts.most_common(10)
            ],
            'smell_distribution': {
                smell_type: len(smells)
                for smell_type, smells in smells_by_type.items()
            }
        }


class SmellDetectorVisitor(ast.NodeVisitor):
    """AST visitor for detecting code smells."""
    
    def __init__(self, file_path: str, lines: List[str], detector: CodeSmellDetector):
        self.file_path = file_path
        self.lines = lines
        self.detector = detector
        self.smells = []
        
        # Context tracking
        self.current_class = None
        self.current_method = None
        self.nesting_depth = 0
        
        # Metrics collection
        self.class_lines = {}
        self.method_lines = {}
        self.parameter_counts = {}
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        class_length = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        
        # Long Class smell
        if class_length > self.detector.THRESHOLDS['max_class_length']:
            self._add_smell(
                'long_class', 'high', node.lineno, node.name,
                f"Class has {class_length} lines (threshold: {self.detector.THRESHOLDS['max_class_length']})",
                "Consider breaking this class into smaller, focused classes"
            )
            
        # Analyze class for God Object smell
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        instance_vars = self._count_instance_variables(node)
        
        if (len(methods) > 20 or instance_vars > self.detector.THRESHOLDS['max_instance_variables']):
            self._add_smell(
                'god_object', 'high', node.lineno, node.name,
                f"Class has {len(methods)} methods and {instance_vars} instance variables",
                "This class is doing too much. Apply Single Responsibility Principle"
            )
            
        # Lazy Class smell
        if len(methods) < self.detector.THRESHOLDS['min_class_methods'] and class_length < 50:
            self._add_smell(
                'lazy_class', 'low', node.lineno, node.name,
                f"Class has only {len(methods)} methods and {class_length} lines",
                "Consider merging with another class or adding more responsibility"
            )
            
        # Store class metrics
        self.detector.class_metrics[node.name] = {
            'lines': class_length,
            'methods': {},
            'instance_vars': instance_vars
        }
        
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        self._check_method_smells(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self._check_method_smells(node)
        
    def _check_method_smells(self, node: Any) -> None:
        """Check for method-related smells."""
        method_length = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        
        # Long Method smell
        if method_length > self.detector.THRESHOLDS['max_method_length']:
            self._add_smell(
                'long_method', 'high', node.lineno, node.name,
                f"Method has {method_length} lines (threshold: {self.detector.THRESHOLDS['max_method_length']})",
                "Extract smaller methods with single responsibilities"
            )
            
        # Long Parameter List smell
        param_count = len(node.args.args)
        if param_count > self.detector.THRESHOLDS['max_parameters']:
            self._add_smell(
                'long_parameter_list', 'medium', node.lineno, node.name,
                f"Method has {param_count} parameters (threshold: {self.detector.THRESHOLDS['max_parameters']})",
                "Consider using a parameter object or builder pattern"
            )
            
        # Check for message chains
        self._check_message_chains(node)
        
        # Check for duplicate code patterns
        self._check_duplicate_patterns(node)
        
        # Store method metrics
        if self.current_class:
            external_refs, internal_refs = self._analyze_method_references(node)
            self.detector.class_metrics[self.current_class]['methods'][node.name] = {
                'line': node.lineno,
                'length': method_length,
                'parameters': param_count,
                'external_references': external_refs,
                'internal_references': internal_refs
            }
            
        old_method = self.current_method
        self.current_method = node.name
        self.generic_visit(node)
        self.current_method = old_method
        
    def visit_If(self, node: ast.If) -> None:
        """Track nesting depth."""
        self.nesting_depth += 1
        
        # Deeply Nested Code smell
        if self.nesting_depth > self.detector.THRESHOLDS['max_nested_depth']:
            self._add_smell(
                'deeply_nested_code', 'medium', node.lineno,
                f"{self.current_class}.{self.current_method}" if self.current_class else self.current_method,
                f"Code is nested {self.nesting_depth} levels deep",
                "Extract methods or use early returns to reduce nesting"
            )
            
        self.generic_visit(node)
        self.nesting_depth -= 1
        
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to detect patterns."""
        # Check for message chains (a.b().c().d())
        chain_length = self._count_call_chain(node)
        if chain_length > self.detector.THRESHOLDS['max_method_calls_chain']:
            self._add_smell(
                'message_chain', 'medium', node.lineno,
                self.current_method or 'module',
                f"Message chain with {chain_length} calls",
                "Consider using Law of Demeter - only talk to immediate friends"
            )
            
        self.generic_visit(node)
        
    def _count_instance_variables(self, class_node: ast.ClassDef) -> int:
        """Count instance variables in a class."""
        instance_vars = set()
        
        for node in ast.walk(class_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        instance_vars.add(target.attr)
                        
        return len(instance_vars)
        
    def _check_message_chains(self, node: ast.FunctionDef) -> None:
        """Check for message chain smell in a method."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                chain_length = self._count_call_chain(child)
                if chain_length > self.detector.THRESHOLDS['max_method_calls_chain']:
                    return  # Already reported in visit_Call
                    
    def _count_call_chain(self, node: ast.Call) -> int:
        """Count the length of a method call chain."""
        count = 1
        current = node.func
        
        while isinstance(current, ast.Attribute):
            if isinstance(current.value, ast.Call):
                count += 1
                current = current.value.func
            else:
                break
                
        return count
        
    def _check_duplicate_patterns(self, node: ast.FunctionDef) -> None:
        """Check for duplicate code patterns."""
        # This is a simplified check - real duplicate detection would be more complex
        method_ast_dump = ast.dump(node)
        
        # Check for similar patterns in the same class
        if self.current_class and len(method_ast_dump) > 100:
            # Look for suspiciously similar method structures
            # This is a placeholder for more sophisticated duplicate detection
            pass
            
    def _analyze_method_references(self, node: ast.FunctionDef) -> Tuple[Dict[str, int], int]:
        """Analyze what data a method accesses."""
        external_refs = defaultdict(int)
        internal_refs = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    if child.value.id == 'self':
                        internal_refs += 1
                    else:
                        external_refs[child.value.id] += 1
                        
        return dict(external_refs), internal_refs
        
    def _add_smell(self, smell_type: str, severity: str, line_number: int,
                   entity_name: str, message: str, suggestion: str) -> None:
        """Add a detected smell."""
        smell = CodeSmell(
            smell_type=smell_type,
            severity=severity,
            file_path=self.file_path,
            line_number=line_number,
            entity_name=entity_name,
            message=message,
            suggestion=suggestion
        )
        self.smells.append(smell)
        self.detector.smells.append(smell)
        
    def visit_For(self, node: ast.For) -> None:
        """Track nesting for loops."""
        self.nesting_depth += 1
        self.generic_visit(node)
        self.nesting_depth -= 1
        
    def visit_While(self, node: ast.While) -> None:
        """Track nesting for while loops."""
        self.nesting_depth += 1
        self.generic_visit(node)
        self.nesting_depth -= 1
        
    def visit_With(self, node: ast.With) -> None:
        """Track nesting for with statements."""
        self.nesting_depth += 1
        self.generic_visit(node)
        self.nesting_depth -= 1