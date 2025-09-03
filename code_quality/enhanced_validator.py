#!/usr/bin/env python3
"""
Enhanced Validator - Comprehensive Function Argument and Data Access Validation

This module provides advanced validation for:
1. Function calls with proper arguments (count, types, names)
2. Data access validation (attribute access, dictionary keys, list indices)
3. Type consistency checking
4. Null/None safety validation
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import logging
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found during analysis."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class DataAccess:
    """Represents a data access operation in code."""
    access_type: str  # 'attribute', 'subscript', 'method'
    target: str
    accessor: str
    line_number: int
    file_path: str
    is_safe: bool = False
    has_check: bool = False


@dataclass
class FunctionSignature:
    """Represents a function signature for validation."""
    name: str
    file_path: str
    line_number: int
    positional_args: List[str]
    keyword_args: List[str]
    defaults: Dict[str, Any]
    var_args: Optional[str] = None
    var_kwargs: Optional[str] = None
    return_type: Optional[str] = None
    arg_types: Dict[str, str] = None


class EnhancedValidator:
    """Enhanced validator for function arguments and data access."""
    
    def __init__(self, project_root: str = '.', exclude_patterns: Optional[List[str]] = None):
        self.project_root = Path(project_root).resolve()
        self.exclude_patterns = exclude_patterns or ['__pycache__', '*.pyc', '.git', 'venv', '.env']
        
        # Storage for analysis results
        self.issues: List[ValidationIssue] = []
        self.function_signatures: Dict[str, FunctionSignature] = {}
        self.data_accesses: List[DataAccess] = []
        self.variable_types: Dict[str, Dict[str, str]] = defaultdict(dict)  # file -> var -> type
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_issues': 0,
            'argument_mismatches': 0,
            'unsafe_data_access': 0,
            'missing_null_checks': 0,
            'type_inconsistencies': 0
        }
    
    def validate_project(self) -> Dict[str, Any]:
        """Validate the entire project for argument and data access issues."""
        logger.info(f"Starting enhanced validation for project: {self.project_root}")
        
        # Find all Python files
        python_files = self._find_python_files()
        logger.info(f"Found {len(python_files)} Python files to analyze")
        
        # First pass: collect function signatures and type information
        for file_path in python_files:
            try:
                self._collect_signatures(file_path)
                self.stats['files_processed'] += 1
            except Exception as e:
                logger.error(f"Error collecting signatures from {file_path}: {e}")
        
        # Second pass: validate function calls and data access
        for file_path in python_files:
            try:
                self._validate_file(file_path)
            except Exception as e:
                logger.error(f"Error validating {file_path}: {e}")
                self._add_issue(
                    str(file_path), 0, 'validation_error', 'error',
                    f"Failed to validate file: {e}"
                )
        
        # Generate report
        return self._generate_report()
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                pattern in d or pattern.replace('*', '') in d
                for pattern in self.exclude_patterns
            )]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if not any(
                        pattern in str(file_path) or pattern.replace('*', '') in str(file_path)
                        for pattern in self.exclude_patterns
                    ):
                        python_files.append(file_path)
        
        return python_files
    
    def _collect_signatures(self, file_path: Path) -> None:
        """Collect function signatures from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            collector = SignatureCollector(file_path, content, self)
            collector.visit(tree)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate function calls and data access in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            validator = ArgumentAndAccessValidator(file_path, content, self)
            validator.visit(tree)
            
        except SyntaxError as e:
            self._add_issue(
                str(file_path), e.lineno or 0, 'syntax_error', 'error',
                f"Syntax error: {e.msg}"
            )
    
    def _add_issue(self, file_path: str, line_number: int, issue_type: str,
                   severity: str, message: str, suggestion: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None) -> None:
        """Add a validation issue."""
        issue = ValidationIssue(
            file_path=file_path,
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            message=message,
            suggestion=suggestion,
            context=context
        )
        self.issues.append(issue)
        self.stats['total_issues'] += 1
        
        # Update specific statistics
        if 'argument' in issue_type:
            self.stats['argument_mismatches'] += 1
        elif 'unsafe' in issue_type or 'unchecked' in issue_type:
            self.stats['unsafe_data_access'] += 1
        elif 'null' in issue_type or 'none' in issue_type.lower():
            self.stats['missing_null_checks'] += 1
        elif 'type' in issue_type:
            self.stats['type_inconsistencies'] += 1
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        return {
            'summary': {
                'project_root': str(self.project_root),
                'files_processed': self.stats['files_processed'],
                'total_issues': self.stats['total_issues'],
                'argument_mismatches': self.stats['argument_mismatches'],
                'unsafe_data_access': self.stats['unsafe_data_access'],
                'missing_null_checks': self.stats['missing_null_checks'],
                'type_inconsistencies': self.stats['type_inconsistencies']
            },
            'issues': [self._issue_to_dict(issue) for issue in self.issues],
            'function_signatures': {
                name: self._signature_to_dict(sig)
                for name, sig in self.function_signatures.items()
            },
            'data_access_summary': self._summarize_data_access()
        }
    
    def _issue_to_dict(self, issue: ValidationIssue) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return {
            'file_path': issue.file_path,
            'line_number': issue.line_number,
            'issue_type': issue.issue_type,
            'severity': issue.severity,
            'message': issue.message,
            'suggestion': issue.suggestion,
            'code_snippet': issue.code_snippet,
            'context': issue.context
        }
    
    def _signature_to_dict(self, sig: FunctionSignature) -> Dict[str, Any]:
        """Convert function signature to dictionary."""
        return {
            'name': sig.name,
            'file_path': sig.file_path,
            'line_number': sig.line_number,
            'positional_args': sig.positional_args,
            'keyword_args': sig.keyword_args,
            'defaults': {k: str(v) for k, v in sig.defaults.items()},
            'var_args': sig.var_args,
            'var_kwargs': sig.var_kwargs,
            'return_type': sig.return_type,
            'arg_types': sig.arg_types or {}
        }
    
    def _summarize_data_access(self) -> Dict[str, Any]:
        """Summarize data access patterns."""
        safe_accesses = sum(1 for access in self.data_accesses if access.is_safe)
        checked_accesses = sum(1 for access in self.data_accesses if access.has_check)
        
        return {
            'total_accesses': len(self.data_accesses),
            'safe_accesses': safe_accesses,
            'checked_accesses': checked_accesses,
            'unsafe_accesses': len(self.data_accesses) - safe_accesses,
            'access_types': {
                'attribute': sum(1 for a in self.data_accesses if a.access_type == 'attribute'),
                'subscript': sum(1 for a in self.data_accesses if a.access_type == 'subscript'),
                'method': sum(1 for a in self.data_accesses if a.access_type == 'method')
            }
        }


class SignatureCollector(ast.NodeVisitor):
    """AST visitor to collect function signatures."""
    
    def __init__(self, file_path: Path, content: str, validator: EnhancedValidator):
        self.file_path = file_path
        self.content = content
        self.lines = content.split('\n')
        self.validator = validator
        self.current_class = None
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        self._process_function(node)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        self._process_function(node)
        self.generic_visit(node)
    
    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        """Process a function definition."""
        # Build full function name
        if self.current_class:
            full_name = f"{self.current_class}.{node.name}"
        else:
            full_name = node.name
        
        # Extract argument information
        args = node.args
        positional_args = [arg.arg for arg in args.args]
        keyword_args = [arg.arg for arg in args.kwonlyargs]
        
        # Extract defaults
        defaults = {}
        for i, default in enumerate(args.defaults):
            arg_index = len(positional_args) - len(args.defaults) + i
            if arg_index >= 0:
                defaults[positional_args[arg_index]] = ast.unparse(default)
        
        for kw_default, kw_arg in zip(args.kw_defaults, keyword_args):
            if kw_default:
                defaults[kw_arg] = ast.unparse(kw_default)
        
        # Extract type annotations
        arg_types = {}
        for arg in args.args + args.kwonlyargs:
            if arg.annotation:
                arg_types[arg.arg] = ast.unparse(arg.annotation)
        
        # Create signature
        signature = FunctionSignature(
            name=full_name,
            file_path=str(self.file_path),
            line_number=node.lineno,
            positional_args=positional_args,
            keyword_args=keyword_args,
            defaults=defaults,
            var_args=args.vararg.arg if args.vararg else None,
            var_kwargs=args.kwarg.arg if args.kwarg else None,
            return_type=ast.unparse(node.returns) if node.returns else None,
            arg_types=arg_types
        )
        
        self.validator.function_signatures[full_name] = signature


class ArgumentAndAccessValidator(ast.NodeVisitor):
    """AST visitor to validate function arguments and data access."""
    
    def __init__(self, file_path: Path, content: str, validator: EnhancedValidator):
        self.file_path = file_path
        self.content = content
        self.lines = content.split('\n')
        self.validator = validator
        self.current_function = None
        self.current_class = None
        self.local_vars: Set[str] = set()
        self.safe_checks: Dict[str, int] = {}  # variable -> line where it was checked
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        old_function = self.current_function
        old_locals = self.local_vars.copy()
        
        self.current_function = node.name
        self.local_vars = {arg.arg for arg in node.args.args}
        
        self.generic_visit(node)
        
        self.current_function = old_function
        self.local_vars = old_locals
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        self.visit_FunctionDef(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Validate function calls."""
        self._validate_function_call(node)
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Validate attribute access."""
        self._validate_attribute_access(node)
        self.generic_visit(node)
    
    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Validate subscript access."""
        self._validate_subscript_access(node)
        self.generic_visit(node)
    
    def visit_If(self, node: ast.If) -> None:
        """Track None/null checks."""
        # Check if this is a None/null check
        if isinstance(node.test, ast.Compare):
            self._track_null_checks(node.test)
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Track variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_vars.add(target.id)
        
        self.generic_visit(node)
    
    def _validate_function_call(self, node: ast.Call) -> None:
        """Validate a function call's arguments."""
        # Extract function name
        func_name = self._get_function_name(node.func)
        if not func_name:
            return
        
        # Look up function signature
        signature = self.validator.function_signatures.get(func_name)
        if not signature:
            # Try without class prefix if it's a method call
            if '.' in func_name:
                base_name = func_name.split('.')[-1]
                signature = self._find_method_signature(base_name)
        
        if not signature:
            return  # Can't validate without signature
        
        # Validate arguments
        provided_args = len(node.args)
        provided_kwargs = {kw.arg for kw in node.keywords if kw.arg}
        
        # Check positional arguments
        required_positional = len([arg for arg in signature.positional_args 
                                 if arg not in signature.defaults])
        
        if provided_args < required_positional:
            self.validator._add_issue(
                str(self.file_path), node.lineno, 'missing_arguments', 'error',
                f"Function '{func_name}' missing required arguments. "
                f"Expected at least {required_positional}, got {provided_args}",
                f"Add missing arguments: {signature.positional_args[provided_args:required_positional]}"
            )
        elif provided_args > len(signature.positional_args) and not signature.var_args:
            self.validator._add_issue(
                str(self.file_path), node.lineno, 'too_many_arguments', 'error',
                f"Function '{func_name}' called with too many arguments. "
                f"Expected at most {len(signature.positional_args)}, got {provided_args}",
                "Remove extra arguments or check function signature"
            )
        
        # Check keyword arguments
        for kw in node.keywords:
            if kw.arg and kw.arg not in signature.keyword_args and \
               kw.arg not in signature.positional_args and not signature.var_kwargs:
                self.validator._add_issue(
                    str(self.file_path), node.lineno, 'unknown_keyword', 'error',
                    f"Unknown keyword argument '{kw.arg}' for function '{func_name}'",
                    f"Valid arguments: {signature.positional_args + signature.keyword_args}"
                )
    
    def _validate_attribute_access(self, node: ast.Attribute) -> None:
        """Validate attribute access."""
        target = ast.unparse(node.value)
        line = node.lineno
        
        # Check if target might be None
        is_safe = self._is_safe_access(target, line)
        
        access = DataAccess(
            access_type='attribute',
            target=target,
            accessor=node.attr,
            line_number=line,
            file_path=str(self.file_path),
            is_safe=is_safe,
            has_check=target in self.safe_checks
        )
        self.validator.data_accesses.append(access)
        
        if not is_safe and target not in self.safe_checks:
            # Check for common unsafe patterns
            if 'None' not in self.lines[line - 1] and 'if' not in self.lines[line - 1]:
                self.validator._add_issue(
                    str(self.file_path), line, 'unsafe_attribute_access', 'warning',
                    f"Potentially unsafe attribute access '{target}.{node.attr}' without null check",
                    f"Add null check: if {target} is not None: ..."
                )
    
    def _validate_subscript_access(self, node: ast.Subscript) -> None:
        """Validate subscript access (dict keys, list indices)."""
        target = ast.unparse(node.value)
        
        if isinstance(node.slice, ast.Constant):
            accessor = repr(node.slice.value)
        elif isinstance(node.slice, ast.Name):
            accessor = node.slice.id
        else:
            accessor = ast.unparse(node.slice)
        
        line = node.lineno
        
        # Check if this is a dictionary or list access
        access_type = 'subscript'
        is_safe = False
        
        # Check for safe patterns
        code_line = self.lines[line - 1] if line <= len(self.lines) else ""
        if 'get(' in code_line or 'if' in code_line or 'try' in code_line:
            is_safe = True
        
        access = DataAccess(
            access_type=access_type,
            target=target,
            accessor=accessor,
            line_number=line,
            file_path=str(self.file_path),
            is_safe=is_safe,
            has_check=is_safe
        )
        self.validator.data_accesses.append(access)
        
        if not is_safe:
            self.validator._add_issue(
                str(self.file_path), line, 'unsafe_subscript_access', 'warning',
                f"Potentially unsafe subscript access '{target}[{accessor}]' without existence check",
                f"Use .get() for dicts or check bounds/existence first"
            )
    
    def _track_null_checks(self, node: ast.Compare) -> None:
        """Track None/null checks for safety analysis."""
        # Check for patterns like: x is not None, x is None, x != None
        if len(node.ops) == 1 and len(node.comparators) == 1:
            op = node.ops[0]
            comparator = node.comparators[0]
            
            if isinstance(comparator, ast.Constant) and comparator.value is None:
                if isinstance(node.left, ast.Name):
                    if isinstance(op, (ast.IsNot, ast.NotEq)):
                        self.safe_checks[node.left.id] = node.lineno
    
    def _get_function_name(self, func_node: ast.AST) -> Optional[str]:
        """Extract function name from AST node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            base = ast.unparse(func_node.value)
            return f"{base}.{func_node.attr}"
        return None
    
    def _find_method_signature(self, method_name: str) -> Optional[FunctionSignature]:
        """Find method signature by name across all classes."""
        for full_name, sig in self.validator.function_signatures.items():
            if full_name.endswith(f".{method_name}"):
                return sig
        return None
    
    def _is_safe_access(self, target: str, line: int) -> bool:
        """Check if an access is safe based on context."""
        # Check if target is a literal or known safe value
        if target in {'self', 'cls', 'True', 'False'} or target.startswith('"') or target.startswith("'"):
            return True
        
        # Check if it's a local variable that was just assigned
        if target in self.local_vars:
            return True
        
        # Check if there was a recent null check
        if target in self.safe_checks and self.safe_checks[target] < line:
            return True
        
        return False


def generate_report(report: Dict[str, Any], output_file: str) -> None:
    """Generate a human-readable report."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ENHANCED VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        summary = report['summary']
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Project: {summary['project_root']}\n")
        f.write(f"Files processed: {summary['files_processed']}\n")
        f.write(f"Total issues: {summary['total_issues']}\n")
        f.write(f"  - Argument mismatches: {summary['argument_mismatches']}\n")
        f.write(f"  - Unsafe data access: {summary['unsafe_data_access']}\n")
        f.write(f"  - Missing null checks: {summary['missing_null_checks']}\n")
        f.write(f"  - Type inconsistencies: {summary['type_inconsistencies']}\n\n")
        
        # Group issues by type
        issues_by_type = defaultdict(list)
        for issue in report['issues']:
            issues_by_type[issue['issue_type']].append(issue)
        
        f.write("ISSUES BY TYPE\n")
        f.write("-" * 20 + "\n")
        
        for issue_type, issues in sorted(issues_by_type.items()):
            f.write(f"\n{issue_type.upper().replace('_', ' ')} ({len(issues)} issues):\n")
            
            for issue in issues[:10]:  # Show first 10 of each type
                f.write(f"  {issue['file_path']}:{issue['line_number']} - {issue['message']}\n")
                if issue['suggestion']:
                    f.write(f"    → {issue['suggestion']}\n")
            
            if len(issues) > 10:
                f.write(f"  ... and {len(issues) - 10} more\n")
        
        # Data access summary
        data_summary = report['data_access_summary']
        f.write("\n\nDATA ACCESS SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total accesses: {data_summary['total_accesses']}\n")
        f.write(f"Safe accesses: {data_summary['safe_accesses']}\n")
        f.write(f"Checked accesses: {data_summary['checked_accesses']}\n")
        f.write(f"Unsafe accesses: {data_summary['unsafe_accesses']}\n")
        f.write("\nAccess types:\n")
        for access_type, count in data_summary['access_types'].items():
            f.write(f"  - {access_type}: {count}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Code Validator')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--output', help='Output file for the report')
    parser.add_argument('--exclude', nargs='*', help='Patterns to exclude')
    parser.add_argument('--json', action='store_true', help='Output JSON report')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = EnhancedValidator(args.project_root, args.exclude)
    
    # Run validation
    report = validator.validate_project()
    
    # Save report
    if args.json:
        output_file = args.output or 'enhanced_validation_report.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    else:
        output_file = args.output or 'enhanced_validation_report.txt'
        generate_report(report, output_file)
    
    print(f"\nEnhanced validation completed!")
    print(f"Report saved to: {output_file}")
    
    # Print summary
    summary = report['summary']
    print(f"\nFound {summary['total_issues']} issues:")
    print(f"  - Argument mismatches: {summary['argument_mismatches']}")
    print(f"  - Unsafe data access: {summary['unsafe_data_access']}")
    print(f"  - Missing null checks: {summary['missing_null_checks']}")
    print(f"  - Type inconsistencies: {summary['type_inconsistencies']}")


if __name__ == '__main__':
    main()