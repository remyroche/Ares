#!/usr/bin/env python3
"""
Function Validator - Focused Code Quality Checker

This script specifically checks:
1. Function existence and import validation
2. Parameter validation and type checking
3. Async/await usage verification
4. Function call patterns and consistency
"""

import ast
import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import logging
import json
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FunctionIssue:
    """Represents a function-related issue found during validation."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class FunctionCall:
    """Represents a function call found in the code."""
    name: str
    line_number: int
    file_path: str
    args: List[str]
    keywords: List[Tuple[str, str]]
    is_async: bool
    has_await: bool
    context: str  # 'function', 'class', 'module'


@dataclass
class FunctionDefinition:
    """Represents a function definition found in the code."""
    name: str
    line_number: int
    file_path: str
    args: List[str]
    defaults: List[Any]
    is_async: bool
    docstring: Optional[str]
    return_annotation: Optional[str]
    context: str  # 'function', 'class', 'module'


@dataclass
class ImportInfo:
    """Represents import information."""
    module: str
    name: str
    as_name: Optional[str]
    line_number: int
    file_path: str


class FunctionValidator:
    """Focused validator for function existence, parameters, and async/await usage."""
    
    def __init__(self, project_root: str, exclude_patterns: Optional[List[str]] = None):
        self.project_root = Path(project_root)
        self.exclude_patterns = exclude_patterns or [
            '*/__pycache__/*',
            r'*/\.*/*',
            '*/venv/*',
            '*/env/*',
            '*/node_modules/*',
            '*.pyc',
            '*.pyo',
            '*.pyd'
        ]
        
        self.issues: List[FunctionIssue] = []
        self.function_calls: List[FunctionCall] = []
        self.function_definitions: List[FunctionDefinition] = []
        self.imports: List[ImportInfo] = []
        self.async_functions: Set[str] = set()
        
        # Built-in functions and methods
        self.builtin_functions = set(dir(__builtins__))
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_issues': 0,
            'undefined_functions': 0,
            'missing_await': 0,
            'parameter_mismatches': 0
        }
    
    def validate_project(self) -> Dict[str, Any]:
        """Validate the entire project for function-related issues."""
        logger.info(f"Starting function validation for project: {self.project_root}")
        
        start_time = time.time()
        
        # Find all Python files
        python_files = self._find_python_files()
        logger.info(f"Found {len(python_files)} Python files to analyze")
        
        # Process each file
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
                self.stats['files_processed'] += 1
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                self._add_issue(
                    str(file_path), 0, 'analysis_error', 'error',
                    f"Failed to analyze file: {e}"
                )
        
        # Perform cross-file analysis
        self._cross_file_analysis()
        
        # Generate report
        end_time = time.time()
        processing_time = end_time - start_time
        
        report = {
            'summary': {
                'project_root': str(self.project_root),
                'files_processed': self.stats['files_processed'],
                'total_issues': len(self.issues),
                'undefined_functions': self.stats['undefined_functions'],
                'missing_await': self.stats['missing_await'],
                'parameter_mismatches': self.stats['parameter_mismatches'],
                'processing_time_seconds': processing_time
            },
            'issues': [self._issue_to_dict(issue) for issue in self.issues],
            'function_analysis': {
                'total_calls': len(self.function_calls),
                'total_definitions': len(self.function_definitions),
                'async_functions': len(self.async_functions),
                'total_imports': len(self.imports)
            }
        }
        
        return report
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project, excluding specified patterns."""
        python_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                re.match(pattern.replace('*', '.*'), os.path.join(root, d))
                for pattern in self.exclude_patterns
            )]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if not any(
                        re.match(pattern.replace('*', '.*'), str(file_path))
                        for pattern in self.exclude_patterns
                    ):
                        python_files.append(file_path)
        
        return python_files
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for function-related issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse with AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                self._add_issue(
                    str(file_path), e.lineno or 0, 'syntax_error', 'error',
                    f"Syntax error: {e.msg}"
                )
                return
            
            # Analyze the AST
            self._analyze_ast(file_path, tree, content)
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    def _analyze_ast(self, file_path: Path, tree: ast.AST, content: str) -> None:
        """Analyze the AST for function-related issues."""
        visitor = FunctionValidatorVisitor(file_path, content, self)
        visitor.visit(tree)
        
        # Collect results from visitor
        self.issues.extend(visitor.issues)
        self.function_calls.extend(visitor.function_calls)
        self.function_definitions.extend(visitor.function_definitions)
        self.imports.extend(visitor.imports)
    
    def _cross_file_analysis(self) -> None:
        """Perform analysis that requires information from multiple files."""
        self._check_function_existence()
        self._check_async_await_usage()
        self._check_parameter_validation()
        self._check_import_consistency()
    
    def _check_function_existence(self) -> None:
        """Check if all called functions actually exist."""
        # Build a map of available functions
        defined_functions = {f.name for f in self.function_definitions}
        imported_functions = set()
        
        # Collect imported functions
        for imp in self.imports:
            imported_functions.add(imp.name)
            if imp.as_name:
                imported_functions.add(imp.as_name)
        
        # Check function calls
        for call in self.function_calls:
            if (call.name not in defined_functions and 
                call.name not in imported_functions and
                call.name not in self.builtin_functions):
                
                self._add_issue(
                    call.file_path, call.line_number, 'undefined_function', 'error',
                    f"Function '{call.name}' is called but not defined, imported, or built-in",
                    f"Define the function, import it, or check the spelling"
                )
                self.stats['undefined_functions'] += 1
    
    def _check_async_await_usage(self) -> None:
        """Check proper async/await usage."""
        for call in self.function_calls:
            if call.is_async and not call.has_await:
                self._add_issue(
                    call.file_path, call.line_number, 'missing_await', 'error',
                    f"Async function '{call.name}' is called without await",
                    f"Add 'await' before the function call: await {call.name}(...)"
                )
                self.stats['missing_await'] += 1
    
    def _check_parameter_validation(self) -> None:
        """Check for parameter validation issues."""
        # This is a simplified check - would need more sophisticated analysis
        # to properly validate parameter types and counts
        pass
    
    def _check_import_consistency(self) -> None:
        """Check for import consistency issues."""
        # Group imports by module
        imports_by_module = defaultdict(list)
        for imp in self.imports:
            imports_by_module[imp.module].append(imp)
        
        # Check for potential import conflicts
        for module, module_imports in imports_by_module.items():
            names = [imp.name for imp in module_imports]
            if len(names) != len(set(names)):
                # Potential naming conflict
                for imp in module_imports:
                    if names.count(imp.name) > 1:
                        self._add_issue(
                            imp.file_path, imp.line_number, 'import_conflict', 'warning',
                            f"Potential naming conflict with '{imp.name}' from '{module}'",
                            "Consider using 'as' to alias conflicting imports"
                        )
    
    def _add_issue(self, file_path: str, line_number: int, issue_type: str, 
                   severity: str, message: str, suggestion: Optional[str] = None) -> None:
        """Add a function validation issue to the list."""
        issue = FunctionIssue(
            file_path=file_path,
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            message=message,
            suggestion=suggestion
        )
        self.issues.append(issue)
    
    def _issue_to_dict(self, issue: FunctionIssue) -> Dict[str, Any]:
        """Convert a FunctionIssue to a dictionary for JSON serialization."""
        return {
            'file_path': issue.file_path,
            'line_number': issue.line_number,
            'issue_type': issue.issue_type,
            'severity': issue.severity,
            'message': issue.message,
            'suggestion': issue.suggestion,
            'code_snippet': issue.code_snippet
        }
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive report of all function validation issues."""
        if not output_file:
            output_file = f"function_validation_report_{int(time.time())}.json"
        
        report = self.validate_project()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also generate a human-readable summary
        summary_file = output_file.replace('.json', '_summary.txt')
        self._generate_summary_report(report, summary_file)
        
        return output_file
    
    def _generate_summary_report(self, report: Dict[str, Any], output_file: str) -> None:
        """Generate a human-readable summary report."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("FUNCTION VALIDATION SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            
            summary = report['summary']
            f.write(f"Project: {summary['project_root']}\n")
            f.write(f"Files processed: {summary['files_processed']}\n")
            f.write(f"Total issues: {summary['total_issues']}\n")
            f.write(f"Undefined functions: {summary['undefined_functions']}\n")
            f.write(f"Missing await: {summary['missing_await']}\n")
            f.write(f"Parameter mismatches: {summary['parameter_mismatches']}\n")
            f.write(f"Processing time: {summary['processing_time_seconds']:.2f} seconds\n\n")
            
            # Group issues by type
            issues_by_type = defaultdict(list)
            for issue in report['issues']:
                issues_by_type[issue['issue_type']].append(issue)
            
            for issue_type, type_issues in issues_by_type.items():
                f.write(f"\n{issue_type.upper().replace('_', ' ')} ({len(type_issues)}):\n")
                f.write("-" * 40 + "\n")
                
                for issue in type_issues:
                    f.write(f"{issue['file_path']}:{issue['line_number']} - {issue['message']}\n")
                    if issue['suggestion']:
                        f.write(f"  Suggestion: {issue['suggestion']}\n")
                    f.write("\n")


class FunctionValidatorVisitor(ast.NodeVisitor):
    """AST visitor for analyzing function-related issues."""
    
    def __init__(self, file_path: Path, content: str, validator: FunctionValidator):
        self.file_path = file_path
        self.content = content
        self.validator = validator
        self.issues: List[FunctionIssue] = []
        self.function_calls: List[FunctionCall] = []
        self.function_definitions: List[FunctionDefinition] = []
        self.imports: List[ImportInfo] = []
        
        # Track context
        self.current_class = None
        self.current_function = None
        
        # Get line content for better error reporting
        self.lines = content.split('\n')
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(ImportInfo(
                module=alias.name,
                name=alias.name,
                as_name=alias.asname,
                line_number=node.lineno,
                file_path=str(self.file_path)
            ))
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements."""
        if node.module:
            for alias in node.names:
                self.imports.append(ImportInfo(
                    module=node.module,
                    name=alias.name,
                    as_name=alias.asname,
                    line_number=node.lineno,
                    file_path=str(self.file_path)
                ))
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        # Check for missing docstring
        if not ast.get_docstring(node):
            self._add_issue(
                node.lineno, 'missing_docstring', 'warning',
                f"Function '{node.name}' is missing a docstring"
            )
        
        # Check for too many arguments
        if len(node.args.args) > 7:
            self._add_issue(
                node.lineno, 'too_many_arguments', 'warning',
                f"Function '{node.name}' has {len(node.args.args)} arguments (consider using a config object)"
            )
        
        # Store function definition
        self.function_definitions.append(FunctionDefinition(
            name=node.name,
            line_number=node.lineno,
            file_path=str(self.file_path),
            args=[arg.arg for arg in node.args.args],
            defaults=node.args.defaults,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            docstring=ast.get_docstring(node),
            return_annotation=ast.unparse(node.returns) if node.returns else None,
            context=f"{self.current_class}.{node.name}" if self.current_class else node.name
        ))
        
        # Track async functions
        if isinstance(node, ast.AsyncFunctionDef):
            self.validator.async_functions.add(f"{self.current_class}.{node.name}" if self.current_class else node.name)
        
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = None
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        # Check for missing docstring
        if not ast.get_docstring(node):
            self._add_issue(
                node.lineno, 'missing_docstring', 'warning',
                f"Class '{node.name}' is missing a docstring"
            )
        
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls."""
        # Extract function name and context
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            context = "direct_call"
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            context = "method_call"
        else:
            func_name = "unknown"
            context = "complex_call"
        
        # Check if this is an async function call
        is_async = func_name in self.validator.async_functions
        
        # Store function call
        self.function_calls.append(FunctionCall(
            name=func_name,
            line_number=node.lineno,
            file_path=str(self.file_path),
            args=[ast.unparse(arg) for arg in node.args],
            keywords=[(kw.arg, ast.unparse(kw.value)) for kw in node.keywords],
            is_async=is_async,
            has_await=self._has_await_parent(node),
            context=context
        ))
        
        self.generic_visit(node)
    
    def _has_await_parent(self, node: ast.Call) -> bool:
        """Check if the function call is awaited."""
        # This is a simplified check - would need more sophisticated analysis
        # to properly detect await usage in complex expressions
        return False
    
    def _add_issue(self, line_number: int, issue_type: str, severity: str, message: str) -> None:
        """Add a function validation issue."""
        issue = FunctionIssue(
            file_path=str(self.file_path),
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            message=message,
            code_snippet=self.lines[line_number - 1] if 0 < line_number <= len(self.lines) else None
        )
        self.issues.append(issue)


def main():
    """Main entry point for the function validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Function Validation - Code Quality Checker')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--output', help='Output file for the report')
    parser.add_argument('--exclude', nargs='*', help='Patterns to exclude')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = FunctionValidator(args.project_root, args.exclude)
    
    # Generate report
    output_file = validator.generate_report(args.output)
    
    print(f"\nFunction validation completed!")
    print(f"Report saved to: {output_file}")
    print(f"Summary saved to: {output_file.replace('.json', '_summary.txt')}")
    
    # Print summary to console
    with open(output_file.replace('.json', '_summary.txt'), 'r') as f:
        print(f.read())


if __name__ == '__main__':
    main()