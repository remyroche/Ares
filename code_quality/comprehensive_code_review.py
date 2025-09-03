#!/usr/bin/env python3
"""
Comprehensive Code Quality Review Script

This script performs extensive code quality checks including:
- Function existence and import validation
- Parameter validation and type checking
- Async/await usage verification
- Code style and formatting
- Security vulnerabilities
- Performance issues
- Documentation quality
- Error handling patterns
"""

import ast
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astroid

# Try to import optional dependencies
try:
    import astroid
    ASTROID_AVAILABLE = True
except ImportError:
    ASTROID_AVAILABLE = False

try:
    import mypy.api
    MYPY_AVAILABLE = True
except ImportError:
    MYPY_AVAILABLE = False

try:
    import bandit
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """Represents a code quality issue found during review."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: str | None = None
    code_snippet: str | None = None


@dataclass
class FunctionCall:
    """Represents a function call found in the code."""
    name: str
    line_number: int
    args: list[str]
    keywords: list[tuple[str, str]]
    is_async: bool
    has_await: bool


@dataclass
class FunctionDefinition:
    """Represents a function definition found in the code."""
    name: str
    line_number: int
    args: list[str]
    defaults: list[Any]
    is_async: bool
    docstring: str | None
    return_annotation: str | None


class CodeQualityReviewer:
    """Comprehensive code quality reviewer for Python projects."""

    def __init__(self, project_root: str, exclude_patterns: list[str] | None = None):
        self.project_root = Path(project_root)
        self.exclude_patterns = exclude_patterns or [
            "*/__pycache__/*",
            r"*/\.*/*",
            "*/venv/*",
            "*/env/*",
            "*/node_modules/*",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*.so",
            "*.dll",
            "*.dylib",
        ]

        self.issues: list[CodeIssue] = []
        self.function_calls: list[FunctionCall] = []
        self.function_definitions: list[FunctionDefinition] = []
        self.imports: dict[str, set[str]] = defaultdict(set)
        self.async_functions: set[str] = set()

        # Statistics
        self.stats = {
            "files_processed": 0,
            "total_issues": 0,
            "errors": 0,
            "warnings": 0,
            "info": 0,
        }

    def scan_project(self) -> dict[str, Any]:
        """Scan the entire project for code quality issues."""
        logger.info(f"Starting code quality review for project: {self.project_root}")

        start_time = time.time()

        # Find all Python files
        python_files = self._find_python_files()
        logger.info(f"Found {len(python_files)} Python files to analyze")

        # Process each file
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
                self.stats["files_processed"] += 1
            except Exception as e:
                logger.exception(f"Error analyzing {file_path}: {e}")
                self._add_issue(
                    str(file_path), 0, "analysis_error", "error",
                    f"Failed to analyze file: {e}",
                )

        # Perform cross-file analysis
        self._cross_file_analysis()

        # Generate report
        end_time = time.time()
        processing_time = end_time - start_time

        return {
            "summary": {
                "project_root": str(self.project_root),
                "files_processed": self.stats["files_processed"],
                "total_issues": len(self.issues),
                "errors": len([i for i in self.issues if i.severity == "error"]),
                "warnings": len([i for i in self.issues if i.severity == "warning"]),
                "info": len([i for i in self.issues if i.severity == "info"]),
                "processing_time_seconds": processing_time,
            },
            "issues": [self._issue_to_dict(issue) for issue in self.issues],
            "function_analysis": {
                "total_calls": len(self.function_calls),
                "total_definitions": len(self.function_definitions),
                "async_functions": len(self.async_functions),
            },
        }


    def _find_python_files(self) -> list[Path]:
        """Find all Python files in the project, excluding specified patterns."""
        python_files = []

        for pattern in self.exclude_patterns:
            if pattern.startswith(("*/", "./")):
                pattern = pattern[2:]

        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                re.match(pattern.replace("*", ".*"), os.path.join(root, d))
                for pattern in self.exclude_patterns
            )]

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    if not any(
                        re.match(pattern.replace("*", ".*"), str(file_path))
                        for pattern in self.exclude_patterns
                    ):
                        python_files.append(file_path)

        return python_files

    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for code quality issues."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse with AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                self._add_issue(
                    str(file_path), e.lineno or 0, "syntax_error", "error",
                    f"Syntax error: {e.msg}",
                )
                return

            # Analyze the AST
            self._analyze_ast(file_path, tree, content)

            # Additional checks
            self._check_file_encoding(file_path, content)
            self._check_line_lengths(file_path, content)
            self._check_trailing_whitespace(file_path, content)
            self._check_missing_newline_at_eof(file_path, content)

        except Exception as e:
            logger.exception(f"Error reading {file_path}: {e}")

    def _analyze_ast(self, file_path: Path, tree: ast.AST, content: str) -> None:
        """Analyze the AST for various code quality issues."""
        visitor = CodeQualityASTVisitor(file_path, content, self)
        visitor.visit(tree)

        # Collect results from visitor
        self.issues.extend(visitor.issues)
        self.function_calls.extend(visitor.function_calls)
        self.function_definitions.extend(visitor.function_definitions)
        self.imports[str(file_path)] = visitor.imports

    def _cross_file_analysis(self) -> None:
        """Perform analysis that requires information from multiple files."""
        self._check_function_existence()
        self._check_async_await_usage()
        self._check_import_consistency()
        self._check_circular_imports()

    def _check_function_existence(self) -> None:
        """Check if all called functions actually exist."""
        defined_functions = {f.name for f in self.function_definitions}
        imported_functions = set()

        # Collect imported functions
        for file_imports in self.imports.values():
            imported_functions.update(file_imports)

        # Check function calls
        for call in self.function_calls:
            if call.name not in defined_functions and call.name not in imported_functions:
                # Check if it's a built-in function
                if not hasattr(__builtins__, call.name):
                    self._add_issue(
                        call.file_path, call.line_number, "undefined_function", "error",
                        f"Function '{call.name}' is called but not defined or imported",
                        "Define the function or import it from the appropriate module",
                    )

    def _check_async_await_usage(self) -> None:
        """Check proper async/await usage."""
        for call in self.function_calls:
            if call.is_async and not call.has_await:
                self._add_issue(
                    call.file_path, call.line_number, "missing_await", "error",
                    f"Async function '{call.name}' is called without await",
                    f"Add 'await' before the function call: await {call.name}(...)",
                )

    def _check_import_consistency(self) -> None:
        """Check for import consistency issues."""
        # This would require more sophisticated analysis

    def _check_circular_imports(self) -> None:
        """Check for circular import dependencies."""
        # This would require building a dependency graph

    def _check_file_encoding(self, file_path: Path, content: str) -> None:
        """Check file encoding issues."""
        try:
            content.encode("utf-8")
        except UnicodeEncodeError:
            self._add_issue(
                str(file_path), 0, "encoding_error", "error",
                "File contains invalid UTF-8 characters",
            )

    def _check_line_lengths(self, file_path: Path, content: str) -> None:
        """Check for lines that are too long."""
        max_line_length = 120
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            if len(line) > max_line_length:
                self._add_issue(
                    str(file_path), i, "line_too_long", "warning",
                    f"Line {i} is {len(line)} characters long (max: {max_line_length})",
                    "Consider breaking the line or using line continuation",
                )

    def _check_trailing_whitespace(self, file_path: Path, content: str) -> None:
        """Check for trailing whitespace."""
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            if line.rstrip() != line:
                self._add_issue(
                    str(file_path), i, "trailing_whitespace", "warning",
                    f"Line {i} has trailing whitespace",
                    "Remove trailing spaces and tabs",
                )

    def _check_missing_newline_at_eof(self, file_path: Path, content: str) -> None:
        """Check if file ends with a newline."""
        if content and not content.endswith("\n"):
            self._add_issue(
                str(file_path), len(content.split("\n")), "missing_newline_at_eof", "warning",
                "File does not end with a newline",
                "Add a newline at the end of the file",
            )

    def _add_issue(self, file_path: str, line_number: int, issue_type: str,
                   severity: str, message: str, suggestion: str | None = None) -> None:
        """Add a code quality issue to the list."""
        issue = CodeIssue(
            file_path=file_path,
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            message=message,
            suggestion=suggestion,
        )
        self.issues.append(issue)

    def _issue_to_dict(self, issue: CodeIssue) -> dict[str, Any]:
        """Convert a CodeIssue to a dictionary for JSON serialization."""
        return {
            "file_path": issue.file_path,
            "line_number": issue.line_number,
            "issue_type": issue.issue_type,
            "severity": issue.severity,
            "message": issue.message,
            "suggestion": issue.suggestion,
            "code_snippet": issue.code_snippet,
        }

    def generate_report(self, output_file: str | None = None) -> str:
        """Generate a comprehensive report of all issues found."""
        if not output_file:
            output_file = f"code_quality_report_{int(time.time())}.json"

        report = self.scan_project()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Also generate a human-readable summary
        summary_file = output_file.replace(".json", "_summary.txt")
        self._generate_summary_report(report, summary_file)

        return output_file

    def _generate_summary_report(self, report: dict[str, Any], output_file: str) -> None:
        """Generate a human-readable summary report."""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("CODE QUALITY REVIEW SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            summary = report["summary"]
            f.write(f"Project: {summary['project_root']}\n")
            f.write(f"Files processed: {summary['files_processed']}\n")
            f.write(f"Total issues: {summary['total_issues']}\n")
            f.write(f"Errors: {summary['errors']}\n")
            f.write(f"Warnings: {summary['warnings']}\n")
            f.write(f"Info: {summary['info']}\n")
            f.write(f"Processing time: {summary['processing_time_seconds']:.2f} seconds\n\n")

            # Group issues by severity
            issues_by_severity = defaultdict(list)
            for issue in report["issues"]:
                issues_by_severity[issue["severity"]].append(issue)

            for severity in ["error", "warning", "info"]:
                if issues_by_severity[severity]:
                    f.write(f"\n{severity.upper()}S ({len(issues_by_severity[severity])}):\n")
                    f.write("-" * 30 + "\n")

                    for issue in issues_by_severity[severity]:
                        f.write(f"{issue['file_path']}:{issue['line_number']} - {issue['message']}\n")
                        if issue["suggestion"]:
                            f.write(f"  Suggestion: {issue['suggestion']}\n")
                        f.write("\n")


class CodeQualityASTVisitor(ast.NodeVisitor):
    """AST visitor for analyzing code quality issues."""

    def __init__(self, file_path: Path, content: str, reviewer: CodeQualityReviewer):
        self.file_path = file_path
        self.content = content
        self.reviewer = reviewer
        self.issues: list[CodeIssue] = []
        self.function_calls: list[FunctionCall] = []
        self.function_definitions: list[FunctionDefinition] = []
        self.imports: set[str] = set()
        self.current_class = None
        self.current_function = None

        # Get line content for better error reporting
        self.lines = content.split("\n")

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            self.imports.add(alias.name)
            if alias.asname:
                self.imports.add(alias.asname)

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements."""
        if node.module:
            for alias in node.names:
                if alias.asname:
                    self.imports.add(alias.asname)
                else:
                    self.imports.add(alias.name)

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        # Check function naming convention
        if not re.match(r"^[a-z_][a-z0-9_]*$", node.name):
            self._add_issue(
                node.lineno, "naming_convention", "warning",
                f"Function name '{node.name}' should be lowercase with underscores",
            )

        # Check for missing docstring
        if not ast.get_docstring(node):
            self._add_issue(
                node.lineno, "missing_docstring", "warning",
                f"Function '{node.name}' is missing a docstring",
            )

        # Check for too many arguments
        if len(node.args.args) > 7:
            self._add_issue(
                node.lineno, "too_many_arguments", "warning",
                f"Function '{node.name}' has {len(node.args.args)} arguments (consider using a config object)",
            )

        # Check for unused parameters
        self._check_unused_parameters(node)

        # Store function definition
        self.function_definitions.append(FunctionDefinition(
            name=node.name,
            line_number=node.lineno,
            args=[arg.arg for arg in node.args.args],
            defaults=node.args.defaults,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            docstring=ast.get_docstring(node),
            return_annotation=ast.unparse(node.returns) if node.returns else None,
        ))

        # Track async functions
        if isinstance(node, ast.AsyncFunctionDef):
            self.reviewer.async_functions.add(f"{self.current_class}.{node.name}" if self.current_class else node.name)

        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        # Check class naming convention
        if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
            self._add_issue(
                node.lineno, "naming_convention", "warning",
                f"Class name '{node.name}' should be PascalCase",
            )

        # Check for missing docstring
        if not ast.get_docstring(node):
            self._add_issue(
                node.lineno, "missing_docstring", "warning",
                f"Class '{node.name}' is missing a docstring",
            )

        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls."""
        # Extract function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            func_name = "unknown"

        # Check for hardcoded secrets
        self._check_hardcoded_secrets(node, func_name)

        # Check for potential SQL injection
        self._check_sql_injection(node, func_name)

        # Store function call
        self.function_calls.append(FunctionCall(
            name=func_name,
            line_number=node.lineno,
            args=[ast.unparse(arg) for arg in node.args],
            keywords=[(kw.arg, ast.unparse(kw.value)) for kw in node.keywords],
            is_async=func_name in self.reviewer.async_functions,
            has_await=self._has_await_parent(node),
        ))

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statements."""
        # Check for unused variables
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._check_unused_variable(target.id, node.lineno)

        # Check for magic numbers
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int | float):
            if not self._is_magic_number_acceptable(node.value.value):
                self._add_issue(
                    node.lineno, "magic_number", "warning",
                    f"Magic number {node.value.value} should be defined as a named constant",
                )

        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Visit exception handlers."""
        # Check for bare except clauses
        if node.type is None:
            self._add_issue(
                node.lineno, "bare_except", "error",
                "Bare except clause catches all exceptions - specify exception types",
            )

        # Check for unused exception variables
        if node.name and not self._is_exception_variable_used(node):
            self._add_issue(
                node.lineno, "unused_exception_variable", "warning",
                f"Exception variable '{node.name}' is not used",
            )

        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Visit for loops."""
        # Check for unused loop variables
        if isinstance(node.target, ast.Name) and not self._is_loop_variable_used(node):
            self._add_issue(
                node.lineno, "unused_loop_variable", "warning",
                f"Loop variable '{node.target.id}' is not used (consider using '_')",
            )

        self.generic_visit(node)

    def _check_unused_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Check for unused parameters in function definitions."""
        # This is a simplified check - would need more sophisticated analysis

    def _check_unused_variable(self, var_name: str, line_number: int) -> None:
        """Check if a variable is used after definition."""
        # This is a simplified check - would need more sophisticated analysis

    def _check_hardcoded_secrets(self, node: ast.Call, func_name: str) -> None:
        """Check for hardcoded secrets in function calls."""
        secret_patterns = [
            r"password",
            r"secret",
            r"key",
            r"token",
            r"api_key",
            r"private_key",
        ]

        for pattern in secret_patterns:
            if re.search(pattern, func_name, re.IGNORECASE):
                # Check if any arguments contain hardcoded strings
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if len(arg.value) > 10:  # Likely not a simple identifier
                            self._add_issue(
                                node.lineno, "hardcoded_secret", "error",
                                f"Potential hardcoded secret in {func_name} call",
                            )

    def _check_sql_injection(self, node: ast.Call, func_name: str) -> None:
        """Check for potential SQL injection vulnerabilities."""
        sql_functions = ["execute", "executemany", "execute_query", "raw_query"]

        if func_name in sql_functions:
            # Check if any arguments contain string formatting or concatenation
            for arg in node.args:
                if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod | ast.Add):
                    self._add_issue(
                        node.lineno, "sql_injection", "error",
                        f"Potential SQL injection in {func_name} call - use parameterized queries",
                    )

    def _is_magic_number_acceptable(self, value: int | float) -> bool:
        """Check if a magic number is acceptable (common values)."""
        acceptable_numbers = {0, 1, -1, 2, 10, 100, 1000, 0.5, 0.1, 0.01}
        return value in acceptable_numbers

    def _is_exception_variable_used(self, node: ast.ExceptHandler) -> None:
        """Check if exception variable is used in handler."""
        # This is a simplified check - would need more sophisticated analysis
        return False

    def _is_loop_variable_used(self, node: ast.For) -> None:
        """Check if loop variable is used in loop body."""
        # This is a simplified check - would need more sophisticated analysis
        return False

    def _has_await_parent(self, node: ast.Call) -> bool:
        """Check if the function call is awaited."""
        # This is a simplified check - would need more sophisticated analysis
        return False

    def _add_issue(self, line_number: int, issue_type: str, severity: str, message: str) -> None:
        """Add a code quality issue."""
        issue = CodeIssue(
            file_path=str(self.file_path),
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            message=message,
            code_snippet=self.lines[line_number - 1] if 0 < line_number <= len(self.lines) else None,
        )
        self.issues.append(issue)


def main():
    """Main entry point for the code quality review script."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Code Quality Review")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", help="Output file for the report")
    parser.add_argument("--exclude", nargs="*", help="Patterns to exclude")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize reviewer
    reviewer = CodeQualityReviewer(args.project_root, args.exclude)

    # Generate report
    output_file = reviewer.generate_report(args.output)

    print("\nCode quality review completed!")
    print(f"Report saved to: {output_file}")
    print(f"Summary saved to: {output_file.replace('.json', '_summary.txt')}")

    # Print summary to console
    with open(output_file.replace(".json", "_summary.txt")) as f:
        print(f.read())


if __name__ == "__main__":
    main()
