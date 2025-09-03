#!/usr/bin/env python3
"""
Data Flow Analyzer

Analyzes data flow patterns including:
- Variable lifecycle tracking
- Uninitialized variable usage
- Unused variables and parameters
- Variable shadowing
- Data validation presence
- Null/None safety
- Input validation
- Output validation
- Boundary checking
- Type consistency in data flow
"""

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VariableInfo:
    """Information about a variable."""
    name: str
    scope: str  # 'global', 'class', 'function', 'local'
    defined_at: int  # line number
    used_at: list[int] = field(default_factory=list)
    assigned_at: list[int] = field(default_factory=list)
    type_hints: str | None = None
    is_parameter: bool = False
    is_initialized: bool = False
    possible_none: bool = False


@dataclass
class DataFlowIssue:
    """Represents a data flow issue."""
    file_path: str
    line_number: int
    variable_name: str
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: str


@dataclass
class ValidationInfo:
    """Information about data validation."""
    variable: str
    validation_type: str  # 'type', 'range', 'null', 'format'
    line_number: int
    is_input: bool
    is_output: bool


class DataFlowAnalyzer:
    """Analyzes data flow patterns in Python code."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues: list[DataFlowIssue] = []
        self.file_variables: dict[str, dict[str, VariableInfo]] = {}
        self.validations: dict[str, list[ValidationInfo]] = {}

        # Validation patterns
        self.validation_patterns = {
            "type_check": [
                r"isinstance\s*\(",
                r"type\s*\(",
                r"hasattr\s*\(",
                r"callable\s*\(",
            ],
            "null_check": [
                r"is\s+None",
                r"is\s+not\s+None",
                r"if\s+\w+:",
                r"if\s+not\s+\w+:",
            ],
            "range_check": [
                r"[<>]=?",
                r"in\s+range",
                r"between",
                r"min\s*\(",
                r"max\s*\(",
            ],
            "format_check": [
                r"\.match\s*\(",
                r"\.search\s*\(",
                r"re\.\w+\s*\(",
                r"\.startswith\s*\(",
                r"\.endswith\s*\(",
            ],
        }

        # Input functions that need validation
        self.input_sources = {
            "input", "raw_input", "open", "json.loads",
            "request.get", "request.post", "request.form",
            "request.args", "request.json", "sys.argv",
        }

    def analyze_file(self, file_path: Path) -> dict[str, Any]:
        """Analyze data flow in a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            tree = ast.parse(content, filename=str(file_path))

            # Analyze data flow
            analyzer = DataFlowVisitor(str(file_path), content, lines, self)
            analyzer.visit(tree)

            # Post-analysis checks
            self._check_data_flow_issues(str(file_path), analyzer.variables)

            return {
                "variables": len(analyzer.variables),
                "issues": len([i for i in self.issues if i.file_path == str(file_path)]),
                "validations": len(self.validations.get(str(file_path), [])),
            }

        except Exception as e:
            return {"error": str(e)}

    def _check_data_flow_issues(self, file_path: str, variables: dict[str, VariableInfo]) -> None:
        """Check for data flow issues in collected variables."""
        for var_name, var_info in variables.items():
            # Check for unused variables
            if not var_info.used_at and not var_name.startswith("_"):
                if var_info.is_parameter:
                    self._add_issue(
                        file_path, var_info.defined_at, var_name,
                        "unused_parameter", "warning",
                        f"Parameter '{var_name}' is never used",
                        "Remove the parameter or use it in the function",
                    )
                else:
                    self._add_issue(
                        file_path, var_info.defined_at, var_name,
                        "unused_variable", "warning",
                        f"Variable '{var_name}' is assigned but never used",
                        "Remove the variable or use it",
                    )

            # Check for uninitialized usage
            if var_info.used_at and not var_info.is_initialized and not var_info.is_parameter:
                for usage_line in var_info.used_at:
                    if usage_line < var_info.defined_at:
                        self._add_issue(
                            file_path, usage_line, var_name,
                            "uninitialized_usage", "error",
                            f"Variable '{var_name}' used before initialization",
                            "Initialize the variable before using it",
                        )

            # Check for potential None dereference
            if var_info.possible_none and var_info.used_at:
                # Check if there's validation before usage
                validations = self.validations.get(file_path, [])
                var_validations = [v for v in validations if v.variable == var_name]

                for usage_line in var_info.used_at:
                    has_null_check = any(
                        v.validation_type == "null" and v.line_number < usage_line
                        for v in var_validations
                    )

                    if not has_null_check:
                        self._add_issue(
                            file_path, usage_line, var_name,
                            "potential_none_access", "warning",
                            f"Variable '{var_name}' might be None",
                            "Add None check before accessing the variable",
                        )

    def _add_issue(self, file_path: str, line_number: int, variable_name: str,
                   issue_type: str, severity: str, message: str, suggestion: str) -> None:
        """Add a data flow issue."""
        self.issues.append(DataFlowIssue(
            file_path=file_path,
            line_number=line_number,
            variable_name=variable_name,
            issue_type=issue_type,
            severity=severity,
            message=message,
            suggestion=suggestion,
        ))

    def generate_report(self) -> dict[str, Any]:
        """Generate data flow analysis report."""
        # Group issues by type
        issues_by_type = defaultdict(list)
        for issue in self.issues:
            issues_by_type[issue.issue_type].append({
                "file": issue.file_path,
                "line": issue.line_number,
                "variable": issue.variable_name,
                "message": issue.message,
                "suggestion": issue.suggestion,
            })

        # Calculate validation coverage
        total_input_validations = 0
        total_output_validations = 0
        for file_validations in self.validations.values():
            total_input_validations += len([v for v in file_validations if v.is_input])
            total_output_validations += len([v for v in file_validations if v.is_output])

        # Find files with most issues
        file_issue_counts = defaultdict(int)
        for issue in self.issues:
            file_issue_counts[issue.file_path] += 1

        return {
            "summary": {
                "total_issues": len(self.issues),
                "unused_variables": len(issues_by_type["unused_variable"]),
                "unused_parameters": len(issues_by_type["unused_parameter"]),
                "uninitialized_usage": len(issues_by_type["uninitialized_usage"]),
                "potential_none_access": len(issues_by_type["potential_none_access"]),
                "variable_shadowing": len(issues_by_type["variable_shadowing"]),
                "input_validations": total_input_validations,
                "output_validations": total_output_validations,
            },
            "issues_by_type": dict(issues_by_type),
            "most_problematic_files": sorted(
                [{"file": f, "issue_count": c} for f, c in file_issue_counts.items()],
                key=lambda x: x["issue_count"],
                reverse=True,
            )[:10],
            "validation_coverage": {
                "files_with_validation": len(self.validations),
                "total_validations": sum(len(v) for v in self.validations.values()),
            },
        }


class DataFlowVisitor(ast.NodeVisitor):
    """AST visitor for data flow analysis."""

    def __init__(self, file_path: str, content: str, lines: list[str], analyzer: DataFlowAnalyzer):
        self.file_path = file_path
        self.content = content
        self.lines = lines
        self.analyzer = analyzer

        # Variable tracking
        self.variables: dict[str, VariableInfo] = {}
        self.scopes: list[str] = ["global"]
        self.current_function = None
        self.current_class = None

        # Track variable assignments and usage
        self.assignments: dict[str, list[int]] = defaultdict(list)
        self.usages: dict[str, list[int]] = defaultdict(list)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self._visit_function(node)

    def _visit_function(self, node: Any) -> None:
        """Process function definition."""
        old_function = self.current_function
        self.current_function = node.name
        self.scopes.append("function")

        # Track parameters
        for arg in node.args.args:
            if arg.arg != "self":
                var_info = VariableInfo(
                    name=arg.arg,
                    scope="function",
                    defined_at=node.lineno,
                    type_hints=ast.unparse(arg.annotation) if arg.annotation else None,
                    is_parameter=True,
                    is_initialized=True,
                )
                self.variables[f"{self.current_function}.{arg.arg}"] = var_info

        # Check for output validation
        self._check_return_validation(node)

        self.generic_visit(node)

        self.scopes.pop()
        self.current_function = old_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name
        self.scopes.append("class")

        self.generic_visit(node)

        self.scopes.pop()
        self.current_class = old_class

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statement."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                scope_prefix = self._get_scope_prefix()
                full_name = f"{scope_prefix}{var_name}" if scope_prefix else var_name

                # Check if variable already exists (shadowing)
                if var_name in self.variables:
                    self.analyzer._add_issue(
                        self.file_path, node.lineno, var_name,
                        "variable_shadowing", "warning",
                        f"Variable '{var_name}' shadows an outer scope variable",
                        "Use a different variable name",
                    )

                # Track variable
                if full_name not in self.variables:
                    self.variables[full_name] = VariableInfo(
                        name=var_name,
                        scope=self.scopes[-1],
                        defined_at=node.lineno,
                        is_initialized=True,
                    )

                # Check if assigned None
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    self.variables[full_name].possible_none = True

                # Check for input sources that need validation
                if isinstance(node.value, ast.Call):
                    self._check_input_validation(node, var_name)

                self.assignments[full_name].append(node.lineno)

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignment."""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            scope_prefix = self._get_scope_prefix()
            full_name = f"{scope_prefix}{var_name}" if scope_prefix else var_name

            if full_name not in self.variables:
                self.variables[full_name] = VariableInfo(
                    name=var_name,
                    scope=self.scopes[-1],
                    defined_at=node.lineno,
                    type_hints=ast.unparse(node.annotation) if node.annotation else None,
                    is_initialized=node.value is not None,
                )

            if node.value is not None:
                self.assignments[full_name].append(node.lineno)

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Visit name reference."""
        if isinstance(node.ctx, ast.Load):
            var_name = node.id

            # Try to find the variable in current and outer scopes
            full_name = self._find_variable(var_name)
            if full_name:
                self.usages[full_name].append(node.lineno)
                if full_name in self.variables:
                    self.variables[full_name].used_at.append(node.lineno)

        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """Visit if statement to check for validation patterns."""
        # Check if condition contains validation
        test_str = ast.unparse(node.test)

        # Extract variable being tested
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.left, ast.Name):
                var_name = node.test.left.id
                self._check_validation_pattern(node.lineno, var_name, test_str)
        elif isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not):
            if isinstance(node.test.operand, ast.Name):
                var_name = node.test.operand.id
                self._check_validation_pattern(node.lineno, var_name, test_str)
        elif isinstance(node.test, ast.Name):
            var_name = node.test.id
            self._check_validation_pattern(node.lineno, var_name, test_str)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call."""
        # Check for validation functions
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Check for type checking functions
            if func_name in ["isinstance", "type", "hasattr", "callable"]:
                if node.args and isinstance(node.args[0], ast.Name):
                    var_name = node.args[0].id
                    self._add_validation(var_name, "type", node.lineno)

        self.generic_visit(node)

    def _get_scope_prefix(self) -> str:
        """Get current scope prefix for variable names."""
        prefix_parts = []
        if self.current_class:
            prefix_parts.append(self.current_class)
        if self.current_function:
            prefix_parts.append(self.current_function)
        return ".".join(prefix_parts) + "." if prefix_parts else ""

    def _find_variable(self, var_name: str) -> str | None:
        """Find variable in current or outer scopes."""
        # Check current scope first
        scope_prefix = self._get_scope_prefix()
        full_name = f"{scope_prefix}{var_name}"

        if full_name in self.variables:
            return full_name

        # Check outer scopes
        if self.current_function:
            # Check class scope
            if self.current_class:
                class_var = f"{self.current_class}.{var_name}"
                if class_var in self.variables:
                    return class_var

        # Check global scope
        if var_name in self.variables:
            return var_name

        return None

    def _check_input_validation(self, node: ast.Assign, var_name: str) -> None:
        """Check if input needs validation."""
        if isinstance(node.value, ast.Call):
            call_str = ast.unparse(node.value.func)

            # Check if it's an input source
            for input_source in self.analyzer.input_sources:
                if input_source in call_str:
                    # Check if validation follows
                    self._add_validation(var_name, "input", node.lineno, is_input=True)

                    # Look for validation in next few lines
                    has_validation = self._has_nearby_validation(node.lineno, var_name)

                    if not has_validation:
                        self.analyzer._add_issue(
                            self.file_path, node.lineno, var_name,
                            "unvalidated_input", "error",
                            f"Input from '{input_source}' is not validated",
                            "Add input validation before using the data",
                        )

    def _check_return_validation(self, node: ast.FunctionDef) -> None:
        """Check if function validates output."""
        # Look for return statements
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value:
                # Check if return value is validated
                return_line = child.lineno

                # Simple heuristic: check if there's validation before return
                has_validation = any(
                    isinstance(n, ast.If) and n.lineno < return_line
                    for n in ast.walk(node)
                )

                if has_validation:
                    self._add_validation("return", "output", return_line, is_output=True)

    def _check_validation_pattern(self, line_number: int, var_name: str, test_str: str) -> None:
        """Check if test string contains validation pattern."""
        for val_type, patterns in self.analyzer.validation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, test_str, re.IGNORECASE):
                    self._add_validation(var_name, val_type, line_number)
                    break

    def _add_validation(self, variable: str, validation_type: str, line_number: int,
                       is_input: bool = False, is_output: bool = False) -> None:
        """Add validation info."""
        if self.file_path not in self.analyzer.validations:
            self.analyzer.validations[self.file_path] = []

        self.analyzer.validations[self.file_path].append(ValidationInfo(
            variable=variable,
            validation_type=validation_type,
            line_number=line_number,
            is_input=is_input,
            is_output=is_output,
        ))

    def _has_nearby_validation(self, line_number: int, var_name: str,
                              distance: int = 5) -> bool:
        """Check if variable has validation within a few lines."""
        validations = self.analyzer.validations.get(self.file_path, [])

        return any(
            v.variable == var_name and
            line_number < v.line_number <= line_number + distance
            for v in validations
        )
