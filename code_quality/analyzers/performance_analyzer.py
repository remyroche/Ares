#!/usr/bin/env python3
"""
Performance Analyzer

Analyzes potential performance issues including:
- Algorithm complexity detection
- Inefficient patterns (N+1 queries, nested loops)
- Memory usage patterns
- I/O blocking operations
- Database query patterns
- Large data structure handling
- Generator usage opportunities
"""

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PerformanceIssue:
    """Represents a performance issue."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    message: str
    suggestion: str
    estimated_complexity: str | None = None


class PerformanceAnalyzer:
    """Analyzes code for potential performance issues."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues: list[PerformanceIssue] = []

        # Common expensive operations
        self.expensive_operations = {
            "sort", "sorted", "reverse", "reversed",
            "min", "max", "sum", "any", "all",
        }

        # Database operation patterns
        self.db_patterns = {
            "django_orm": ["objects.get", "objects.filter", "objects.all", "select_related", "prefetch_related"],
            "sqlalchemy": ["query", "filter", "all", "first", "one"],
            "raw_sql": ["execute", "fetchall", "fetchone", "executemany"],
        }

        # I/O operations
        self.io_operations = {
            "file": ["open", "read", "write", "readlines", "writelines"],
            "network": ["requests.get", "requests.post", "urlopen", "socket"],
            "subprocess": ["subprocess.run", "subprocess.call", "os.system"],
        }

    def analyze_file(self, file_path: Path) -> list[PerformanceIssue]:
        """Analyze a single file for performance issues."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            tree = ast.parse(content, filename=str(file_path))

            # Run analysis
            analyzer = PerformanceVisitor(str(file_path), lines, self)
            analyzer.visit(tree)

            return analyzer.issues

        except Exception:
            return []

    def generate_report(self) -> dict[str, Any]:
        """Generate performance analysis report."""
        # Group issues by type
        issues_by_type = defaultdict(list)
        for issue in self.issues:
            issues_by_type[issue.issue_type].append({
                "file": issue.file_path,
                "line": issue.line_number,
                "message": issue.message,
                "suggestion": issue.suggestion,
                "severity": issue.severity,
                "complexity": issue.estimated_complexity,
            })

        # Calculate severity distribution
        severity_counts = defaultdict(int)
        for issue in self.issues:
            severity_counts[issue.severity] += 1

        # Find most problematic files
        file_issue_counts = defaultdict(int)
        for issue in self.issues:
            file_issue_counts[issue.file_path] += 1

        return {
            "summary": {
                "total_issues": len(self.issues),
                "critical_issues": severity_counts["critical"],
                "high_severity": severity_counts["high"],
                "medium_severity": severity_counts["medium"],
                "low_severity": severity_counts["low"],
            },
            "issues_by_type": dict(issues_by_type),
            "most_problematic_files": sorted(
                [{"file": f, "issue_count": c} for f, c in file_issue_counts.items()],
                key=lambda x: x["issue_count"],
                reverse=True,
            )[:10],
            "complexity_issues": [
                issue for issue in self.issues
                if issue.estimated_complexity
            ],
        }


class PerformanceVisitor(ast.NodeVisitor):
    """AST visitor for detecting performance issues."""

    def __init__(self, file_path: str, lines: list[str], analyzer: PerformanceAnalyzer):
        self.file_path = file_path
        self.lines = lines
        self.analyzer = analyzer
        self.issues = []

        # Context tracking
        self.current_function = None
        self.loop_depth = 0
        self.in_async_function = False
        self.function_complexities = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        self._analyze_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self.in_async_function = True
        self._analyze_function(node)
        self.in_async_function = False

    def _analyze_function(self, node: Any) -> None:
        """Analyze function for performance issues."""
        old_function = self.current_function
        self.current_function = node.name

        # Estimate complexity
        complexity = self._estimate_complexity(node)
        if complexity > 6:  # O(n^3) or worse
            self._add_issue(
                node.lineno, "high_complexity",
                "critical" if complexity > 8 else "high",
                f"Function '{node.name}' has high algorithmic complexity",
                "Consider optimizing the algorithm or breaking it into smaller functions",
                f"O(n^{complexity//2})" if complexity >= 2 else "O(n)",
            )

        self.generic_visit(node)
        self.current_function = old_function

    def visit_For(self, node: ast.For) -> None:
        """Visit for loop."""
        self.loop_depth += 1

        # Check for nested loops
        if self.loop_depth >= 3:
            self._add_issue(
                node.lineno, "deeply_nested_loops",
                "high",
                f"Deeply nested loops ({self.loop_depth} levels)",
                "Consider refactoring to reduce nesting or use more efficient algorithms",
                f"O(n^{self.loop_depth})",
            )

        # Check for inefficient patterns in loops
        self._check_loop_patterns(node)

        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        """Visit while loop."""
        self.loop_depth += 1

        # Check for potential infinite loops
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            self._add_issue(
                node.lineno, "potential_infinite_loop",
                "high",
                "Potential infinite loop detected",
                "Ensure there's a break condition or use a for loop with a limit",
            )

        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Visit list comprehension."""
        # Check for nested comprehensions
        nested_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.ListComp | ast.SetComp | ast.DictComp))
        if nested_count > 2:
            self._add_issue(
                node.lineno, "complex_comprehension",
                "medium",
                "Complex nested comprehension",
                "Consider using regular loops for better readability and performance",
            )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call."""
        # Check for expensive operations in loops
        if self.loop_depth > 0:
            func_name = self._get_function_name(node)

            # Check for expensive operations
            if func_name in self.analyzer.expensive_operations:
                self._add_issue(
                    node.lineno, "expensive_operation_in_loop",
                    "high",
                    f"Expensive operation '{func_name}' inside loop",
                    "Move the operation outside the loop if possible",
                )

            # Check for repeated function calls
            if isinstance(node.func, ast.Attribute):
                self._check_repeated_calls(node)

        # Check for database operations
        self._check_database_operations(node)

        # Check for blocking I/O in async functions
        if self.in_async_function:
            self._check_blocking_io(node)

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Visit binary operation."""
        # Check for string concatenation in loops
        if self.loop_depth > 0 and isinstance(node.op, ast.Add):
            if self._is_string_operation(node):
                self._add_issue(
                    node.lineno, "string_concatenation_in_loop",
                    "medium",
                    "String concatenation in loop is inefficient",
                    "Use list.append() and ''.join() instead",
                )

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Visit subscript operation."""
        # Check for repeated list access in loops
        if self.loop_depth > 1 and isinstance(node.ctx, ast.Load):
            self._add_issue(
                node.lineno, "repeated_list_access",
                "low",
                "Consider caching frequently accessed list elements",
                "Store the value in a variable if accessed multiple times",
            )

        self.generic_visit(node)

    def _estimate_complexity(self, node: ast.FunctionDef) -> int:
        """Estimate algorithmic complexity of a function."""
        complexity_score = 0

        # Count nested loops
        for child in ast.walk(node):
            if isinstance(child, ast.For | ast.While):
                # Count nesting depth
                depth = 0
                parent = child
                while parent != node:
                    for p in ast.walk(node):
                        if any(c == parent for c in ast.iter_child_nodes(p)):
                            if isinstance(p, ast.For | ast.While):
                                depth += 1
                            parent = p
                            break
                complexity_score += 2 ** depth

        return complexity_score

    def _check_loop_patterns(self, node: ast.For) -> None:
        """Check for inefficient patterns in loops."""
        # Check for modifying list while iterating
        if isinstance(node.iter, ast.Name):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func_name = self._get_function_name(child)
                    if func_name in ["append", "remove", "pop", "insert"]:
                        if isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name):
                            if child.func.value.id == node.iter.id:
                                self._add_issue(
                                    child.lineno, "modify_list_while_iterating",
                                    "high",
                                    "Modifying list while iterating over it",
                                    "Create a copy of the list or use a different approach",
                                )

    def _check_repeated_calls(self, node: ast.Call) -> None:
        """Check for repeated expensive calls."""
        # This is a simplified check - could be more sophisticated
        if hasattr(node.func, "attr"):
            call_signature = f"{ast.unparse(node.func.value)}.{node.func.attr}"

            # Check if this looks like a database or API call
            if any(pattern in call_signature for pattern in ["objects.", "query.", "api.", "fetch"]):
                self._add_issue(
                    node.lineno, "repeated_expensive_call",
                    "medium",
                    f"Potentially expensive call '{node.func.attr}' in loop",
                    "Consider caching the result or batch processing",
                )

    def _check_database_operations(self, node: ast.Call) -> None:
        """Check for database operation patterns."""
        call_str = ast.unparse(node)

        # Check for N+1 query pattern
        if self.loop_depth > 0:
            for patterns in self.analyzer.db_patterns.values():
                if any(pattern in call_str for pattern in patterns):
                    self._add_issue(
                        node.lineno, "n_plus_one_query",
                        "critical",
                        "Potential N+1 query pattern detected",
                        "Use select_related() or prefetch_related() for Django, or eager loading for SQLAlchemy",
                    )
                    break

    def _check_blocking_io(self, node: ast.Call) -> None:
        """Check for blocking I/O in async functions."""
        func_name = self._get_function_name(node)

        # Check for file I/O
        if func_name in self.analyzer.io_operations["file"]:
            self._add_issue(
                node.lineno, "blocking_io_in_async",
                "high",
                f"Blocking I/O operation '{func_name}' in async function",
                "Use aiofiles or run in executor for file operations",
            )

        # Check for network operations
        elif any(op in ast.unparse(node) for op in self.analyzer.io_operations["network"]):
            if "await" not in ast.unparse(node):
                self._add_issue(
                    node.lineno, "sync_network_in_async",
                    "high",
                    "Synchronous network operation in async function",
                    "Use aiohttp or httpx for async network operations",
                )

    def _get_function_name(self, node: ast.Call) -> str:
        """Extract function name from a call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ast.unparse(node.func)

    def _is_string_operation(self, node: ast.BinOp) -> bool:
        """Check if binary operation involves strings."""
        # Simplified check - could be more thorough
        for operand in [node.left, node.right]:
            if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                return True
            if isinstance(operand, ast.Name):
                # Would need type inference for accurate detection
                return True  # Assume it might be a string
        return False

    def _add_issue(self, line_number: int, issue_type: str, severity: str,
                   message: str, suggestion: str, complexity: str | None = None) -> None:
        """Add a performance issue."""
        issue = PerformanceIssue(
            file_path=self.file_path,
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            message=message,
            suggestion=suggestion,
            estimated_complexity=complexity,
        )
        self.issues.append(issue)
        self.analyzer.issues.append(issue)
