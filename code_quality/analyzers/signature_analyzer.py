"""
Function Signature Analysis - Detects function signature changes and ensures calling code compatibility.
"""

import ast
import os
from collections import defaultdict
from typing import Any, NamedTuple

from ..core.config import CodeQualityConfig


class FunctionSignature(NamedTuple):
    """Represents a function signature."""
    name: str
    args: list[str]
    defaults: list[Any]
    vararg: str | None
    kwarg: str | None
    returns: str | None
    decorators: list[str]
    line_number: int
    file_path: str


class FunctionCall(NamedTuple):
    """Represents a function call."""
    function_name: str
    args: list[str]
    keywords: list[tuple[str, str]]
    line_number: int
    file_path: str


class SignatureIssue:
    """Represents a function signature-related issue."""

    def __init__(self, file_path: str, line_number: int, issue_type: str,
                 message: str, severity: str = "warning", details: dict | None = None):
        self.file_path = file_path
        self.line_number = line_number
        self.issue_type = issue_type
        self.message = message
        self.severity = severity
        self.details = details or {}


class SignatureAnalyzer:
    """Analyzes function signatures for changes and compatibility issues."""

    def __init__(self, config: CodeQualityConfig):
        self.config = config
        self.functions_by_file = defaultdict(list)
        self.function_calls_by_file = defaultdict(list)
        self.signature_changes = []
        self.compatibility_issues = []
        self.missing_functions = []
        self.unused_functions = []

    def analyze_directory(self, directory_path: str) -> dict[str, Any]:
        """Analyze function signatures in all Python files in a directory."""
        python_files = []
        for root, dirs, files in os.walk(directory_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.config.analysis.exclude_patterns]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        return self.analyze_files(python_files)

    def analyze_files(self, file_paths: list[str]) -> dict[str, Any]:
        """Analyze function signatures in specific Python files."""
        print(f"Analyzing function signatures in {len(file_paths)} files...")

        # First pass: collect all function definitions and calls
        for file_path in file_paths:
            try:
                self._analyze_file_signatures(file_path)
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")

        # Second pass: detect issues
        self._detect_signature_changes()
        self._detect_compatibility_issues()
        self._detect_missing_functions()
        self._detect_unused_functions()

        return self._generate_report()

    def _analyze_file_signatures(self, file_path: str) -> None:
        """Analyze function signatures in a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Collect function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    signature = self._extract_function_signature(node, file_path)
                    self.functions_by_file[file_path].append(signature)

                elif isinstance(node, ast.Call):
                    call = self._extract_function_call(node, file_path)
                    if call:
                        self.function_calls_by_file[file_path].append(call)

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    def _extract_function_signature(self, node: ast.FunctionDef, file_path: str) -> FunctionSignature:
        """Extract function signature from an AST node."""
        # Extract arguments
        args = []
        defaults = []

        # Positional arguments
        for arg in node.args.args:
            args.append(arg.arg)

        # Default values
        if node.args.defaults:
            # Calculate how many args have defaults
            num_defaults = len(node.args.defaults)
            num_args = len(node.args.args)

            # Add None for args without defaults
            for _i in range(num_args - num_defaults):
                defaults.append(None)

            # Add actual default values
            defaults.extend([self._get_default_value(d) for d in node.args.defaults])
        else:
            defaults = [None] * len(node.args.args)

        # Vararg (*args)
        vararg = node.args.vararg.arg if node.args.vararg else None

        # Kwarg (**kwargs)
        kwarg = node.args.kwarg.arg if node.args.kwarg else None

        # Return annotation
        returns = None
        if node.returns:
            if isinstance(node.returns, ast.Name):
                returns = node.returns.id
            elif isinstance(node.returns, ast.Constant):
                returns = str(node.returns.value)
            else:
                returns = ast.unparse(node.returns) if hasattr(ast, "unparse") else str(node.returns)

        # Decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
            else:
                decorators.append(str(decorator))

        return FunctionSignature(
            name=node.name,
            args=args,
            defaults=defaults,
            vararg=vararg,
            kwarg=kwarg,
            returns=returns,
            decorators=decorators,
            line_number=node.lineno,
            file_path=file_path,
        )

    def _get_default_value(self, node: ast.AST) -> Any:
        """Extract the default value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.List):
            return "[]"
        if isinstance(node, ast.Dict):
            return "{}"
        if isinstance(node, ast.Tuple):
            return "()"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return f"{node.func.id}()"
        return str(node)

    def _extract_function_call(self, node: ast.Call, file_path: str) -> FunctionCall | None:
        """Extract function call information from an AST node."""
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        else:
            return None

        # Extract arguments
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                args.append(arg.id)
            elif isinstance(arg, ast.Constant):
                args.append(str(arg.value))
            else:
                args.append(str(arg))

        # Extract keyword arguments
        keywords = []
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Name):
                keywords.append((keyword.arg, keyword.value.id))
            elif isinstance(keyword.value, ast.Constant):
                keywords.append((keyword.arg, str(keyword.value.value)))
            else:
                keywords.append((keyword.arg, str(keyword.value)))

        return FunctionCall(
            function_name=function_name,
            args=args,
            keywords=keywords,
            line_number=node.lineno,
            file_path=file_path,
        )

    def _detect_signature_changes(self) -> None:
        """Detect function signature changes across files."""
        # Group functions by name across all files
        functions_by_name = defaultdict(list)
        for functions in self.functions_by_file.values():
            for func in functions:
                functions_by_name[func.name].append(func)

        # Check for signature changes
        for func_name, func_list in functions_by_name.items():
            if len(func_list) > 1:
                # Compare signatures
                base_signature = func_list[0]
                for other_func in func_list[1:]:
                    if self._signatures_differ(base_signature, other_func):
                        self.signature_changes.append(SignatureIssue(
                            file_path=other_func.file_path,
                            line_number=other_func.line_number,
                            issue_type="signature_change",
                            message=f"Function '{func_name}' has different signature than in {base_signature.file_path}",
                            severity="warning",
                            details={
                                "function_name": func_name,
                                "base_signature": self._signature_to_dict(base_signature),
                                "changed_signature": self._signature_to_dict(other_func),
                                "differences": self._get_signature_differences(base_signature, other_func),
                            },
                        ))

    def _signatures_differ(self, sig1: FunctionSignature, sig2: FunctionSignature) -> bool:
        """Check if two function signatures are different."""
        return (sig1.args != sig2.args or
                sig1.defaults != sig2.defaults or
                sig1.vararg != sig2.vararg or
                sig1.kwarg != sig2.kwarg or
                sig1.returns != sig2.returns)

    def _signature_to_dict(self, sig: FunctionSignature) -> dict[str, Any]:
        """Convert function signature to dictionary for comparison."""
        return {
            "args": sig.args,
            "defaults": sig.defaults,
            "vararg": sig.vararg,
            "kwarg": sig.kwarg,
            "returns": sig.returns,
            "decorators": sig.decorators,
        }

    def _get_signature_differences(self, sig1: FunctionSignature, sig2: FunctionSignature) -> list[str]:
        """Get a list of differences between two signatures."""
        differences = []

        if sig1.args != sig2.args:
            differences.append(f"Arguments: {sig1.args} vs {sig2.args}")

        if sig1.defaults != sig2.defaults:
            differences.append(f"Defaults: {sig1.defaults} vs {sig2.defaults}")

        if sig1.vararg != sig2.vararg:
            differences.append(f"Vararg: {sig1.vararg} vs {sig2.vararg}")

        if sig1.kwarg != sig2.kwarg:
            differences.append(f"Kwarg: {sig1.kwarg} vs {sig2.kwarg}")

        if sig1.returns != sig2.returns:
            differences.append(f"Returns: {sig1.returns} vs {sig2.returns}")

        return differences

    def _detect_compatibility_issues(self) -> None:
        """Detect compatibility issues between function calls and definitions."""
        # Create a map of function definitions
        function_definitions = {}
        for functions in self.functions_by_file.values():
            for func in functions:
                function_definitions[func.name] = func

        # Check each function call against definitions
        for calls in self.function_calls_by_file.values():
            for call in calls:
                if call.function_name in function_definitions:
                    func_def = function_definitions[call.function_name]
                    issues = self._check_call_compatibility(call, func_def)

                    for issue in issues:
                        self.compatibility_issues.append(SignatureIssue(
                            file_path=call.file_path,
                            line_number=call.line_number,
                            issue_type="compatibility_issue",
                            message=issue,
                            severity="error",
                            details={
                                "function_name": call.function_name,
                                "call": {
                                    "args": call.args,
                                    "keywords": call.keywords,
                                },
                                "definition": self._signature_to_dict(func_def),
                            },
                        ))
                else:
                    # Function not defined anywhere
                    self.missing_functions.append(SignatureIssue(
                        file_path=call.file_path,
                        line_number=call.line_number,
                        issue_type="missing_function",
                        message=f"Function '{call.function_name}' is called but not defined",
                        severity="error",
                        details={
                            "function_name": call.function_name,
                            "call": {
                                "args": call.args,
                                "keywords": call.keywords,
                            },
                        },
                    ))

    def _check_call_compatibility(self, call: FunctionCall, func_def: FunctionSignature) -> list[str]:
        """Check if a function call is compatible with its definition."""
        issues = []

        # Check positional arguments
        if len(call.args) > len(func_def.args):
            if not func_def.vararg:
                issues.append(f"Too many positional arguments: {len(call.args)} provided, {len(func_def.args)} expected")

        # Check keyword arguments
        defined_args = set(func_def.args)
        for keyword_name, _ in call.keywords:
            if keyword_name not in defined_args and not func_def.kwarg:
                issues.append(f"Unknown keyword argument: '{keyword_name}'")

        # Check required arguments
        required_args = []
        for _i, (arg, default) in enumerate(zip(func_def.args, func_def.defaults, strict=False)):
            if default is None:
                required_args.append(arg)

        if len(call.args) < len(required_args):
            missing = required_args[len(call.args):]
            issues.append(f"Missing required arguments: {', '.join(missing)}")

        return issues

    def _detect_missing_functions(self) -> None:
        """Detect functions that are called but not defined."""
        # This is already handled in _detect_compatibility_issues

    def _detect_unused_functions(self) -> None:
        """Detect functions that are defined but never called."""
        # Create a set of all called function names
        called_functions = set()
        for calls in self.function_calls_by_file.values():
            for call in calls:
                called_functions.add(call.function_name)

        # Check for unused functions
        for functions in self.functions_by_file.values():
            for func in functions:
                if func.name not in called_functions:
                    # Check if it's a main function or has special decorators
                    if not self._is_special_function(func):
                        self.unused_functions.append(SignatureIssue(
                            file_path=func.file_path,
                            line_number=func.line_number,
                            issue_type="unused_function",
                            message=f"Function '{func.name}' is defined but never called",
                            severity="warning",
                            details={
                                "function_name": func.name,
                                "signature": self._signature_to_dict(func),
                            },
                        ))

    def _is_special_function(self, func: FunctionSignature) -> bool:
        """Check if a function is special (main, test, etc.)."""
        special_names = {"main", "__main__", "test_", "setup", "teardown"}
        special_decorators = {"pytest", "test", "main", "cli"}

        # Check function name
        if any(name in func.name for name in special_names):
            return True

        # Check decorators
        return bool(any(decorator in str(func.decorators) for decorator in special_decorators))

    def _generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive signature analysis report."""
        total_issues = (len(self.signature_changes) +
                       len(self.compatibility_issues) +
                       len(self.missing_functions) +
                       len(self.unused_functions))

        return {
            "summary": {
                "total_files_analyzed": len(self.functions_by_file),
                "total_functions": sum(len(funcs) for funcs in self.functions_by_file.values()),
                "total_function_calls": sum(len(calls) for calls in self.function_calls_by_file.values()),
                "total_issues": total_issues,
                "signature_changes": len(self.signature_changes),
                "compatibility_issues": len(self.compatibility_issues),
                "missing_functions": len(self.missing_functions),
                "unused_functions": len(self.unused_functions),
            },
            "issues": {
                "signature_changes": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details,
                    }
                    for issue in self.signature_changes
                ],
                "compatibility_issues": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details,
                    }
                    for issue in self.compatibility_issues
                ],
                "missing_functions": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details,
                    }
                    for issue in self.missing_functions
                ],
                "unused_functions": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details,
                    }
                    for issue in self.unused_functions
                ],
            },
            "functions": {
                file_path: [
                    {
                        "name": func.name,
                        "args": func.args,
                        "defaults": func.defaults,
                        "vararg": func.vararg,
                        "kwarg": func.kwarg,
                        "returns": func.returns,
                        "decorators": func.decorators,
                        "line": func.line_number,
                    }
                    for func in functions
                ]
                for file_path, functions in self.functions_by_file.items()
            },
            "calls": {
                file_path: [
                    {
                        "function_name": call.function_name,
                        "args": call.args,
                        "keywords": call.keywords,
                        "line": call.line_number,
                    }
                    for call in calls
                ]
                for file_path, calls in self.function_calls_by_file.items()
            },
        }

    def get_function_signatures(self) -> dict[str, list[FunctionSignature]]:
        """Get all function signatures by file."""
        return dict(self.functions_by_file)

    def get_function_calls(self) -> dict[str, list[FunctionCall]]:
        """Get all function calls by file."""
        return dict(self.function_calls_by_file)


def analyze_signatures(directory_path: str, config: CodeQualityConfig) -> dict[str, Any]:
    """Convenience function to analyze function signatures in a directory."""
    analyzer = SignatureAnalyzer(config)
    return analyzer.analyze_directory(directory_path)
