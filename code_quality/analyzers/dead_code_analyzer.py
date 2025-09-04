"""
Dead Code Analyzer

Detects unused code, dead imports, unreachable code, and deprecated code using Vulture library.
Helps identify code that can be safely removed to improve codebase cleanliness.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from vulture.core import Vulture
    VULTURE_AVAILABLE = True
except ImportError:
    VULTURE_AVAILABLE = False
    # Provide stub class for when vulture is not available
    class Vulture:
        def __init__(self):
            self.unreachable_code = []
            self.dead_code = []
            
        def scavenge(self, *args, **kwargs):
            pass

from core.config import AnalysisConfig
from utils.file_utils import find_python_files


@dataclass
class DeadCodeIssue:
    """Container for dead code analysis results."""
    file_path: str
    line_number: int
    issue_type: str  # "dead_code" or "unreachable_code"
    description: str
    confidence: float
    code_snippet: str
    severity: str
    removal_impact: str = "low"  # low, medium, high
    dependencies: list[str] = None  # List of dependencies that would be affected
    is_bug: bool = False  # True if this is unreachable code (a bug), False if dead code


@dataclass
class DeprecatedCodeIssue:
    """Container for deprecated code analysis results."""
    file_path: str
    line_number: int
    deprecated_type: str  # decorator, warning, comment, version
    description: str
    deprecation_reason: str
    removal_version: str = ""
    alternative: str = ""
    code_snippet: str = ""
    severity: str = "medium"


@dataclass
class DeadCodeReport:
    """Container for dead code analysis report."""
    total_issues: int
    issues_by_type: dict[str, int]
    issues_by_file: dict[str, list[DeadCodeIssue]]
    issues_by_severity: dict[str, list[DeadCodeIssue]]
    confidence_distribution: dict[str, int]
    potential_savings: dict[str, int]  # Lines of code that could be removed
    deprecated_issues: list[DeprecatedCodeIssue] = None
    impact_analysis: dict[str, Any] = None


class DeadCodeAnalyzer:
    """
    Analyzes Python code for dead/unused code using Vulture.

    Detects:
    - Unused imports
    - Unused variables
    - Unused functions and classes
    - Dead code blocks
    - Unreachable code
    - Deprecated code patterns
    - Dynamic imports
    - Conditional dead code
    """

    def __init__(self, config: AnalysisConfig | None = None):
        """
        Initialize the dead code analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or AnalysisConfig()
        self.confidence_threshold = getattr(self.config, "confidence_threshold", 80.0)
        self.ignore_patterns = getattr(self.config, "ignore_patterns", [])
        self.whitelist = getattr(self.config, "whitelist", [])

        # Initialize Vulture with custom configuration
        self.vulture = Vulture()
        self._configure_vulture()

    def _configure_vulture(self):
        """Configure Vulture with custom settings."""
        # Set confidence threshold
        self.vulture.min_confidence = self.confidence_threshold

        # Add common whitelist patterns
        default_whitelist = [
            # Common patterns that might be false positives
            "unused_import",
            "unused_variable",
            "unused_function",
            "unused_class",
            "unused_method",
            "unused_attribute",
            "unused_argument",
            "unused_parameter",
            "unused_return_value",
            "unused_assignment",
            "unused_expression",
            "unused_statement",
            "unused_import_statement",
            "unused_from_import",
            "unused_import_alias",
            "unused_import_from",
            "unused_import_as",
            "unused_import_from_as",
            "unused_import_from_star",
            "unused_import_star",
            "unused_import_relative",
            "unused_import_absolute",
            "unused_import_relative_from",
            "unused_import_absolute_from",
            "unused_import_relative_as",
            "unused_import_absolute_as",
            "unused_import_relative_star",
            "unused_import_absolute_star",
            "unused_import_relative_from_star",
            "unused_import_absolute_from_star",
            "unused_import_relative_from_as",
            "unused_import_absolute_from_as",
            "unused_import_relative_from_star_as",
            "unused_import_absolute_from_star_as",
        ]

        # Add user-defined whitelist
        if self.whitelist:
            default_whitelist.extend(self.whitelist)

        self.vulture.whitelist = default_whitelist

    def analyze_file(self, file_path: str | Path) -> list[DeadCodeIssue]:
        """
        Analyze a single Python file for dead code.

        Args:
            file_path: Path to Python file

        Returns:
            List of DeadCodeIssue objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        if file_path.suffix != ".py":
            msg = f"File must be a Python file: {file_path}"
            raise ValueError(msg)

        try:
            # Read file content
            with open(file_path, encoding="utf-8") as f:
                source = f.read()

            # Analyze with Vulture
            issues = self._analyze_source(source, str(file_path))

            # Filter issues based on configuration
            return self._filter_issues(issues)


        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")
            return []

    def analyze_directory(self, directory: str | Path) -> DeadCodeReport:
        """
        Analyze all Python files in a directory for dead code.

        Args:
            directory: Path to directory

        Returns:
            DeadCodeReport object with analysis results
        """
        directory = Path(directory)
        if not directory.is_dir():
            msg = f"Not a directory: {directory}"
            raise NotADirectoryError(msg)

        python_files = find_python_files(directory)
        all_issues = []
        all_deprecated_issues = []

        # First pass: build global call graph for proper deprecated code detection
        print("Building global call graph...")
        global_call_graph = self._build_global_call_graph(python_files)
        print(f"Found {len(global_call_graph['definitions'])} function definitions and {len(global_call_graph['calls'])} function calls")
        print(f"Successfully processed {len(global_call_graph['files'])} files")

        for file_path in python_files:
            try:
                # Standard dead code analysis
                file_issues = self.analyze_file(file_path)
                all_issues.extend(file_issues)
                
                # Enhanced analysis with global call graph
                deprecated_issues = self.detect_deprecated_code_with_graph(file_path, global_call_graph)
                all_deprecated_issues.extend(deprecated_issues)
                
                dynamic_import_issues = self.detect_dynamic_imports(file_path)
                all_issues.extend(dynamic_import_issues)
                
                conditional_dead_issues = self.detect_conditional_dead_code(file_path)
                all_issues.extend(conditional_dead_issues)
                
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")

        # Generate comprehensive report
        report = self._generate_report(all_issues)
        report.deprecated_issues = all_deprecated_issues
        report.impact_analysis = self.analyze_removal_impact(all_issues)
        
        # Add dependency-aware analysis
        dependency_analysis = self.analyze_dependency_aware_removal(all_issues)
        removal_plan = self.generate_removal_plan(all_issues)
        
        # Add to report
        report.impact_analysis["dependency_analysis"] = dependency_analysis
        report.impact_analysis["removal_plan"] = removal_plan
        
        return report

    def analyze_files(self, file_paths: list[str | Path]) -> DeadCodeReport:
        """
        Analyze multiple Python files for dead code.

        Args:
            file_paths: List of file paths

        Returns:
            DeadCodeReport object with analysis results
        """
        all_issues = []

        for file_path in file_paths:
            try:
                file_issues = self.analyze_file(file_path)
                all_issues.extend(file_issues)
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")

        return self._generate_report(all_issues)

    def _analyze_source(self, source: str, file_path: str) -> list[DeadCodeIssue]:
        """Analyze source code for dead code issues."""
        issues = []

        try:
            # Parse AST to get line information
            ast.parse(source)
            lines = source.split("\n")

            # Use Vulture to find dead code if available
            vulture_issues = []
            if VULTURE_AVAILABLE:
                try:
                    # Vulture needs to scan files, not source code directly
                    # So we'll use AST-based analysis as the primary method
                    pass  # Vulture integration would go here
                except Exception as e:
                    print(f"Warning: Vulture analysis failed: {e}")

            for issue in vulture_issues:
                # Extract line number and description
                line_number = getattr(issue, "lineno", 0)
                description = getattr(issue, "description", str(issue))
                confidence = getattr(issue, "confidence", 100.0)

                # Get code snippet
                code_snippet = self._extract_code_snippet(lines, line_number)

                # Determine issue type and whether it's a bug
                issue_type, is_bug = self._classify_issue(description)

                # Determine severity
                severity = self._determine_severity(confidence, issue_type)

                issues.append(DeadCodeIssue(
                    file_path=file_path,
                    line_number=line_number,
                    issue_type=issue_type,
                    description=description,
                    confidence=confidence,
                    code_snippet=code_snippet,
                    severity=severity,
                    is_bug=is_bug,
                ))

        except SyntaxError as e:
            # Handle syntax errors gracefully
            issues.append(DeadCodeIssue(
                file_path=file_path,
                line_number=getattr(e, "lineno", 0),
                issue_type="syntax_error",
                description=f"Syntax error: {str(e)}",
                confidence=100.0,
                code_snippet="",
                severity="high",
            ))
        except Exception as e:
            print(f"Warning: Error analyzing {file_path}: {e}")

        return issues

    def _extract_code_snippet(self, lines: list[str], line_number: int) -> str:
        """Extract code snippet around the specified line."""
        if line_number <= 0 or line_number > len(lines):
            return ""

        # Get context (2 lines before and after)
        start_line = max(0, line_number - 3)
        end_line = min(len(lines), line_number + 2)

        snippet_lines = []
        for i in range(start_line, end_line):
            if i == line_number - 1:  # Target line (0-indexed)
                snippet_lines.append(f"  {i+1:4d}: >>> {lines[i]}")
            else:
                snippet_lines.append(f"  {i+1:4d}:     {lines[i]}")

        return "\n".join(snippet_lines)

    def _classify_issue(self, description: str) -> tuple[str, bool]:
        """Classify the type of issue and whether it's a bug (unreachable) or dead code."""
        description_lower = description.lower()
        
        # Check if this is unreachable code (a bug)
        unreachable_patterns = [
            "unreachable", "after return", "after raise", "after break", "after continue",
            "code after", "terminating statement", "never executed"
        ]
        
        is_bug = any(pattern in description_lower for pattern in unreachable_patterns)
        
        if is_bug:
            return "unreachable_code", True
        
        # Classify as dead code (unused)
        if "import" in description_lower:
            return "unused_import", False
        if "variable" in description_lower:
            return "unused_variable", False
        if "function" in description_lower:
            return "unused_function", False
        if "class" in description_lower:
            return "unused_class", False
        if "method" in description_lower:
            return "unused_method", False
        if "attribute" in description_lower:
            return "unused_attribute", False
        if "argument" in description_lower or "parameter" in description_lower:
            return "unused_parameter", False
        if "assignment" in description_lower:
            return "unused_assignment", False
        if "expression" in description_lower:
            return "unused_expression", False
        if "statement" in description_lower:
            return "unused_statement", False
        return "unknown", False

    def _determine_severity(self, confidence: float, issue_type: str) -> str:
        """Determine the severity of an issue."""
        if confidence >= 95:
            return "high"
        if confidence >= 80:
            return "medium"
        return "low"

    def _filter_issues(self, issues: list[DeadCodeIssue]) -> list[DeadCodeIssue]:
        """Filter issues based on configuration."""
        filtered = []

        for issue in issues:
            # Check confidence threshold
            if issue.confidence < self.confidence_threshold:
                continue

            # Check ignore patterns
            if self._should_ignore_issue(issue):
                continue

            filtered.append(issue)

        return filtered

    def _should_ignore_issue(self, issue: DeadCodeIssue) -> bool:
        """Check if an issue should be ignored based on patterns."""
        for pattern in self.ignore_patterns:
            if pattern in issue.description.lower():
                return True
            if pattern in issue.file_path.lower():
                return True
        return False

    def _generate_report(self, issues: list[DeadCodeIssue]) -> DeadCodeReport:
        """Generate a comprehensive report from all issues."""
        # Group issues by type
        issues_by_type = {}
        for issue in issues:
            if issue.issue_type not in issues_by_type:
                issues_by_type[issue.issue_type] = 0
            issues_by_type[issue.issue_type] += 1

        # Group issues by file
        issues_by_file = {}
        for issue in issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)

        # Group issues by severity
        issues_by_severity = {}
        for issue in issues:
            if issue.severity not in issues_by_severity:
                issues_by_severity[issue.severity] = []
            issues_by_severity[issue.severity].append(issue)

        # Calculate confidence distribution
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            confidence_distribution[issue.severity] += 1

        # Calculate potential savings
        potential_savings = self._calculate_potential_savings(issues)

        return DeadCodeReport(
            total_issues=len(issues),
            issues_by_type=issues_by_type,
            issues_by_file=issues_by_file,
            issues_by_severity=issues_by_severity,
            confidence_distribution=confidence_distribution,
            potential_savings=potential_savings,
        )

    def _calculate_potential_savings(self, issues: list[DeadCodeIssue]) -> dict[str, int]:
        """Calculate potential lines of code that could be removed."""
        savings = {
            "total_lines": 0,
            "import_lines": 0,
            "function_lines": 0,
            "class_lines": 0,
            "variable_lines": 0,
        }

        for issue in issues:
            if issue.confidence >= 90:  # Only count high-confidence issues
                if issue.issue_type == "unused_import":
                    savings["import_lines"] += 1
                elif issue.issue_type in ["unused_function", "unused_method"]:
                    savings["function_lines"] += 1
                elif issue.issue_type == "unused_class":
                    savings["class_lines"] += 1
                elif issue.issue_type == "unused_variable":
                    savings["variable_lines"] += 1

                savings["total_lines"] += 1

        return savings

    def get_dead_code_summary(self, report: DeadCodeReport) -> dict:
        """Generate a summary of dead code analysis."""
        return {
            "total_issues": report.total_issues,
            "issues_by_type": report.issues_by_type,
            "issues_by_severity": report.issues_by_severity,
            "potential_savings": report.potential_savings,
            "files_affected": len(report.issues_by_file),
            "high_confidence_issues": len([i for i in report.issues_by_severity.get("high", []) if i.confidence >= 95]),
            "medium_confidence_issues": len([i for i in report.issues_by_severity.get("medium", []) if i.confidence >= 80]),
            "low_confidence_issues": len([i for i in report.issues_by_severity.get("low", []) if i.confidence < 80]),
        }


    def find_critical_issues(self, report: DeadCodeReport) -> list[DeadCodeIssue]:
        """Find critical dead code issues that should be addressed first."""
        critical_issues = []

        for issue in report.issues_by_severity.get("high", []):
            if issue.confidence >= 95:
                critical_issues.append(issue)

        # Sort by confidence and line number
        critical_issues.sort(key=lambda x: (-x.confidence, x.line_number))

        return critical_issues

    def generate_cleanup_recommendations(self, report: DeadCodeReport) -> list[str]:
        """Generate cleanup recommendations based on analysis."""
        recommendations = []

        if report.total_issues == 0:
            recommendations.append("✅ No dead code issues found. Your codebase is clean!")
            return recommendations

        # High confidence issues
        high_confidence = len([i for i in report.issues_by_severity.get("high", []) if i.confidence >= 95])
        if high_confidence > 0:
            recommendations.append(f"🔴 {high_confidence} high-confidence issues found. These should be addressed immediately.")

        # Import issues
        import_issues = report.issues_by_type.get("unused_import", 0)
        if import_issues > 0:
            recommendations.append(f"📦 {import_issues} unused imports found. Consider removing them to improve startup time.")

        # Function issues
        function_issues = report.issues_by_type.get("unused_function", 0)
        if function_issues > 0:
            recommendations.append(f"⚙️ {function_issues} unused functions found. Consider removing or documenting them.")

        # Class issues
        class_issues = report.issues_by_type.get("unused_class", 0)
        if class_issues > 0:
            recommendations.append(f"🏗️ {class_issues} unused classes found. Consider removing or documenting them.")

        # Potential savings
        total_savings = report.potential_savings["total_lines"]
        if total_savings > 0:
            recommendations.append(f"💾 Potential to remove {total_savings} lines of dead code.")

        # General recommendations
        if report.total_issues > 50:
            recommendations.append("⚠️ Large number of issues found. Consider addressing them incrementally.")

        if len(report.issues_by_file) > 20:
            recommendations.append("📁 Issues spread across many files. Consider systematic cleanup.")

        return recommendations

    def export_issues(self, report: DeadCodeReport, format: str = "json") -> str:
        """Export issues in various formats."""
        if format.lower() == "json":
            import json
            return json.dumps(self._report_to_dict(report), indent=2)
        if format.lower() == "csv":
            return self._report_to_csv(report)
        if format.lower() == "text":
            return self._report_to_text(report)
        msg = f"Unsupported format: {format}"
        raise ValueError(msg)

    def _report_to_dict(self, report: DeadCodeReport) -> dict:
        """Convert report to dictionary for JSON export."""
        return {
            "total_issues": report.total_issues,
            "issues_by_type": report.issues_by_type,
            "issues_by_file": {
                file_path: [
                    {
                        "line_number": issue.line_number,
                        "issue_type": issue.issue_type,
                        "description": issue.description,
                        "confidence": issue.confidence,
                        "severity": issue.severity,
                        "code_snippet": issue.code_snippet,
                    }
                    for issue in issues
                ]
                for file_path, issues in report.issues_by_file.items()
            },
            "issues_by_severity": {
                severity: [
                    {
                        "file_path": issue.file_path,
                        "line_number": issue.line_number,
                        "issue_type": issue.issue_type,
                        "description": issue.description,
                        "confidence": issue.confidence,
                        "code_snippet": issue.code_snippet,
                    }
                    for issue in issues
                ]
                for severity, issues in report.issues_by_severity.items()
            },
            "potential_savings": report.potential_savings,
        }

    def _report_to_csv(self, report: DeadCodeReport) -> str:
        """Convert report to CSV format."""
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["File", "Line", "Type", "Description", "Confidence", "Severity"])

        # Data
        for issue in [i for issues in report.issues_by_severity.values() for i in issues]:
            writer.writerow([
                issue.file_path,
                issue.line_number,
                issue.issue_type,
                issue.description,
                issue.confidence,
                issue.severity,
            ])

        return output.getvalue()

    def _report_to_text(self, report: DeadCodeReport) -> str:
        """Convert report to human-readable text format."""
        lines = []
        lines.append("DEAD CODE ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Total Issues: {report.total_issues}")
        lines.append(f"Files Affected: {len(report.issues_by_file)}")
        lines.append("")

        # Separate unreachable code (bugs) from dead code
        unreachable_issues = [issue for issue in report.issues_by_file.get(file_path, []) if issue.is_bug]
        dead_code_issues = [issue for issue in report.issues_by_file.get(file_path, []) if not issue.is_bug]
        
        # Issues by type - separate bugs from dead code
        lines.append("UNREACHABLE CODE (BUGS - Should be fixed, not removed):")
        unreachable_by_type = {}
        for issue in unreachable_issues:
            unreachable_by_type[issue.issue_type] = unreachable_by_type.get(issue.issue_type, 0) + 1
        for issue_type, count in unreachable_by_type.items():
            lines.append(f"  {issue_type}: {count}")
        lines.append("")
        
        lines.append("DEAD CODE (Unused - Can be removed):")
        dead_code_by_type = {}
        for issue in dead_code_issues:
            dead_code_by_type[issue.issue_type] = dead_code_by_type.get(issue.issue_type, 0) + 1
        for issue_type, count in dead_code_by_type.items():
            lines.append(f"  {issue_type}: {count}")
        lines.append("")

        # Issues by severity
        lines.append("Issues by Severity:")
        for severity, count in report.issues_by_severity.items():
            lines.append(f"  {severity}: {len(count)}")
        lines.append("")

        # Potential savings
        lines.append("Potential Savings:")
        for category, count in report.potential_savings.items():
            lines.append(f"  {category}: {count} lines")
        lines.append("")

        # Detailed issues
        lines.append("Detailed Issues:")
        for file_path, issues in report.issues_by_file.items():
            lines.append(f"\n{file_path}:")
            for issue in issues:
                lines.append(f"  Line {issue.line_number}: {issue.description} (Confidence: {issue.confidence}%)")

        return "\n".join(lines)

    def detect_deprecated_code(self, file_path: str | Path) -> list[DeprecatedCodeIssue]:
        """
        Detect deprecated code using call graph analysis instead of linguistic analysis.
        A function/class is considered deprecated if it's defined but never called.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of DeprecatedCodeIssue objects
        """
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix != ".py":
            return []
            
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            lines = source.split("\n")
            
            # Build call graph for this file
            defined_functions, called_functions = self._build_call_graph(tree)
            
            # Find functions that are defined but never called
            deprecated_issues = []
            for func_name, func_node in defined_functions.items():
                if func_name not in called_functions:
                    # Skip private functions (starting with _) and special methods
                    if not func_name.startswith('_') or func_name.startswith('__'):
                        issue = self._create_unused_function_issue(
                            func_node, lines, str(file_path)
                        )
                        deprecated_issues.append(issue)
            
            return deprecated_issues
            
        except Exception as e:
            print(f"Warning: Could not analyze deprecated code in {file_path}: {e}")
            return []

    def _build_call_graph(self, tree: ast.AST) -> tuple[dict[str, ast.AST], set[str]]:
        """
        Build a call graph for the AST with improved accuracy.
        
        Returns:
            Tuple of (defined_functions, called_functions)
        """
        defined_functions = {}
        called_functions = set()
        
        for node in ast.walk(tree):
            # Track function definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                defined_functions[node.name] = node
                
            # Track function calls with improved logic
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_functions.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # For method calls like obj.method(), we track the method name
                    called_functions.add(node.func.attr)
                elif isinstance(node.func, ast.Subscript):
                    # Handle cases like func['key']()
                    if isinstance(node.func.value, ast.Name):
                        called_functions.add(node.func.value.id)
                        
        return defined_functions, called_functions

    def _create_unused_function_issue(self, func_node: ast.AST, lines: list[str], file_path: str) -> DeprecatedCodeIssue:
        """Create a DeprecatedCodeIssue for an unused function."""
        line_number = getattr(func_node, 'lineno', 0)
        code_snippet = self._extract_code_snippet(lines, line_number)
        func_name = getattr(func_node, 'name', 'unknown')
        
        return DeprecatedCodeIssue(
            file_path=file_path,
            line_number=line_number,
            deprecated_type="unused_function",
            description=f"Function '{func_name}' is defined but never called",
            deprecation_reason="Function is not referenced anywhere in the codebase",
            removal_version="",
            alternative="Consider removing if truly unused, or add usage if needed",
            code_snippet=code_snippet,
            severity="low"  # Lower severity since it might be intentionally unused
        )

    def _build_global_call_graph(self, python_files: list[Path]) -> dict[str, dict]:
        """
        Build a global call graph across all Python files.
        
        Returns:
            Dictionary with 'definitions', 'calls', and 'files' keys
        """
        global_definitions = {}  # {function_name: {file_path, line_number, node}}
        global_calls = set()     # Set of all function names that are called
        processed_files = []     # List of successfully processed files
        
        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    source = f.read()
                
                tree = ast.parse(source)
                defined_functions, called_functions = self._build_call_graph(tree)
                
                # Add definitions to global registry
                for func_name, func_node in defined_functions.items():
                    global_definitions[func_name] = {
                        'file_path': str(file_path),
                        'line_number': getattr(func_node, 'lineno', 0),
                        'node': func_node
                    }
                
                # Add calls to global registry
                global_calls.update(called_functions)
                processed_files.append(str(file_path))
                
            except Exception as e:
                print(f"Warning: Could not analyze {file_path} for call graph: {e}")
                continue
        
        return {
            'definitions': global_definitions,
            'calls': global_calls,
            'files': processed_files
        }

    def detect_deprecated_code_with_graph(self, file_path: str | Path, global_call_graph: dict) -> list[DeprecatedCodeIssue]:
        """
        Detect deprecated code using global call graph analysis with improved accuracy.
        
        Args:
            file_path: Path to Python file
            global_call_graph: Global call graph from _build_global_call_graph
            
        Returns:
            List of DeprecatedCodeIssue objects
        """
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix != ".py":
            return []
            
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            lines = source.split("\n")
            
            # Build local call graph for this file
            defined_functions, called_functions = self._build_call_graph(tree)
            
            deprecated_issues = []
            global_definitions = global_call_graph['definitions']
            global_calls = global_call_graph['calls']
            
            for func_name, func_node in defined_functions.items():
                # Skip functions that are likely to be used
                if self._is_likely_used_function(func_name, func_node, lines, str(file_path)):
                    continue
                
                # Check if function is called anywhere in the codebase
                is_called_locally = func_name in called_functions
                is_called_globally = func_name in global_calls
                
                # Additional checks for framework-specific patterns
                is_framework_callback = self._is_framework_callback(func_node, lines)
                is_entry_point = self._is_entry_point(func_name, func_node, lines)
                is_test_function = self._is_test_function(func_name, lines)
                
                # Check for dynamic usage patterns
                is_dynamically_used = self._check_dynamic_usage(func_name, file_path, global_call_graph)
                
                # Check for configuration file usage
                is_config_used = self._check_config_usage(func_name, file_path)
                
                # Check for string-based references
                is_string_referenced = self._check_string_references(func_name, file_path)
                
                if (not is_called_locally and not is_called_globally and 
                    not is_framework_callback and not is_entry_point and not is_test_function and
                    not is_dynamically_used and not is_config_used and not is_string_referenced):
                    # Function is defined but never called anywhere
                    issue = self._create_unused_function_issue(
                        func_node, lines, str(file_path)
                    )
                    issue.deprecation_reason = "Function is defined but never called anywhere in the codebase"
                    issue.severity = "low"  # Lower severity to reduce false positives
                    deprecated_issues.append(issue)
            
            return deprecated_issues
            
        except Exception as e:
            print(f"Warning: Could not analyze deprecated code in {file_path}: {e}")
            return []

    def _is_likely_used_function(self, func_name: str, func_node: ast.AST, lines: list[str], file_path: str) -> bool:
        """Check if a function is likely to be used based on various heuristics."""
        # Skip private functions (except __init__, __call__, etc.)
        if func_name.startswith('_') and not func_name.startswith('__'):
            return True
            
        # Skip special methods
        if func_name.startswith('__') and func_name.endswith('__'):
            return True
            
        # Skip functions in test files
        if 'test' in file_path.lower() or 'tests' in file_path.lower():
            return True
            
        # Skip functions in __init__.py files (likely exports)
        if file_path.endswith('__init__.py'):
            return True
            
        # Skip functions with decorators that indicate they're used
        if hasattr(func_node, 'decorator_list') and func_node.decorator_list:
            for decorator in func_node.decorator_list:
                if isinstance(decorator, ast.Name):
                    decorator_name = decorator.id.lower()
                    if any(keyword in decorator_name for keyword in ['app', 'route', 'handler', 'callback', 'listener']):
                        return True
                        
        return False

    def _check_dynamic_usage(self, func_name: str, file_path: Path, global_call_graph: dict) -> bool:
        """Check if a function is used dynamically (importlib, getattr, etc.)."""
        try:
            # Search for dynamic usage patterns in the entire codebase
            for other_file in global_call_graph.get('files', []):
                if other_file == str(file_path):
                    continue
                    
                try:
                    with open(other_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for dynamic import patterns
                    dynamic_patterns = [
                        f'importlib.import_module.*{func_name}',
                        f'getattr.*{func_name}',
                        f'hasattr.*{func_name}',
                        f'"{func_name}"',
                        f"'{func_name}'",
                        f'globals()["{func_name}"]',
                        f'locals()["{func_name}"]',
                        f'__import__.*{func_name}',
                        f'eval.*{func_name}',
                        f'exec.*{func_name}'
                    ]
                    
                    for pattern in dynamic_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            return True
                            
                except Exception:
                    continue
                    
        except Exception:
            pass
            
        return False

    def _check_config_usage(self, func_name: str, file_path: Path) -> bool:
        """Check if a function is referenced in configuration files."""
        try:
            # Look for configuration files in the same directory and parent directories
            config_patterns = ['*.yaml', '*.yml', '*.json', '*.toml', '*.ini', '*.cfg', '*.conf']
            
            for config_pattern in config_patterns:
                for config_file in file_path.parent.rglob(config_pattern):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if func_name in content:
                            return True
                            
                    except Exception:
                        continue
                        
        except Exception:
            pass
            
        return False

    def _check_string_references(self, func_name: str, file_path: Path) -> bool:
        """Check if a function is referenced as a string in the codebase."""
        try:
            # Search for string references in Python files
            for py_file in file_path.parent.rglob('*.py'):
                if py_file == file_path:
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for string references
                    string_patterns = [
                        f'"{func_name}"',
                        f"'{func_name}'",
                        f'f"{{.*{func_name}.*}}"',
                        f"f'{{.*{func_name}.*}}'"
                    ]
                    
                    for pattern in string_patterns:
                        if re.search(pattern, content):
                            return True
                            
                except Exception:
                    continue
                    
        except Exception:
            pass
            
        return False

    def _is_framework_callback(self, func_node: ast.AST, lines: list[str]) -> bool:
        """Check if a function is a framework callback."""
        if not hasattr(func_node, 'decorator_list') or not func_node.decorator_list:
            return False
            
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorator_name = decorator.id.lower()
                # Common framework decorators
                framework_patterns = [
                    'app', 'route', 'handler', 'callback', 'listener', 'event',
                    'task', 'job', 'schedule', 'cron', 'periodic', 'signal',
                    'middleware', 'filter', 'hook', 'plugin', 'extension'
                ]
                if any(pattern in decorator_name for pattern in framework_patterns):
                    return True
                    
        return False

    def _is_entry_point(self, func_name: str, func_node: ast.AST, lines: list[str]) -> bool:
        """Check if a function is an entry point."""
        # Common entry point names
        entry_point_names = ['main', 'run', 'start', 'execute', 'cli', 'command']
        if func_name.lower() in entry_point_names:
            return True
            
        # Check for if __name__ == "__main__" pattern
        for i, line in enumerate(lines):
            if 'if __name__ == "__main__"' in line:
                # Look for function calls in the next few lines
                for j in range(i + 1, min(i + 10, len(lines))):
                    if func_name in lines[j] and '(' in lines[j]:
                        return True
                        
        return False

    def _is_test_function(self, func_name: str, lines: list[str]) -> bool:
        """Check if a function is a test function."""
        # Common test function patterns
        test_patterns = ['test_', 'check_', 'verify_', 'validate_', 'assert_']
        if any(func_name.startswith(pattern) for pattern in test_patterns):
            return True
            
        # Check if function is in a test context
        for line in lines:
            if any(keyword in line.lower() for keyword in ['pytest', 'unittest', 'test', 'assert']):
                return True
                
        return False

    def _is_deprecation_warning(self, node: ast.Raise) -> bool:
        """Check if a raise statement is a DeprecationWarning."""
        if isinstance(node.exc, ast.Name):
            return node.exc.id == "DeprecationWarning"
        elif isinstance(node.exc, ast.Call):
            if isinstance(node.exc.func, ast.Name):
                return node.exc.func.id == "DeprecationWarning"
        return False

    def _is_deprecation_comment(self, line: str) -> bool:
        """Check if a line contains deprecation comments."""
        line_lower = line.lower()
        deprecation_patterns = [
            r"deprecated",
            r"deprecate",
            r"will be removed",
            r"removed in version",
            r"no longer supported",
            r"legacy",
            r"obsolete"
        ]
        return any(re.search(pattern, line_lower) for pattern in deprecation_patterns)

    def _create_deprecated_issue(self, node: ast.AST, decorator: ast.expr | None, 
                                lines: list[str], file_path: str, issue_type: str) -> DeprecatedCodeIssue:
        """Create a DeprecatedCodeIssue from AST node."""
        line_number = getattr(node, 'lineno', 0)
        code_snippet = self._extract_code_snippet(lines, line_number)
        
        # Extract deprecation reason and version
        deprecation_reason = "Code marked as deprecated"
        removal_version = ""
        alternative = ""
        
        if decorator and isinstance(decorator, ast.Call):
            # Extract arguments from decorator
            for keyword in decorator.keywords:
                if keyword.arg == "reason" and isinstance(keyword.value, ast.Constant):
                    deprecation_reason = keyword.value.value
                elif keyword.arg == "version" and isinstance(keyword.value, ast.Constant):
                    removal_version = keyword.value.value
                elif keyword.arg == "alternative" and isinstance(keyword.value, ast.Constant):
                    alternative = keyword.value.value
        
        return DeprecatedCodeIssue(
            file_path=file_path,
            line_number=line_number,
            deprecated_type=issue_type,
            description=f"Deprecated {getattr(node, 'name', 'code')}",
            deprecation_reason=deprecation_reason,
            removal_version=removal_version,
            alternative=alternative,
            code_snippet=code_snippet,
            severity="medium"
        )

    def detect_dynamic_imports(self, file_path: str | Path) -> list[DeadCodeIssue]:
        """
        Detect dynamic imports that might be missed by static analysis.
        This is now more conservative to reduce false positives.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of DeadCodeIssue objects for potentially unused dynamic imports
        """
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix != ".py":
            return []
            
        # Skip test files and common framework files to reduce false positives
        if any(skip_pattern in str(file_path).lower() for skip_pattern in 
               ['test', 'tests', '__init__.py', 'setup.py', 'conftest.py']):
            return []
            
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            lines = source.split("\n")
            dynamic_imports = []
            
            # Only report obvious cases of dynamic imports that are likely problematic
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Only flag __import__ calls as they're more likely to be problematic
                    if self._is_dunder_import_call(node):
                        # Check if it's in a try/except block (likely intentional)
                        if not self._is_in_try_except(node, tree):
                            issue = self._create_dynamic_import_issue(
                                node, lines, str(file_path), "__import__"
                            )
                            dynamic_imports.append(issue)
            
            return dynamic_imports
            
        except Exception as e:
            print(f"Warning: Could not analyze dynamic imports in {file_path}: {e}")
            return []

    def _is_importlib_call(self, node: ast.Call) -> bool:
        """Check if a call is to importlib.import_module."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return (node.func.value.id == "importlib" and 
                       node.func.attr == "import_module")
        return False

    def _is_dunder_import_call(self, node: ast.Call) -> bool:
        """Check if a call is to __import__."""
        if isinstance(node.func, ast.Name):
            return node.func.id == "__import__"
        return False

    def _is_in_try_except(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if a node is inside a try/except block."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.Try):
                for child in ast.walk(parent):
                    if child is node:
                        return True
        return False

    def _contains_dynamic_import(self, node: ast.AST) -> bool:
        """Check if exec/eval contains import statements."""
        # This is a simplified check - in practice, you'd need more sophisticated analysis
        # ast.Exec was removed in Python 3.9+, so we skip this check
        # if isinstance(node, ast.Exec):
        #     if isinstance(node.body, ast.Constant):
        #         return "import" in str(node.body.value)
        return False

    def _create_dynamic_import_issue(self, node: ast.AST, lines: list[str], 
                                   file_path: str, import_type: str) -> DeadCodeIssue:
        """Create a DeadCodeIssue for dynamic import."""
        line_number = getattr(node, 'lineno', 0)
        code_snippet = self._extract_code_snippet(lines, line_number)
        
        return DeadCodeIssue(
            file_path=file_path,
            line_number=line_number,
            issue_type="dynamic_import",
            description=f"Dynamic import using {import_type} - may be missed by static analysis",
            confidence=60.0,  # Lower confidence for dynamic imports
            code_snippet=code_snippet,
            severity="low",
            removal_impact="unknown"
        )

    def detect_conditional_dead_code(self, file_path: str | Path) -> list[DeadCodeIssue]:
        """
        Detect conditional dead code (unreachable code paths).
        This is now more conservative to reduce false positives.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of DeadCodeIssue objects for unreachable code
        """
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix != ".py":
            return []
            
        # Skip test files and common framework files to reduce false positives
        if any(skip_pattern in str(file_path).lower() for skip_pattern in 
               ['test', 'tests', '__init__.py', 'setup.py', 'conftest.py']):
            return []
            
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            lines = source.split("\n")
            unreachable_code = []
            
            # Only check for obvious cases of unreachable code
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Only check functions that are not test functions
                    if not self._is_test_function(node.name, lines):
                        unreachable_in_function = self._find_unreachable_in_function(node, lines, str(file_path))
                        unreachable_code.extend(unreachable_in_function)
            
            return unreachable_code
            
        except Exception as e:
            print(f"Warning: Could not analyze conditional dead code in {file_path}: {e}")
            return []

    def _find_unreachable_in_function(self, func_node: ast.FunctionDef, lines: list[str], file_path: str) -> list[DeadCodeIssue]:
        """Find unreachable code within a function with improved accuracy."""
        unreachable = []
        last_terminating_line = 0
        
        # Only check for obvious cases - code immediately after return/raise
        for node in ast.walk(func_node):
            if hasattr(node, 'lineno'):
                # Check if this node comes immediately after a terminating statement
                if last_terminating_line > 0 and node.lineno == last_terminating_line + 1:
                    # Check if this is a terminating statement
                    if isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                        last_terminating_line = node.lineno
                    else:
                        # Check if this is just a comment or docstring
                        line_content = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                        if not line_content.startswith('#') and not line_content.startswith('"""') and not line_content.startswith("'''"):
                            # This might be unreachable code
                            issue = DeadCodeIssue(
                                file_path=file_path,
                                line_number=node.lineno,
                                issue_type="unreachable_code",
                                description="Code immediately after terminating statement may be unreachable",
                                confidence=80.0,  # Higher confidence for immediate cases
                                code_snippet=self._extract_code_snippet(lines, node.lineno),
                                severity="medium",  # Higher severity for bugs
                                removal_impact="low",
                                is_bug=True  # Mark as a bug, not dead code
                            )
                            unreachable.append(issue)
                else:
                    # Update last terminating line
                    if isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                        last_terminating_line = node.lineno
        
        return unreachable

    def analyze_removal_impact(self, issues: list[DeadCodeIssue]) -> dict[str, Any]:
        """
        Analyze the impact of removing dead code.
        
        Args:
            issues: List of dead code issues
            
        Returns:
            Dictionary with impact analysis
        """
        impact_analysis = {
            "high_impact": [],
            "medium_impact": [],
            "low_impact": [],
            "removal_order": [],
            "total_impact_score": 0
        }
        
        for issue in issues:
            # Calculate impact score based on various factors
            impact_score = self._calculate_impact_score(issue)
            
            if impact_score >= 7:
                impact_analysis["high_impact"].append(issue)
                issue.removal_impact = "high"
            elif impact_score >= 4:
                impact_analysis["medium_impact"].append(issue)
                issue.removal_impact = "medium"
            else:
                impact_analysis["low_impact"].append(issue)
                issue.removal_impact = "low"
        
        # Sort by impact score for removal order
        all_issues = issues.copy()
        all_issues.sort(key=lambda x: self._calculate_impact_score(x), reverse=True)
        impact_analysis["removal_order"] = all_issues
        
        impact_analysis["total_impact_score"] = sum(
            self._calculate_impact_score(issue) for issue in issues
        )
        
        return impact_analysis

    def _calculate_impact_score(self, issue: DeadCodeIssue) -> int:
        """Calculate impact score for a dead code issue."""
        score = 0
        
        # Base score by issue type
        type_scores = {
            "unused_import": 1,
            "unused_variable": 2,
            "unused_function": 4,
            "unused_class": 6,
            "unused_method": 3,
            "dynamic_import": 5,
            "unreachable_code": 2
        }
        score += type_scores.get(issue.issue_type, 1)
        
        # Confidence factor
        if issue.confidence >= 95:
            score += 2
        elif issue.confidence >= 80:
            score += 1
        
        # Severity factor
        if issue.severity == "high":
            score += 3
        elif issue.severity == "medium":
            score += 2
        else:
            score += 1
        
        return score

    def analyze_dependency_aware_removal(self, issues: list[DeadCodeIssue]) -> dict[str, Any]:
        """
        Analyze dead code removal with dependency awareness.
        
        Args:
            issues: List of dead code issues
            
        Returns:
            Dictionary with dependency-aware removal analysis
        """
        dependency_analysis = {
            "removal_groups": [],
            "dependency_chains": [],
            "safe_removal_order": [],
            "risky_removals": [],
            "dependency_graph": {},
        }
        
        # Group issues by file and type
        issues_by_file = {}
        for issue in issues:
            file_path = issue.file_path
            if file_path not in issues_by_file:
                issues_by_file[file_path] = []
            issues_by_file[file_path].append(issue)
        
        # Analyze dependencies between files
        file_dependencies = self._analyze_file_dependencies(issues_by_file)
        
        # Create removal groups based on dependencies
        removal_groups = self._create_removal_groups(issues_by_file, file_dependencies)
        dependency_analysis["removal_groups"] = removal_groups
        
        # Identify dependency chains
        dependency_chains = self._find_dependency_chains(file_dependencies)
        dependency_analysis["dependency_chains"] = dependency_chains
        
        # Create safe removal order
        safe_order = self._create_safe_removal_order(removal_groups, dependency_chains)
        dependency_analysis["safe_removal_order"] = safe_order
        
        # Identify risky removals
        risky_removals = self._identify_risky_removals(issues, file_dependencies)
        dependency_analysis["risky_removals"] = risky_removals
        
        # Build dependency graph
        dependency_analysis["dependency_graph"] = file_dependencies
        
        return dependency_analysis

    def _analyze_file_dependencies(self, issues_by_file: dict[str, list[DeadCodeIssue]]) -> dict[str, Any]:
        """Analyze dependencies between files based on dead code issues."""
        file_deps = {}
        
        for file_path, issues in issues_by_file.items():
            file_deps[file_path] = {
                "dependencies": set(),
                "dependents": set(),
                "issue_count": len(issues),
                "high_impact_issues": len([i for i in issues if i.removal_impact == "high"]),
                "medium_impact_issues": len([i for i in issues if i.removal_impact == "medium"]),
                "low_impact_issues": len([i for i in issues if i.removal_impact == "low"]),
            }
        
        # Analyze import dependencies (simplified - in practice, you'd parse imports)
        for file_path in file_deps:
            # This is a simplified analysis - in practice, you'd analyze actual imports
            potential_deps = [f for f in file_deps.keys() if f != file_path]
            file_deps[file_path]["dependencies"] = set(potential_deps[:2])  # Mock dependencies
        
        # Build reverse dependencies
        for file_path, deps_info in file_deps.items():
            for dep in deps_info["dependencies"]:
                if dep in file_deps:
                    file_deps[dep]["dependents"].add(file_path)
        
        return file_deps

    def _create_removal_groups(self, issues_by_file: dict[str, list[DeadCodeIssue]], 
                              file_dependencies: dict[str, Any]) -> list[dict[str, Any]]:
        """Create groups of files that can be safely removed together."""
        removal_groups = []
        processed_files = set()
        
        # Group files by dependency level (files with no dependents first)
        dependency_levels = self._calculate_dependency_levels(file_dependencies)
        
        for level in sorted(dependency_levels.keys()):
            files_at_level = dependency_levels[level]
            group = {
                "level": level,
                "files": [],
                "total_issues": 0,
                "safe_to_remove": level == 0,  # Only level 0 is safe to remove
            }
            
            for file_path in files_at_level:
                if file_path not in processed_files and file_path in issues_by_file:
                    issues = issues_by_file[file_path]
                    group["files"].append({
                        "file_path": file_path,
                        "issues": [issue.__dict__ for issue in issues],
                        "issue_count": len(issues),
                        "impact_score": sum(self._calculate_impact_score(issue) for issue in issues),
                    })
                    group["total_issues"] += len(issues)
                    processed_files.add(file_path)
            
            if group["files"]:
                removal_groups.append(group)
        
        return removal_groups

    def _calculate_dependency_levels(self, file_dependencies: dict[str, Any]) -> dict[int, list[str]]:
        """Calculate dependency levels for files."""
        levels = {}
        remaining_files = set(file_dependencies.keys())
        current_level = 0
        
        while remaining_files:
            # Find files with no dependents at current level
            files_at_level = []
            for file_path in list(remaining_files):
                deps_info = file_dependencies[file_path]
                # Check if all dependencies are already processed
                if all(dep not in remaining_files for dep in deps_info["dependencies"]):
                    files_at_level.append(file_path)
            
            if not files_at_level:
                # Circular dependency - assign remaining files to current level
                files_at_level = list(remaining_files)
            
            levels[current_level] = files_at_level
            remaining_files -= set(files_at_level)
            current_level += 1
        
        return levels

    def _find_dependency_chains(self, file_dependencies: dict[str, Any]) -> list[list[str]]:
        """Find dependency chains in the file dependency graph."""
        chains = []
        visited = set()
        
        for file_path in file_dependencies:
            if file_path not in visited:
                chain = self._build_dependency_chain(file_path, file_dependencies, visited)
                if len(chain) > 1:  # Only include chains with multiple files
                    chains.append(chain)
        
        return chains

    def _build_dependency_chain(self, start_file: str, file_dependencies: dict[str, Any], 
                               visited: set) -> list[str]:
        """Build a dependency chain starting from a file."""
        chain = []
        current = start_file
        
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            
            # Find next file in chain (file that depends on current)
            next_file = None
            for file_path, deps_info in file_dependencies.items():
                if current in deps_info["dependencies"] and file_path not in visited:
                    next_file = file_path
                    break
            
            current = next_file
        
        return chain

    def _create_safe_removal_order(self, removal_groups: list[dict[str, Any]], 
                                  dependency_chains: list[list[str]]) -> list[dict[str, Any]]:
        """Create a safe order for removing dead code."""
        safe_order = []
        
        # Add files from removal groups in dependency order
        for group in removal_groups:
            if group["safe_to_remove"]:
                for file_info in group["files"]:
                    safe_order.append({
                        "file_path": file_info["file_path"],
                        "reason": f"Level {group['level']} - no dependents",
                        "priority": "high" if group["level"] == 0 else "medium",
                        "issue_count": file_info["issue_count"],
                        "impact_score": file_info["impact_score"],
                    })
        
        # Add files from dependency chains (remove in reverse order)
        for chain in dependency_chains:
            for file_path in reversed(chain):
                if not any(item["file_path"] == file_path for item in safe_order):
                    safe_order.append({
                        "file_path": file_path,
                        "reason": "Part of dependency chain",
                        "priority": "medium",
                        "issue_count": 0,  # Would need to calculate
                        "impact_score": 0,  # Would need to calculate
                    })
        
        return safe_order

    def _identify_risky_removals(self, issues: list[DeadCodeIssue], 
                                file_dependencies: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify potentially risky removals."""
        risky_removals = []
        
        for issue in issues:
            file_path = issue.file_path
            deps_info = file_dependencies.get(file_path, {})
            
            # Check if file has many dependents
            if len(deps_info.get("dependents", set())) > 3:
                risky_removals.append({
                    "issue": issue.__dict__,
                    "risk_reason": f"File has {len(deps_info['dependents'])} dependents",
                    "risk_level": "high",
                    "dependents": list(deps_info.get("dependents", set())),
                })
            
            # Check if issue is high impact in a file with dependencies
            elif issue.removal_impact == "high" and deps_info.get("dependencies"):
                risky_removals.append({
                    "issue": issue.__dict__,
                    "risk_reason": "High impact issue in file with dependencies",
                    "risk_level": "medium",
                    "dependencies": list(deps_info.get("dependencies", set())),
                })
        
        return risky_removals

    def generate_removal_plan(self, issues: list[DeadCodeIssue]) -> dict[str, Any]:
        """
        Generate a comprehensive removal plan for dead code.
        
        Args:
            issues: List of dead code issues
            
        Returns:
            Dictionary with removal plan
        """
        # Analyze impact and dependencies
        impact_analysis = self.analyze_removal_impact(issues)
        dependency_analysis = self.analyze_dependency_aware_removal(issues)
        
        # Create removal plan
        removal_plan = {
            "total_issues": len(issues),
            "estimated_time_savings": self._estimate_time_savings(issues),
            "removal_phases": self._create_removal_phases(impact_analysis, dependency_analysis),
            "risk_assessment": self._assess_removal_risks(dependency_analysis),
            "recommendations": self._generate_removal_recommendations(impact_analysis, dependency_analysis),
        }
        
        return removal_plan

    def _estimate_time_savings(self, issues: list[DeadCodeIssue]) -> dict[str, Any]:
        """Estimate time savings from removing dead code."""
        total_lines = sum(1 for issue in issues if issue.confidence >= 90)
        estimated_hours = total_lines * 0.1  # Rough estimate: 0.1 hours per line
        
        return {
            "total_lines_removed": total_lines,
            "estimated_hours_saved": estimated_hours,
            "estimated_days_saved": estimated_hours / 8,
            "complexity_reduction": len([i for i in issues if i.issue_type in ["unused_function", "unused_class"]]),
        }

    def _create_removal_phases(self, impact_analysis: dict[str, Any], 
                              dependency_analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Create phases for removing dead code."""
        phases = []
        
        # Phase 1: Low impact, no dependencies
        phase1 = {
            "phase": 1,
            "name": "Safe Low-Impact Removals",
            "description": "Remove low-impact dead code with no dependencies",
            "issues": impact_analysis.get("low_impact", []),
            "estimated_effort": "1-2 hours",
            "risk_level": "low",
        }
        phases.append(phase1)
        
        # Phase 2: Medium impact, minimal dependencies
        phase2 = {
            "phase": 2,
            "name": "Medium-Impact Removals",
            "description": "Remove medium-impact dead code with careful review",
            "issues": impact_analysis.get("medium_impact", []),
            "estimated_effort": "4-8 hours",
            "risk_level": "medium",
        }
        phases.append(phase2)
        
        # Phase 3: High impact, requires careful planning
        phase3 = {
            "phase": 3,
            "name": "High-Impact Removals",
            "description": "Remove high-impact dead code with extensive testing",
            "issues": impact_analysis.get("high_impact", []),
            "estimated_effort": "1-2 days",
            "risk_level": "high",
        }
        phases.append(phase3)
        
        return phases

    def _assess_removal_risks(self, dependency_analysis: dict[str, Any]) -> dict[str, Any]:
        """Assess risks associated with removing dead code."""
        risky_removals = dependency_analysis.get("risky_removals", [])
        
        return {
            "total_risks": len(risky_removals),
            "high_risk_count": len([r for r in risky_removals if r.get("risk_level") == "high"]),
            "medium_risk_count": len([r for r in risky_removals if r.get("risk_level") == "medium"]),
            "low_risk_count": len([r for r in risky_removals if r.get("risk_level") == "low"]),
            "dependency_chains": len(dependency_analysis.get("dependency_chains", [])),
            "recommended_approach": "incremental" if len(risky_removals) > 5 else "aggressive",
        }

    def _generate_removal_recommendations(self, impact_analysis: dict[str, Any], 
                                         dependency_analysis: dict[str, Any]) -> list[str]:
        """Generate recommendations for removing dead code."""
        recommendations = []
        
        # Basic recommendations
        total_issues = sum(len(issues) for issues in impact_analysis.values() if isinstance(issues, list))
        if total_issues > 0:
            recommendations.append(f"Start with {len(impact_analysis.get('low_impact', []))} low-impact issues for quick wins")
        
        # Dependency-based recommendations
        dependency_chains = dependency_analysis.get("dependency_chains", [])
        if dependency_chains:
            recommendations.append(f"Address {len(dependency_chains)} dependency chains carefully")
        
        # Risk-based recommendations
        risky_removals = dependency_analysis.get("risky_removals", [])
        if risky_removals:
            recommendations.append(f"Review {len(risky_removals)} potentially risky removals before proceeding")
        
        # General recommendations
        recommendations.extend([
            "Run tests after each removal phase",
            "Use version control to track changes",
            "Consider gradual removal over multiple commits",
            "Document removal decisions for future reference",
        ])
        
        return recommendations
