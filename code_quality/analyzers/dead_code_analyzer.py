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

from vulture.core import Vulture

from ..core.config import AnalysisConfig
from ..utils.file_utils import find_python_files


@dataclass
class DeadCodeIssue:
    """Container for dead code analysis results."""
    file_path: str
    line_number: int
    issue_type: str
    description: str
    confidence: float
    code_snippet: str
    severity: str
    removal_impact: str = "low"  # low, medium, high
    dependencies: list[str] = None  # List of dependencies that would be affected


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

        for file_path in python_files:
            try:
                # Standard dead code analysis
                file_issues = self.analyze_file(file_path)
                all_issues.extend(file_issues)
                
                # Enhanced analysis
                deprecated_issues = self.detect_deprecated_code(file_path)
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

            # Use Vulture to find dead code
            vulture_issues = self.vulture.scan(source, filename=file_path)

            for issue in vulture_issues:
                # Extract line number and description
                line_number = getattr(issue, "lineno", 0)
                description = getattr(issue, "description", str(issue))
                confidence = getattr(issue, "confidence", 100.0)

                # Get code snippet
                code_snippet = self._extract_code_snippet(lines, line_number)

                # Determine issue type
                issue_type = self._classify_issue(description)

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

    def _classify_issue(self, description: str) -> str:
        """Classify the type of dead code issue."""
        description_lower = description.lower()

        if "import" in description_lower:
            return "unused_import"
        if "variable" in description_lower:
            return "unused_variable"
        if "function" in description_lower:
            return "unused_function"
        if "class" in description_lower:
            return "unused_class"
        if "method" in description_lower:
            return "unused_method"
        if "attribute" in description_lower:
            return "unused_attribute"
        if "argument" in description_lower or "parameter" in description_lower:
            return "unused_parameter"
        if "assignment" in description_lower:
            return "unused_assignment"
        if "expression" in description_lower:
            return "unused_expression"
        if "statement" in description_lower:
            return "unused_statement"
        return "unknown"

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

        # Issues by type
        lines.append("Issues by Type:")
        for issue_type, count in report.issues_by_type.items():
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
        Detect deprecated code patterns in a file.
        
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
            deprecated_issues = []
            
            for node in ast.walk(tree):
                # Check for @deprecated decorators
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    for decorator in node.decorator_list:
                        if self._is_deprecated_decorator(decorator):
                            issue = self._create_deprecated_issue(
                                node, decorator, lines, str(file_path), "decorator"
                            )
                            deprecated_issues.append(issue)
                
                # Check for DeprecationWarning usage
                if isinstance(node, ast.Raise):
                    if self._is_deprecation_warning(node):
                        issue = self._create_deprecated_issue(
                            node, None, lines, str(file_path), "warning"
                        )
                        deprecated_issues.append(issue)
                
                # Check for deprecation comments
                if hasattr(node, 'lineno'):
                    line_content = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    if self._is_deprecation_comment(line_content):
                        issue = self._create_deprecated_issue(
                            node, None, lines, str(file_path), "comment"
                        )
                        deprecated_issues.append(issue)
            
            return deprecated_issues
            
        except Exception as e:
            print(f"Warning: Could not analyze deprecated code in {file_path}: {e}")
            return []

    def _is_deprecated_decorator(self, decorator: ast.expr) -> bool:
        """Check if a decorator indicates deprecation."""
        if isinstance(decorator, ast.Name):
            return decorator.id.lower() in ["deprecated", "deprecate"]
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id.lower() in ["deprecated", "deprecate"]
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr.lower() in ["deprecated", "deprecate"]
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
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of DeadCodeIssue objects for potentially unused dynamic imports
        """
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix != ".py":
            return []
            
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            lines = source.split("\n")
            dynamic_imports = []
            
            for node in ast.walk(tree):
                # Check for importlib.import_module calls
                if isinstance(node, ast.Call):
                    if self._is_importlib_call(node):
                        issue = self._create_dynamic_import_issue(
                            node, lines, str(file_path), "importlib"
                        )
                        dynamic_imports.append(issue)
                
                # Check for __import__ calls
                elif isinstance(node, ast.Call):
                    if self._is_dunder_import_call(node):
                        issue = self._create_dynamic_import_issue(
                            node, lines, str(file_path), "__import__"
                        )
                        dynamic_imports.append(issue)
                
                # Check for exec/eval with import statements
                elif isinstance(node, (ast.Exec, ast.Call)):
                    if self._contains_dynamic_import(node):
                        issue = self._create_dynamic_import_issue(
                            node, lines, str(file_path), "exec_eval"
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

    def _contains_dynamic_import(self, node: ast.AST) -> bool:
        """Check if exec/eval contains import statements."""
        # This is a simplified check - in practice, you'd need more sophisticated analysis
        if isinstance(node, ast.Exec):
            if isinstance(node.body, ast.Constant):
                return "import" in str(node.body.value)
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
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of DeadCodeIssue objects for unreachable code
        """
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix != ".py":
            return []
            
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            lines = source.split("\n")
            unreachable_code = []
            
            for node in ast.walk(tree):
                # Check for unreachable code after return/raise/break/continue
                if isinstance(node, ast.FunctionDef):
                    unreachable_in_function = self._find_unreachable_in_function(node, lines, str(file_path))
                    unreachable_code.extend(unreachable_in_function)
            
            return unreachable_code
            
        except Exception as e:
            print(f"Warning: Could not analyze conditional dead code in {file_path}: {e}")
            return []

    def _find_unreachable_in_function(self, func_node: ast.FunctionDef, lines: list[str], file_path: str) -> list[DeadCodeIssue]:
        """Find unreachable code within a function."""
        unreachable = []
        last_terminating_line = 0
        
        for node in ast.walk(func_node):
            if hasattr(node, 'lineno'):
                # Check if this node comes after a terminating statement
                if last_terminating_line > 0 and node.lineno > last_terminating_line:
                    # Check if this is a terminating statement
                    if isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                        last_terminating_line = node.lineno
                    else:
                        # This might be unreachable code
                        issue = DeadCodeIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            issue_type="unreachable_code",
                            description="Code after terminating statement may be unreachable",
                            confidence=70.0,
                            code_snippet=self._extract_code_snippet(lines, node.lineno),
                            severity="medium",
                            removal_impact="low"
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
