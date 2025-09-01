#!/usr/bin/env python3
"""
Enhanced Python code quality analyzer
Integrates multiple tools and provides comprehensive analysis.
"""

import ast
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import argparse


class CodeQualityAnalyzer:
    """Enhanced code quality analyzer that integrates multiple tools."""

    def __init__(self, exclusions_file: Optional[str] = None):
        self.exclusions = self._load_exclusions(exclusions_file)
        self.issues = defaultdict(lambda: defaultdict(list))
        self.stats = {
            "files_analyzed": 0,
            "syntax_errors": 0,
            "syntax_issues": 0,
            "linting_issues": 0,
            "type_errors": 0,
            "complexity_issues": 0,
            "unused_imports": 0,
            "dead_code": 0,
            "formatting_issues": 0,
            "placeholder_issues": 0,
        }

    def _load_exclusions(self, exclusions_file: Optional[str]) -> Set[str]:
        """Load exclusion patterns from file."""
        exclusions = set()
        if exclusions_file and os.path.exists(exclusions_file):
            with open(exclusions_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        exclusions.add(line)
        return exclusions

    def _should_exclude(self, filepath: str) -> bool:
        """Check if file should be excluded from analysis."""
        for pattern in self.exclusions:
            if pattern in filepath or filepath.endswith(pattern.replace("*", "")):
                return True
        return False

    def _run_command(self, command: List[str], cwd: str) -> Tuple[int, str, str]:
        """Run a command and return results."""
        try:
            result = subprocess.run(
                command, cwd=cwd, capture_output=True, text=True, timeout=300
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def _parse_ruff_output(self, output: str, cwd: str) -> Dict[str, List[Dict]]:
        """Parse ruff output and organize by file."""
        file_issues = defaultdict(list)

        for line in output.strip().split("\n"):
            if not line or ":" not in line:
                continue

            try:
                # Parse ruff output format: file:line:col: code message
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    filepath = parts[0]
                    line_num = int(parts[1])
                    col_num = int(parts[2])
                    message = parts[3].strip()

                    # Make filepath relative to cwd
                    if filepath.startswith(cwd):
                        filepath = os.path.relpath(filepath, cwd)

                    file_issues[filepath].append(
                        {
                            "tool": "ruff",
                            "lineno": line_num,
                            "col": col_num,
                            "message": message,
                            "type": "linting",
                        }
                    )
            except (ValueError, IndexError):
                continue

        return dict(file_issues)

    def _parse_mypy_output(self, output: str, cwd: str) -> Dict[str, List[Dict]]:
        """Parse mypy output and organize by file."""
        file_issues = defaultdict(list)

        for line in output.strip().split("\n"):
            if not line or ":" not in line:
                continue

            try:
                # Parse mypy output format: file:line: message
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    filepath = parts[0]
                    line_num = int(parts[1])
                    message = parts[2].strip()

                    # Make filepath relative to cwd
                    if filepath.startswith(cwd):
                        filepath = os.path.relpath(filepath, cwd)

                    file_issues[filepath].append(
                        {
                            "tool": "mypy",
                            "lineno": line_num,
                            "message": message,
                            "type": "type_error",
                        }
                    )
            except (ValueError, IndexError):
                continue

        return dict(file_issues)

    def _parse_radon_output(self, output: str, cwd: str) -> Dict[str, List[Dict]]:
        """Parse radon output and organize by file."""
        file_issues = defaultdict(list)

        for line in output.strip().split("\n"):
            if not line or ":" not in line:
                continue

            try:
                # Parse radon output format: file:line:function:complexity
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    filepath = parts[0]
                    line_num = int(parts[1])
                    function = parts[2]
                    complexity = parts[3].strip()

                    # Make filepath relative to cwd
                    if filepath.startswith(cwd):
                        filepath = os.path.relpath(filepath, cwd)

                    # Only flag high complexity functions
                    try:
                        comp_value = int(complexity.split()[0])
                        if comp_value > 10:  # Threshold for high complexity
                            file_issues[filepath].append(
                                {
                                    "tool": "radon",
                                    "lineno": line_num,
                                    "function": function,
                                    "complexity": comp_value,
                                    "message": f"High complexity function '{function}' (complexity: {comp_value})",
                                    "type": "complexity",
                                }
                            )
                    except (ValueError, IndexError):
                        continue
            except (ValueError, IndexError):
                continue

        return dict(file_issues)

    def _analyze_file_ast(self, filepath: str) -> Dict[str, List[Dict]]:
        """Analyze a single file using AST."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Try to parse with AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                # File has syntax errors
                return {
                    "syntax_errors": [
                        {
                            "tool": "ast_analysis",
                            "type": "syntax_error",
                            "lineno": e.lineno or 0,
                            "message": f"Syntax error: {e.msg}",
                            "suggestion": "Fix syntax error to enable further analysis",
                        }
                    ]
                }

            # File parsed successfully, analyze for issues
            issues = defaultdict(list)

            # Find unused imports
            issues["unused_imports"].extend(self._find_unused_imports(tree, content))

            # Find dead code
            issues["dead_code"].extend(self._find_dead_code(tree, content))

            # Find formatting issues (only critical ones)
            issues["formatting_issues"].extend(self._find_formatting_issues(content))

            # Find placeholder code
            issues["placeholder_issues"].extend(self._find_placeholder_code(content))

            # Find syntax issues
            issues["syntax_issues"].extend(self._find_syntax_issues(content))

            return dict(issues)

        except Exception as e:
            return {
                "syntax_errors": [
                    {
                        "tool": "ast_analysis",
                        "type": "syntax_error",
                        "lineno": 0,
                        "message": f"Error analyzing file: {e}",
                        "suggestion": "Check file accessibility and encoding",
                    }
                ]
            }

    def _find_unused_imports(self, tree: ast.AST, content: str) -> List[Dict]:
        """Find unused imports in the AST."""
        issues = []
        imports = []
        used_names = set()

        # Collect all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "import",
                            "name": alias.name,
                            "asname": alias.asname,
                            "lineno": node.lineno,
                            "used_name": alias.asname or alias.name.split(".")[0],
                        }
                    )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "from_import",
                            "module": node.module,
                            "name": alias.name,
                            "asname": alias.asname,
                            "lineno": node.lineno,
                            "used_name": alias.asname or alias.name,
                        }
                    )

        # Collect all used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                used_names.add(node.attr)

        # Find unused imports
        for imp in imports:
            if imp["used_name"] not in used_names:
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "unused_import",
                        "lineno": imp["lineno"],
                        "message": f"Unused import: {imp['name']}",
                        "suggestion": "Remove unused import or use it in the code",
                    }
                )

        return issues

    def _find_dead_code(self, tree: ast.AST, content: str) -> List[Dict]:
        """Find dead code patterns."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for functions that are never called
                # This is a simplified check
                pass
            elif isinstance(node, ast.ClassDef):
                # Check for classes that are never instantiated
                # This is a simplified check
                pass

        return issues

    def _find_formatting_issues(self, content: str) -> List[Dict]:
        """Find critical formatting issues in the content."""
        issues = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Only check for critical formatting issues that could cause problems
            # Skip PEP8 style violations like trailing whitespace

            # Check for mixed tabs and spaces (this can cause actual problems)
            if "\t" in line and "    " in line:
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "formatting",
                        "lineno": i,
                        "message": "Mixed tabs and spaces (can cause indentation errors)",
                    }
                )

        return issues

    def _find_placeholder_code(self, content: str) -> List[Dict]:
        """Find placeholder code patterns."""
        issues = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            line_lower = line.lower().strip()

            # Pass statements
            if line_lower == "pass":
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "placeholder",
                        "lineno": i,
                        "message": "Pass statement (placeholder code)",
                    }
                )

            # TODO comments
            elif "todo" in line_lower:
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "placeholder",
                        "lineno": i,
                        "message": "TODO comment found",
                    }
                )

            # NotImplementedError
            elif "notimplementederror" in line_lower:
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "placeholder",
                        "lineno": i,
                        "message": "NotImplementedError raised",
                    }
                )

        return issues

    def _find_syntax_issues(self, content: str) -> List[Dict]:
        """Find common syntax issues in the content."""
        issues = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith("#"):
                continue

            # Check for missing colons after function/class definitions, loops, etc.
            if self._is_missing_colon(line_stripped):
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "syntax_error",
                        "lineno": i,
                        "message": "Missing colon after function/class/loop definition",
                        "suggestion": 'Add ":" at the end of the line',
                    }
                )

            # Check for unmatched parentheses
            if self._has_unmatched_parentheses(line_stripped):
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "syntax_error",
                        "lineno": i,
                        "message": "Unmatched parentheses detected",
                        "suggestion": "Check for missing opening/closing parentheses",
                    }
                )

            # Check for unmatched brackets
            if self._has_unmatched_brackets(line_stripped):
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "syntax_error",
                        "lineno": i,
                        "message": "Unmatched brackets detected",
                        "suggestion": "Check for missing opening/closing brackets",
                    }
                )

            # Check for unmatched braces
            if self._has_unmatched_braces(line_stripped):
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "syntax_error",
                        "lineno": i,
                        "message": "Unmatched braces detected",
                        "suggestion": "Check for missing opening/closing braces",
                    }
                )

            # Check for unterminated strings
            if self._has_unterminated_string(line_stripped):
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "syntax_error",
                        "lineno": i,
                        "message": "Unterminated string literal",
                        "suggestion": "Check for missing quote marks",
                    }
                )

            # Check for invalid indentation
            if self._has_invalid_indentation(line):
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "syntax_error",
                        "lineno": i,
                        "message": "Invalid indentation",
                        "suggestion": "Check indentation consistency (spaces vs tabs)",
                    }
                )

            # Check for missing operators
            if self._is_missing_operator(line_stripped):
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "syntax_error",
                        "lineno": i,
                        "message": "Missing operator between expressions",
                        "suggestion": "Add appropriate operator (+, -, *, /, etc.)",
                    }
                )

            # Check for invalid syntax patterns
            if self._has_invalid_syntax_pattern(line_stripped):
                issues.append(
                    {
                        "tool": "ast_analysis",
                        "type": "syntax_error",
                        "lineno": i,
                        "message": "Invalid syntax pattern detected",
                        "suggestion": "Review syntax for correctness",
                    }
                )

        return issues

    def _is_missing_colon(self, line: str) -> bool:
        """Check if line is missing a colon after function/class/loop definition."""
        # Check for function definitions
        if re.match(r"^\s*def\s+\w+\s*\([^)]*\)\s*$", line):
            return True

        # Check for class definitions
        if re.match(r"^\s*class\s+\w+", line) and not line.rstrip().endswith(":"):
            return True

        # Check for loops and conditionals
        for keyword in [
            "if",
            "elif",
            "else",
            "for",
            "while",
            "try",
            "except",
            "finally",
            "with",
        ]:
            if re.match(rf"^\s*{keyword}\s+", line) and not line.rstrip().endswith(":"):
                return True

        return False

    def _has_unmatched_parentheses(self, line: str) -> bool:
        """Check for unmatched parentheses in a line."""
        open_count = line.count("(")
        close_count = line.count(")")
        return open_count != close_count

    def _has_unmatched_brackets(self, line: str) -> bool:
        """Check for unmatched brackets in a line."""
        open_count = line.count("[")
        close_count = line.count("]")
        return open_count != close_count

    def _has_unmatched_braces(self, line: str) -> bool:
        """Check for unmatched braces in a line."""
        open_count = line.count("{")
        close_count = line.count("}")
        return open_count != close_count

    def _has_unterminated_string(self, line: str) -> bool:
        """Check for unterminated string literals."""
        # Count different types of quotes
        single_quotes = line.count("'")
        double_quotes = line.count('"')
        triple_single = line.count("'''")
        triple_double = line.count('"""')

        # Check for odd counts (unterminated strings)
        if single_quotes % 2 == 1:
            return True
        if double_quotes % 2 == 1:
            return True
        if triple_single % 2 == 1:
            return True
        if triple_double % 2 == 1:
            return True

        return False

    def _has_invalid_indentation(self, line: str) -> bool:
        """Check for invalid indentation patterns that could cause problems."""
        # Check for mixed tabs and spaces (this can cause actual problems)
        if "\t" in line and "    " in line:
            return True

        # Skip strict 4-space rule checks as they are PEP8 style violations
        # Only check for actual problematic indentation patterns

        return False

    def _is_missing_operator(self, line: str) -> bool:
        """Check for missing operators between expressions."""
        # Look for patterns like "variable value" without operators
        # This is a simplified check - could be enhanced
        if re.search(r"\w+\s+\w+", line) and not re.search(r"[+\-*/=<>!&|^~%]", line):
            # Check if it's not a valid statement pattern
            if not re.match(
                r"^\s*(import|from|def|class|if|elif|else|for|while|try|except|finally|with|return|pass|break|continue|raise|assert|del|global|nonlocal)\s+",
                line,
            ):
                # Additional check: skip if it looks like a valid function call or attribute access
                if re.search(r"[a-zA-Z_]\w*\s*\(", line) or re.search(
                    r"[a-zA-Z_]\w*\s*\.", line
                ):
                    return False
                return True
        return False

    def _has_invalid_syntax_pattern(self, line: str) -> bool:
        """Check for other invalid syntax patterns."""
        # Check for invalid decimal literals (actual syntax errors)
        if re.search(r"\d+\.\d+\.\d+", line):
            return True

        # Check for unmatched quotes in f-strings (actual syntax errors)
        if 'f"' in line or "f'" in line:
            if self._has_unterminated_string(line):
                return True

        # Skip emoji checks as they are just style preferences

        return False

    def run_ruff_analysis(self, directory: str) -> Dict[str, List[Dict]]:
        """Run ruff linting analysis."""
        print("🔍 Running ruff linting analysis...")

        if not self._check_tool_available("ruff"):
            print("⚠️  ruff not available, skipping linting analysis")
            return {}

        command = ["ruff", "check", "--output-format=text", "."]
        returncode, stdout, stderr = self._run_command(command, directory)

        if returncode == 0 and not stdout.strip():
            print("✅ No linting issues found")
            return {}

        results = self._parse_ruff_output(stdout, directory)
        print(
            f"📊 Found {sum(len(issues) for issues in results.values())} linting issues"
        )
        return results

    def run_mypy_analysis(self, directory: str) -> Dict[str, List[Dict]]:
        """Run mypy type checking analysis."""
        print("🔍 Running mypy type checking analysis...")

        if not self._check_tool_available("mypy"):
            print("⚠️  mypy not available, skipping type checking analysis")
            return {}

        command = ["mypy", "--no-error-summary", "."]
        returncode, stdout, stderr = self._run_command(command, directory)

        if returncode == 0 and not stdout.strip():
            print("✅ No type errors found")
            return {}

        results = self._parse_mypy_output(stdout, directory)
        print(f"📊 Found {sum(len(issues) for issues in results.values())} type errors")
        return results

    def run_radon_analysis(self, directory: str) -> Dict[str, List[Dict]]:
        """Run radon complexity analysis."""
        print("🔍 Running radon complexity analysis...")

        if not self._check_tool_available("radon"):
            print("⚠️  radon not available, skipping complexity analysis")
            return {}

        command = ["radon", "cc", "-s", "-a", "."]
        returncode, stdout, stderr = self._run_command(command, directory)

        if returncode == 0 and not stdout.strip():
            print("✅ No complexity issues found")
            return {}

        results = self._parse_radon_output(stdout, directory)
        print(
            f"📊 Found {sum(len(issues) for issues in results.values())} complexity issues"
        )
        return results

    def _check_tool_available(self, tool: str) -> bool:
        """Check if a tool is available in PATH."""
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        print(f"🚀 Starting comprehensive code quality analysis of: {directory}")

        # Get current working directory
        cwd = os.getcwd()

        # Run external tool analyses
        ruff_results = self.run_ruff_analysis(directory)
        mypy_results = self.run_mypy_analysis(directory)
        radon_results = self.run_radon_analysis(directory)

        # Run AST-based analysis
        print("🔍 Running AST-based analysis...")
        ast_results = {}

        # Find all Python files
        directory_path = Path(directory)
        python_files = list(directory_path.rglob("*.py"))

        for filepath in python_files:
            if self._should_exclude(str(filepath)):
                continue

            print(f"  Analyzing: {filepath}")
            file_issues = self._analyze_file_ast(str(filepath))

            if file_issues:
                # Convert filepath to relative path
                rel_path = os.path.relpath(str(filepath), cwd)
                ast_results[rel_path] = file_issues

        # Aggregate all results
        self.tool_results = {
            "ruff": ruff_results,
            "mypy": mypy_results,
            "radon": radon_results,
            "ast": ast_results,
        }

        # Aggregate and update statistics
        self._update_statistics(self._aggregate_results())

        print(f"✅ Analysis complete! Analyzed {self.stats['files_analyzed']} files")
        return self._aggregate_results()

    def _aggregate_results(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Aggregate results from all tools into a unified structure."""
        aggregated = defaultdict(lambda: defaultdict(list))

        # Process AST analysis results
        for filepath, issues in self.tool_results["ast"].items():
            for issue_type, issue_list in issues.items():
                if isinstance(issue_list, list):
                    for issue in issue_list:
                        aggregated[filepath][issue_type].append(issue)

        # Process external tool results
        for tool_name, tool_results in self.tool_results.items():
            if tool_name == "ast":
                continue

            for filepath, issues in tool_results.items():
                if isinstance(issues, list):
                    for issue in issues:
                        aggregated[filepath][issue["type"]].append(issue)

        return dict(aggregated)

    def _update_statistics(self, results: Dict[str, Dict[str, List[Dict]]]):
        """Update statistics based on aggregated results."""
        for filepath, issues in results.items():
            self.stats["files_analyzed"] += 1

            for issue_type, issue_list in issues.items():
                if isinstance(issue_list, list):
                    if issue_type in self.stats:
                        self.stats[issue_type] += len(issue_list)
                    else:
                        self.stats[issue_type] = len(issue_list)

    def generate_report(self, results: Dict[str, Dict[str, List[Dict]]]) -> str:
        """Generate a comprehensive report."""
        report = []
        report.append("=" * 100)
        report.append("COMPREHENSIVE CODE QUALITY ANALYSIS REPORT")
        report.append("=" * 100)
        report.append("")

        # Summary statistics
        report.append("📊 SUMMARY STATISTICS:")
        report.append(f"  Files analyzed: {self.stats['files_analyzed']}")
        report.append(
            f"  Total issues found: {sum(self.stats.values()) - self.stats['files_analyzed']}"
        )
        report.append(f"  Syntax errors: {self.stats['syntax_errors']}")
        report.append(f"  Syntax issues: {self.stats['syntax_issues']}")
        report.append(f"  Linting issues: {self.stats['linting_issues']}")
        report.append(f"  Type errors: {self.stats['type_errors']}")
        report.append(f"  Complexity issues: {self.stats['complexity_issues']}")
        report.append(f"  Unused imports: {self.stats['unused_imports']}")
        report.append(f"  Dead code: {self.stats['dead_code']}")
        report.append(f"  Formatting issues: {self.stats['formatting_issues']}")
        report.append(f"  Placeholder issues: {self.stats['placeholder_issues']}")
        report.append("")

        # Per directory breakdown
        report.append("📁 PER DIRECTORY BREAKDOWN:")
        report.append("")

        # Group files by directory
        dir_files = defaultdict(list)
        for filepath in results.keys():
            dir_path = os.path.dirname(filepath) if os.path.dirname(filepath) else "."
            dir_files[dir_path].append(filepath)

        # Sort directories by total issues
        dir_issues = {}
        for dir_path, files in dir_files.items():
            total_issues = sum(
                len(issues)
                for filepath in files
                for issues in results[filepath].values()
            )
            dir_issues[dir_path] = total_issues

        sorted_dirs = sorted(dir_issues.items(), key=lambda x: x[1], reverse=True)

        for dir_path, total_issues in sorted_dirs:
            if total_issues == 0:
                continue

            files_in_dir = dir_files[dir_path]
            report.append(f"  📂 Directory: {dir_path}")
            report.append(
                f"    📊 Summary: {len(files_in_dir)} files, {total_issues} total issues"
            )

            # Issues by type for this directory
            dir_issue_types = defaultdict(int)
            for filepath in files_in_dir:
                for issue_type, issue_list in results[filepath].items():
                    if isinstance(issue_list, list):
                        dir_issue_types[issue_type] += len(issue_list)

            if dir_issue_types:
                report.append("    🔍 Issues by type:")
                for issue_type, count in sorted(
                    dir_issue_types.items(), key=lambda x: x[1], reverse=True
                ):
                    report.append(f"      • {issue_type}: {count}")

            # Files with issues in this directory
            report.append("    📄 Files with issues:")
            for filepath in sorted(
                files_in_dir,
                key=lambda x: sum(len(issues) for issues in results[x].values()),
                reverse=True,
            ):
                file_total = sum(len(issues) for issues in results[filepath].values())
                if file_total > 0:
                    report.append(f"      📝 {filepath}: {file_total} issues")
                    # Show breakdown by issue type
                    for issue_type, issue_list in results[filepath].items():
                        if isinstance(issue_list, list) and len(issue_list) > 0:
                            report.append(f"        - {issue_type}: {len(issue_list)}")
            report.append("")

        # Detailed per file breakdown
        report.append("📄 DETAILED PER FILE BREAKDOWN:")
        report.append("")

        # Sort files by total issues
        file_issues = {}
        for filepath, issues in results.items():
            total_issues = sum(
                len(issue_list)
                for issue_list in issues.values()
                if isinstance(issue_list, list)
            )
            file_issues[filepath] = total_issues

        sorted_files = sorted(file_issues.items(), key=lambda x: x[1], reverse=True)

        for filepath, total_issues in sorted_files:
            if total_issues == 0:
                continue

            report.append(f"  📁 File: {filepath}")
            report.append("  " + "=" * 50)

            for issue_type, issue_list in results[filepath].items():
                if not isinstance(issue_list, list) or len(issue_list) == 0:
                    continue

                report.append(f"    🔍 {issue_type.title()}: {len(issue_list)} issues")

                # Show first 3 examples of each issue type
                for i, issue in enumerate(issue_list[:3]):
                    if isinstance(issue, dict):
                        message = issue.get("message", "Unknown issue")
                        lineno = issue.get("lineno", 0)
                        suggestion = issue.get("suggestion", "")

                        report.append(
                            f"      • [{issue.get('tool', 'unknown')}] {message} (line {lineno})"
                        )
                        if suggestion:
                            report.append(f"        💡 {suggestion}")

                if len(issue_list) > 3:
                    report.append(f"      ... and {len(issue_list) - 3} more issues")
                report.append("")

            report.append(f"    📊 Total issues in file: {total_issues}")
            report.append("")

        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Python code quality analyzer"
    )
    parser.add_argument("directory", help="Directory to analyze")
    parser.add_argument("--exclusions", help="Exclusions file path")
    parser.add_argument("--output", help="Output report to file")
    parser.add_argument(
        "--skip-tools",
        nargs="+",
        choices=["ruff", "mypy", "radon"],
        help="Skip specific tool analyses",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = CodeQualityAnalyzer(exclusions_file=args.exclusions)

    # Analyze directory
    results = analyzer.analyze_directory(args.directory)

    # Generate report
    report = analyzer.generate_report(results)

    # Output report
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"📄 Report written to {args.output}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
