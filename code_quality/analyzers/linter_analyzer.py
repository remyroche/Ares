"""
Linter analyzer module for running various Python linters and collecting error reports.
"""

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..core.config import CodeQualityConfig, get_default_config
from ..utils.file_utils import find_python_files


class LinterResult:
    """Container for linter results."""

    def __init__(self, linter_name: str, file_path: str, line: int, column: int,
                 message: str, error_code: str, severity: str = "error"):
        self.linter_name = linter_name
        self.file_path = file_path
        self.line = line
        self.column = column
        self.message = message
        self.error_code = error_code
        self.severity = severity

    def __repr__(self):
        return f"LinterResult({self.linter_name}, {self.file_path}:{self.line}:{self.column}, {self.message})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "linter_name": self.linter_name,
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity,
        }


class LinterAnalyzer:
    """
    Analyzes Python code using various linters and collects error reports.
    """

    def __init__(self, config: CodeQualityConfig | None = None):
        self.config = config or get_default_config()
        self.results: list[LinterResult] = []
        self.linter_outputs: dict[str, dict[str, Any]] = {}

    def analyze_directory(self, directory: str) -> dict[str, Any]:
        """
        Analyze all Python files in a directory using configured linters.

        Args:
            directory: Directory containing Python files to analyze

        Returns:
            Dictionary containing analysis results
        """
        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Analyzing {len(python_files)} Python files with {len(self.config.analysis.linters)} linters...")

        self.results.clear()
        self.linter_outputs.clear()

        for linter in self.config.analysis.linters:
            if linter == "flake8":
                self._run_flake8(python_files)
            elif linter == "pylint":
                self._run_pylint(python_files)
            elif linter == "mypy":
                self._run_mypy(python_files)
            elif linter == "pycodestyle":
                self._run_pycodestyle(python_files)
            elif linter == "pyflakes":
                self._run_pyflakes(python_files)
            else:
                print(f"Warning: Unknown linter '{linter}' configured.")

        return self._generate_summary()

    def _run_flake8(self, files: list[str]) -> None:
        """Run flake8 linter."""
        print("Running flake8...")
        try:
            cmd = [
                sys.executable, "-m", "flake8",
                "--format", "json",
                "--max-line-length", str(self.config.auto_fix.max_line_length),
            ]

            cmd.extend(files)

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode in [0, 1]:  # flake8 returns 1 when issues are found
                try:
                    # Parse JSON output
                    if result.stdout.strip():
                        issues = json.loads(result.stdout)
                        for issue in issues:
                            linter_result = LinterResult(
                                linter_name="flake8",
                                file_path=issue["filename"],
                                line=issue["line_number"],
                                column=issue["column_number"],
                                message=issue["text"],
                                error_code=issue["code"],
                                severity="error" if issue["code"].startswith("E") else "warning",
                            )
                            self.results.append(linter_result)

                    self.linter_outputs["flake8"] = {
                        "status": "success",
                        "files_processed": len(files),
                        "issues_found": len([r for r in self.results if r.linter_name == "flake8"]),
                    }

                except json.JSONDecodeError:
                    print("Warning: Could not parse flake8 JSON output")
                    self.linter_outputs["flake8"] = {"status": "parse_error"}
            else:
                print(f"flake8 failed: {result.stderr}")
                self.linter_outputs["flake8"] = {"status": "failed", "error": result.stderr}

        except Exception as e:
            print(f"Error running flake8: {e}")
            self.linter_outputs["flake8"] = {"status": "error", "error": str(e)}

    def _run_pylint(self, files: list[str]) -> None:
        """Run pylint linter."""
        print("Running pylint...")
        try:
            cmd = [
                sys.executable, "-m", "pylint",
                "--output-format", "json",
                "--disable", "C0114,C0115,C0116",  # Disable docstring warnings for now
                "--max-line-length", str(self.config.auto_fix.max_line_length),
            ]

            cmd.extend(files)

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode in [0, 1, 2]:  # pylint returns various codes
                try:
                    # Parse JSON output
                    if result.stdout.strip():
                        issues = json.loads(result.stdout)
                        for issue in issues:
                            linter_result = LinterResult(
                                linter_name="pylint",
                                file_path=issue["path"],
                                line=issue["line"],
                                column=issue["column"],
                                message=issue["message"],
                                error_code=issue["symbol"],
                                severity=issue["type"],
                            )
                            self.results.append(linter_result)

                    self.linter_outputs["pylint"] = {
                        "status": "success",
                        "files_processed": len(files),
                        "issues_found": len([r for r in self.results if r.linter_name == "pylint"]),
                    }

                except json.JSONDecodeError:
                    print("Warning: Could not parse pylint JSON output")
                    self.linter_outputs["pylint"] = {"status": "parse_error"}
            else:
                print(f"pylint failed: {result.stderr}")
                self.linter_outputs["pylint"] = {"status": "failed", "error": result.stderr}

        except Exception as e:
            print(f"Error running pylint: {e}")
            self.linter_outputs["pylint"] = {"status": "error", "error": str(e)}

    def _run_mypy(self, files: list[str]) -> None:
        """Run mypy type checker."""
        print("Running mypy...")
        try:
            cmd = [
                sys.executable, "-m", "mypy",
                "--no-error-summary",
                "--show-error-codes",
                "--ignore-missing-imports",
                "--no-incremental",
            ]

            cmd.extend(files)

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            # mypy doesn't have a simple JSON output format, so we parse the text output
            if result.returncode in [0, 1]:  # mypy returns 1 when issues are found
                issues_found = 0
                for line in result.stdout.splitlines():
                    if ":" in line and "error:" in line:
                        # Parse mypy output format: file:line: error: message
                        match = re.match(r"(.+):(\d+):\s*error:\s*(.+)", line)
                        if match:
                            file_path, line_num, message = match.groups()
                            linter_result = LinterResult(
                                linter_name="mypy",
                                file_path=file_path,
                                line=int(line_num),
                                column=0,
                                message=message.strip(),
                                error_code="mypy-error",
                                severity="error",
                            )
                            self.results.append(linter_result)
                            issues_found += 1

                self.linter_outputs["mypy"] = {
                    "status": "success",
                    "files_processed": len(files),
                    "issues_found": issues_found,
                }
            else:
                print(f"mypy failed: {result.stderr}")
                self.linter_outputs["mypy"] = {"status": "failed", "error": result.stderr}

        except Exception as e:
            print(f"Error running mypy: {e}")
            self.linter_outputs["mypy"] = {"status": "error", "error": str(e)}

    def _run_pycodestyle(self, files: list[str]) -> None:
        """Run pycodestyle linter."""
        print("Running pycodestyle...")
        try:
            cmd = [
                sys.executable, "-m", "pycodestyle",
                "--format", "json",
                "--max-line-length", str(self.config.auto_fix.max_line_length),
            ]

            cmd.extend(files)

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode in [0, 1]:  # pycodestyle returns 1 when issues are found
                try:
                    # Parse JSON output
                    if result.stdout.strip():
                        issues = json.loads(result.stdout)
                        for issue in issues:
                            linter_result = LinterResult(
                                linter_name="pycodestyle",
                                file_path=issue["filename"],
                                line=issue["line_number"],
                                column=issue["column_number"],
                                message=issue["text"],
                                error_code=issue["code"],
                                severity="error",
                            )
                            self.results.append(linter_result)

                    self.linter_outputs["pycodestyle"] = {
                        "status": "success",
                        "files_processed": len(files),
                        "issues_found": len([r for r in self.results if r.linter_name == "pycodestyle"]),
                    }

                except json.JSONDecodeError:
                    print("Warning: Could not parse pycodestyle JSON output")
                    self.linter_outputs["pycodestyle"] = {"status": "parse_error"}
            else:
                print(f"pycodestyle failed: {result.stderr}")
                self.linter_outputs["pycodestyle"] = {"status": "failed", "error": result.stderr}

        except Exception as e:
            print(f"Error running pycodestyle: {e}")
            self.linter_outputs["pycodestyle"] = {"status": "error", "error": str(e)}

    def _run_pyflakes(self, files: list[str]) -> None:
        """Run pyflakes linter."""
        print("Running pyflakes...")
        try:
            cmd = [
                sys.executable, "-m", "pyflakes",
            ]

            cmd.extend(files)

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode in [0, 1]:  # pyflakes returns 1 when issues are found
                issues_found = 0
                for line in result.stdout.splitlines():
                    if ":" in line:
                        # Parse pyflakes output format: file:line: message
                        match = re.match(r"(.+):(\d+):\s*(.+)", line)
                        if match:
                            file_path, line_num, message = match.groups()
                            linter_result = LinterResult(
                                linter_name="pyflakes",
                                file_path=file_path,
                                line=int(line_num),
                                column=0,
                                message=message.strip(),
                                error_code="pyflakes-error",
                                severity="error",
                            )
                            self.results.append(linter_result)
                            issues_found += 1

                self.linter_outputs["pyflakes"] = {
                    "status": "success",
                    "files_processed": len(files),
                    "issues_found": issues_found,
                }
            else:
                print(f"pyflakes failed: {result.stderr}")
                self.linter_outputs["pyflakes"] = {"status": "failed", "error": result.stderr}

        except Exception as e:
            print(f"Error running pyflakes: {e}")
            self.linter_outputs["pyflakes"] = {"status": "error", "error": str(e)}

    def _generate_summary(self) -> dict[str, Any]:
        """Generate a summary of all linter results."""
        # Group by file
        file_errors = defaultdict(list)
        for result in self.results:
            file_errors[result.file_path].append(result)

        # Group by directory
        dir_errors = defaultdict(lambda: {"files": 0, "errors": 0, "warnings": 0})
        for file_path, errors in file_errors.items():
            dir_path = str(Path(file_path).parent)
            dir_errors[dir_path]["files"] += 1
            dir_errors[dir_path]["errors"] += len([e for e in errors if e.severity == "error"])
            dir_errors[dir_path]["warnings"] += len([e for e in errors if e.severity == "warning"])

        # Count by linter
        linter_counts = defaultdict(int)
        for result in self.results:
            linter_counts[result.linter_name] += 1

        # Count by error type
        error_type_counts = defaultdict(int)
        for result in self.results:
            error_type_counts[result.error_code] += 1

        return {
            "total_issues": len(self.results),
            "total_files_with_issues": len(file_errors),
            "total_errors": len([r for r in self.results if r.severity == "error"]),
            "total_warnings": len([r for r in self.results if r.severity == "warning"]),
            "by_file": {file_path: [r.to_dict() for r in errors] for file_path, errors in file_errors.items()},
            "by_directory": dict(dir_errors),
            "by_linter": dict(linter_counts),
            "by_error_type": dict(error_type_counts),
            "linter_status": self.linter_outputs,
            "raw_results": [r.to_dict() for r in self.results],
        }


    def get_results(self) -> list[LinterResult]:
        """Get all linter results."""
        return self.results

    def get_file_errors(self, file_path: str) -> list[LinterResult]:
        """Get errors for a specific file."""
        return [r for r in self.results if r.file_path == file_path]

    def get_directory_errors(self, directory: str) -> list[LinterResult]:
        """Get errors for a specific directory."""
        return [r for r in self.results if r.file_path.startswith(directory)]


def main():
    """Command-line interface for the linter analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Python code with various linters")
    parser.add_argument("--path", required=True, help="Path to directory containing Python files")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output file for results (JSON)")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        from ..core.config import load_config
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Run linter analysis
    analyzer = LinterAnalyzer(config)
    results = analyzer.analyze_directory(args.path)

    # Print summary
    print("\n" + "="*50)
    print("LINTER ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total issues found: {results['total_issues']}")
    print(f"Files with issues: {results['total_files_with_issues']}")
    print(f"Errors: {results['total_errors']}")
    print(f"Warnings: {results['total_warnings']}")

    print("\nIssues by linter:")
    for linter, count in results["by_linter"].items():
        print(f"  {linter}: {count}")

    print("\nTop error types:")
    sorted_errors = sorted(results["by_error_type"].items(), key=lambda x: x[1], reverse=True)
    for error_type, count in sorted_errors[:10]:
        print(f"  {error_type}: {count}")

    # Save results if requested
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
