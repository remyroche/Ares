"""
Flake8 linter plugin for code quality tools.
"""

import json
import subprocess
import sys
from typing import Any

from code_quality.core.plugins import BaseCodeAnalyzer


class Flake8Analyzer(BaseCodeAnalyzer):
    """Flake8 linter plugin."""

    def __init__(self, config: dict[str, Any] = None):
        super().__init__(config)
        self.name = "Flake8"
        self.description = "Python linter for style guide enforcement"
        self.version = "1.0.0"

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_version(self) -> str:
        return self.version

    def can_analyze(self, file_path: str) -> bool:
        """Check if Flake8 can analyze this file."""
        return file_path.endswith(".py")

    def analyze(self, file_path: str) -> dict[str, Any]:
        """Analyze code quality using Flake8."""
        try:
            # Build Flake8 command
            cmd = [
                sys.executable, "-m", "flake8",
                "--format", "json",
                "--max-line-length", str(self.get_config("max_line_length", 88)),
            ]

            # Add ignore patterns if configured
            ignore_patterns = self.get_config("ignore_patterns", [])
            if ignore_patterns:
                cmd.extend(["--ignore", ",".join(ignore_patterns)])

            # Add select patterns if configured
            select_patterns = self.get_config("select_patterns", [])
            if select_patterns:
                cmd.extend(["--select", ",".join(select_patterns)])

            cmd.append(file_path)

            # Run Flake8
            result = subprocess.run(
                cmd,
                check=False, capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                # No issues found
                return {
                    "success": True,
                    "tool": "flake8",
                    "file": file_path,
                    "message": "No issues found by Flake8",
                    "issues": [],
                    "total_issues": 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            if result.returncode == 1:
                # Issues found (this is normal for Flake8)
                try:
                    issues = json.loads(result.stdout) if result.stdout.strip() else []
                    return {
                        "success": True,
                        "tool": "flake8",
                        "file": file_path,
                        "message": f"Found {len(issues)} issues with Flake8",
                        "issues": issues,
                        "total_issues": len(issues),
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    return {
                        "success": True,
                        "tool": "flake8",
                        "file": file_path,
                        "message": "Issues found by Flake8 (parsing failed)",
                        "issues": [],
                        "total_issues": 0,
                        "raw_output": result.stdout,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
            else:
                return {
                    "success": False,
                    "tool": "flake8",
                    "file": file_path,
                    "message": f"Flake8 failed with return code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "error": True,
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "tool": "flake8",
                "file": file_path,
                "message": "Flake8 timed out after 60 seconds",
                "error": True,
                "timeout": True,
            }
        except Exception as e:
            return {
                "success": False,
                "tool": "flake8",
                "file": file_path,
                "message": f"Flake8 failed with exception: {str(e)}",
                "error": True,
                "exception": str(e),
            }

    def get_supported_extensions(self) -> list[str]:
        return [".py"]

    def check_installed(self) -> bool:
        """Check if Flake8 is installed."""
        try:
            subprocess.run(
                [sys.executable, "-m", "flake8", "--version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_issue_summary(self, issues: list[dict[str, Any]]) -> dict[str, Any]:
        """Get a summary of Flake8 issues."""
        if not issues:
            return {"total": 0, "by_severity": {}, "by_code": {}}

        by_severity = {}
        by_code = {}

        for issue in issues:
            # Count by severity
            severity = issue.get("severity", "unknown")
            by_severity[severity] = by_severity.get(severity, 0) + 1

            # Count by error code
            code = issue.get("code", "unknown")
            by_code[code] = by_code.get(code, 0) + 1

        return {
            "total": len(issues),
            "by_severity": by_severity,
            "by_code": by_code,
        }
