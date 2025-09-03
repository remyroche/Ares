"""
Ruff linter plugin for code quality tools.
"""

import contextlib
import json
import subprocess
import sys
from typing import Any

from code_quality.core.plugins import BaseCodeAnalyzer


class RuffAnalyzer(BaseCodeAnalyzer):
    """Ruff fast Python linter plugin."""

    def __init__(self, config: dict[str, Any] = None):
        super().__init__(config)
        self.name = "ruff"
        self.description = "Extremely fast Python linter (Rust)"
        self.version = "1.0.0"

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_version(self) -> str:
        return self.version

    def can_analyze(self, file_path: str) -> bool:
        return file_path.endswith(".py")

    def analyze(self, file_path: str) -> dict[str, Any]:
        """Analyze code quality using Ruff."""
        try:
            cmd: list[str] = [
                sys.executable, "-m", "ruff",
                "check",
                "--output-format", "json",
                file_path,
            ]

            result = subprocess.run(
                cmd,
                check=False, capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode in (0, 1):
                issues = []
                if result.stdout.strip():
                    with contextlib.suppress(json.JSONDecodeError):
                        issues = json.loads(result.stdout)
                return {
                    "success": True,
                    "tool": "ruff",
                    "file": file_path,
                    "issues": issues,
                    "total_issues": len(issues),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            return {
                "success": False,
                "tool": "ruff",
                "file": file_path,
                "message": f"Ruff failed with return code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": True,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "tool": "ruff",
                "file": file_path,
                "message": "Ruff timed out after 60 seconds",
                "error": True,
                "timeout": True,
            }
        except Exception as e:
            return {
                "success": False,
                "tool": "ruff",
                "file": file_path,
                "message": f"Ruff failed with exception: {str(e)}",
                "error": True,
                "exception": str(e),
            }

    def get_supported_extensions(self) -> list[str]:
        return [".py"]

    def check_installed(self) -> bool:
        try:
            subprocess.run(
                [sys.executable, "-m", "ruff", "--version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

