"""
autopep8 code formatter plugin for code quality tools.
"""

import subprocess
import sys
from typing import Any

from code_quality.core.plugins import BaseCodeFixer


class Autopep8Fixer(BaseCodeFixer):
    """autopep8 code formatter plugin."""

    def __init__(self, config: dict[str, Any] = None):
        super().__init__(config)
        self.name = "autopep8"
        self.description = "Automatically formats Python code to conform to PEP 8"
        self.version = "1.0.0"

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_version(self) -> str:
        return self.version

    def can_fix(self, file_path: str) -> bool:
        return file_path.endswith(".py")

    def fix(self, file_path: str) -> dict[str, Any]:
        """Fix formatting issues using autopep8."""
        try:
            cmd: list[str] = [
                sys.executable, "-m", "autopep8",
                "--in-place",
                "--max-line-length", str(self.get_config("max_line_length", 88)),
            ]

            # Add aggressive passes if configured
            if self.get_config("aggressive", False):
                cmd.extend(["--aggressive", "--aggressive"])  # two passes is common

            cmd.append(file_path)

            result = subprocess.run(
                cmd,
                check=False, capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "tool": "autopep8",
                    "file": file_path,
                    "message": "Successfully formatted with autopep8",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            return {
                "success": False,
                "tool": "autopep8",
                "file": file_path,
                "message": f"autopep8 failed with return code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": True,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "tool": "autopep8",
                "file": file_path,
                "message": "autopep8 timed out after 60 seconds",
                "error": True,
                "timeout": True,
            }
        except Exception as e:
            return {
                "success": False,
                "tool": "autopep8",
                "file": file_path,
                "message": f"autopep8 failed with exception: {str(e)}",
                "error": True,
                "exception": str(e),
            }

    def get_supported_extensions(self) -> list[str]:
        return [".py"]

    def check_installed(self) -> bool:
        try:
            subprocess.run(
                [sys.executable, "-m", "autopep8", "--version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

