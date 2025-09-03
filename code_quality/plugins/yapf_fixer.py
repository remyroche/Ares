"""
YAPF code formatter plugin for code quality tools.
"""

import subprocess
import sys
from typing import Any

from code_quality.core.plugins import BaseCodeFixer


class YapfFixer(BaseCodeFixer):
    """YAPF code formatter plugin."""

    def __init__(self, config: dict[str, Any] = None):
        super().__init__(config)
        self.name = "yapf"
        self.description = "Yet Another Python Formatter"
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
        """Fix formatting issues using YAPF."""
        try:
            cmd: list[str] = [
                sys.executable, "-m", "yapf",
                "-i",
                "--style", "pep8",
            ]

            # YAPF style can be tuned via config files; keep CLI simple.
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
                    "tool": "yapf",
                    "file": file_path,
                    "message": "Successfully formatted with YAPF",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            return {
                "success": False,
                "tool": "yapf",
                "file": file_path,
                "message": f"YAPF failed with return code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": True,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "tool": "yapf",
                "file": file_path,
                "message": "YAPF timed out after 60 seconds",
                "error": True,
                "timeout": True,
            }
        except Exception as e:
            return {
                "success": False,
                "tool": "yapf",
                "file": file_path,
                "message": f"YAPF failed with exception: {str(e)}",
                "error": True,
                "exception": str(e),
            }

    def get_supported_extensions(self) -> list[str]:
        return [".py"]

    def check_installed(self) -> bool:
        try:
            subprocess.run(
                [sys.executable, "-m", "yapf", "--version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

