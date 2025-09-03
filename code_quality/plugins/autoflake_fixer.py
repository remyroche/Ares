"""
Autoflake fixer plugin to remove unused imports and variables.
"""

import subprocess
import sys
from typing import Any

from code_quality.core.plugins import BaseCodeFixer


class AutoflakeFixer(BaseCodeFixer):
    """Run autoflake to clean unused imports/variables."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "Autoflake"
        self.description = "Remove unused imports and variables"
        self.version = "1.0.0"

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_version(self) -> str:
        return self.version

    def can_fix(self, file_path: str) -> bool:
        return file_path.endswith(".py")

    def _check_available(self) -> bool:
        try:
            subprocess.run([sys.executable, "-m", "autoflake", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def fix(self, file_path: str) -> dict[str, Any]:
        if not self._check_available():
            return {
                "success": False,
                "tool": "autoflake",
                "file": file_path,
                "message": "autoflake not installed",
                "skipped": True,
            }

        args = [
            "--in-place",
            "--remove-all-unused-imports",
            "--ignore-init-module-imports",
        ]

        try:
            cmd = [sys.executable, "-m", "autoflake", *args, file_path]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            ok = result.returncode == 0
            return {
                "success": ok,
                "tool": "autoflake",
                "file": file_path,
                "message": "autoflake applied" if ok else "autoflake failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "tool": "autoflake",
                "file": file_path,
                "message": f"autoflake exception: {exc}",
                "exception": str(exc),
            }

