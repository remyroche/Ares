"""
Pyupgrade fixer plugin to modernize Python syntax.
"""

import subprocess
import sys
from typing import Any

from code_quality.core.plugins import BaseCodeFixer


class PyupgradeFixer(BaseCodeFixer):
    """Run pyupgrade with optional target version."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "Pyupgrade"
        self.description = "Modernize Python syntax with pyupgrade"
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
            subprocess.run([sys.executable, "-m", "pyupgrade", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def fix(self, file_path: str) -> dict[str, Any]:
        if not self._check_available():
            return {
                "success": False,
                "tool": "pyupgrade",
                "file": file_path,
                "message": "pyupgrade not installed",
                "skipped": True,
            }

        target = self.get_config("py311_plus", False)
        args = ["--py311-plus"] if target else []

        try:
            cmd = [sys.executable, "-m", "pyupgrade", *args, file_path]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            ok = result.returncode == 0
            return {
                "success": ok,
                "tool": "pyupgrade",
                "file": file_path,
                "message": "pyupgrade applied" if ok else "pyupgrade failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "tool": "pyupgrade",
                "file": file_path,
                "message": f"pyupgrade exception: {exc}",
                "exception": str(exc),
            }

