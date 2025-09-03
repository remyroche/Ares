"""
Flynt fixer plugin to convert strings to f-strings where safe.
"""

import subprocess
import sys
from typing import Any

from code_quality.core.plugins import BaseCodeFixer


class FlyntFixer(BaseCodeFixer):
    """Run flynt to modernize string formatting to f-strings."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "Flynt"
        self.description = "Convert string formatting to f-strings"
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
            subprocess.run([sys.executable, "-m", "flynt", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def fix(self, file_path: str) -> dict[str, Any]:
        if not self._check_available():
            return {
                "success": False,
                "tool": "flynt",
                "file": file_path,
                "message": "flynt not installed",
                "skipped": True,
            }

        aggressive = self.get_config("aggressive", False)
        args = ["--aggressive"] if aggressive else []

        try:
            cmd = [sys.executable, "-m", "flynt", *args, file_path]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            ok = result.returncode == 0
            return {
                "success": ok,
                "tool": "flynt",
                "file": file_path,
                "message": "flynt applied" if ok else "flynt failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "tool": "flynt",
                "file": file_path,
                "message": f"flynt exception: {exc}",
                "exception": str(exc),
            }

