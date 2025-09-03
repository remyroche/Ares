"""
Ruff fixer plugin (format + check --fix).
"""

import subprocess
import sys
from typing import Any

from code_quality.core.plugins import BaseCodeFixer


class RuffFixer(BaseCodeFixer):
    """Ruff formatter and linter auto-fix plugin."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "Ruff"
        self.description = "Ruff format and check --fix"
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
            subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def fix(self, file_path: str) -> dict[str, Any]:
        if not self._check_available():
            return {
                "success": False,
                "tool": "ruff",
                "file": file_path,
                "message": "Ruff not installed",
                "skipped": True,
            }

        line_length = int(self.get_config("max_line_length", 88))

        try:
            fmt_cmd = [sys.executable, "-m", "ruff", "format", "--line-length", str(line_length), file_path]
            fmt_result = subprocess.run(fmt_cmd, check=False, capture_output=True, text=True)

            chk_cmd = [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--fix",
                "--line-length",
                str(line_length),
                file_path,
            ]
            chk_result = subprocess.run(chk_cmd, check=False, capture_output=True, text=True)

            format_ok = fmt_result.returncode == 0
            check_ok = chk_result.returncode in (0, 1)
            overall = format_ok and check_ok

            return {
                "success": overall,
                "tool": "ruff",
                "file": file_path,
                "message": "Ruff format+check completed" if overall else "Ruff failed",
                "stdout": (fmt_result.stdout or "") + (chk_result.stdout or ""),
                "stderr": (fmt_result.stderr or "") + (chk_result.stderr or ""),
            }

        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "tool": "ruff",
                "file": file_path,
                "message": f"Ruff exception: {exc}",
                "exception": str(exc),
            }

