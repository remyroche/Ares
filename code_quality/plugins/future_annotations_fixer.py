"""
Future annotations injector: add `from __future__ import annotations` at top if missing.
"""

from typing import Any

from code_quality.core.plugins import BaseCodeFixer


class FutureAnnotationsFixer(BaseCodeFixer):
    """Ensure `from __future__ import annotations` is present when enabled."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "FutureAnnotations"
        self.description = "Inject from __future__ import annotations"
        self.version = "1.0.0"

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_version(self) -> str:
        return self.version

    def can_fix(self, file_path: str) -> bool:
        return file_path.endswith(".py") and self.get_config("enabled", True)

    def fix(self, file_path: str) -> dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            needle = "from __future__ import annotations"
            if needle in source:
                return {
                    "success": True,
                    "tool": "future_annotations",
                    "file": file_path,
                    "message": "already present",
                }

            # Insert after shebang or encoding declarations if present
            lines = source.splitlines()
            insert_idx = 0
            if lines and lines[0].startswith("#!"):
                insert_idx = 1
            if len(lines) > insert_idx and lines[insert_idx].startswith("# -*- coding:"):
                insert_idx += 1

            lines.insert(insert_idx, needle)
            new_source = "\n".join(lines)
            if not new_source.endswith("\n"):
                new_source += "\n"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_source)

            return {
                "success": True,
                "tool": "future_annotations",
                "file": file_path,
                "message": "injected",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "tool": "future_annotations",
                "file": file_path,
                "message": f"exception: {exc}",
                "exception": str(exc),
            }

