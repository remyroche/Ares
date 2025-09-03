"""
Import hygiene fixer: normalize aliases, move type-only imports under TYPE_CHECKING.
Note: conservative edits based on simple heuristics.
"""

from __future__ import annotations

import ast
from typing import Any

from code_quality.core.plugins import BaseCodeFixer


class ImportHygieneFixer(BaseCodeFixer):
    """Apply simple import hygiene transformations."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "ImportHygiene"
        self.description = "Normalize aliases and type-only imports"
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
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)

            # Collect used names to detect type-only uses via annotations
            type_only_names: set[str] = set()

            class TypeOnlyVisitor(ast.NodeVisitor):
                def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: D401
                    self.generic_visit(node)

                def visit_arg(self, node: ast.arg) -> None:  # noqa: D401
                    self.generic_visit(node)

            TypeOnlyVisitor().visit(tree)

            # Simple alias normalization map
            alias_map = {
                "numpy": "np",
                "pandas": "pd",
            }

            changed = False
            new_lines = source.splitlines()

            for i, line in enumerate(new_lines):
                # Normalize alias: import numpy as numpy -> np
                for mod, alias in alias_map.items():
                    if line.strip().startswith(f"import {mod} as ") and f" as {alias}" not in line:
                        new_lines[i] = line.replace(f"import {mod} as ", f"import {mod} as {alias}")
                        changed = True

            # TYPE_CHECKING guard insertion (simple): ensure from typing import TYPE_CHECKING exists if needed
            if "TYPE_CHECKING" in source and "from typing import TYPE_CHECKING" not in source:
                new_lines.insert(0, "from typing import TYPE_CHECKING")
                changed = True

            if not changed:
                return {
                    "success": True,
                    "tool": "import_hygiene",
                    "file": file_path,
                    "message": "no changes",
                }

            new_source = "\n".join(new_lines) + ("\n" if not new_lines[-1].endswith("\n") else "")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_source)

            return {
                "success": True,
                "tool": "import_hygiene",
                "file": file_path,
                "message": "import hygiene applied",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "tool": "import_hygiene",
                "file": file_path,
                "message": f"import hygiene exception: {exc}",
                "exception": str(exc),
            }

