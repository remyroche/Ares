"""Detect usage of disallowed DataFrame column prefixes.

The pre-training pipeline relies on standardized double-underscore namespaces
for column naming. This check scans Python sources for assignments that use the
legacy single-underscore prefixes (e.g. ``target_``) so that regressions are
caught during CI.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

DISALLOWED_PREFIXES = ("target_", "label_", "feature_", "meta_")
ALLOWED_PREFIXES = ("target__", "label__", "feature__", "meta__")
COLUMN_CONTAINER_HINTS = ("df", "frame", "labels", "targets")
DEFAULT_SEARCH_ROOTS = [Path("src/training/steps/pre_training")]


@dataclass
class Violation:
    filename: Path
    lineno: int
    prefix: str
    literal: str

    def format(self) -> str:
        return f"{self.filename}:{self.lineno}: disallowed column prefix '{self.prefix}' in literal '{self.literal}'"


class ColumnPrefixVisitor(ast.NodeVisitor):
    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.violations: List[Violation] = []

    def visit_Assign(self, node: ast.Assign) -> None:  # pragma: no cover - simple traversal
        for target in node.targets:
            self._inspect_target(target)
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # pragma: no cover - simple traversal
        self._inspect_target(node.target)
        if node.value is not None:
            self.generic_visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # pragma: no cover - simple traversal
        self._inspect_target(node.target)
        self.generic_visit(node.value)

    def _inspect_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Subscript):
            container_name = self._extract_container_name(target.value)
            if container_name and any(hint in container_name for hint in COLUMN_CONTAINER_HINTS):
                literal = self._extract_literal(target.slice)
                if literal is not None:
                    self._check_literal(literal, getattr(target, "lineno", 0))
        self.generic_visit(target)

    def _extract_container_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id.lower()
        if isinstance(node, ast.Attribute):
            return node.attr.lower()
        return None

    def _extract_literal(self, node: ast.AST | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Index):  # Python <3.9 compatibility
            return self._extract_literal(node.value)
        return None

    def _check_literal(self, value: str, lineno: int) -> None:
        if any(value.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            return
        for prefix in DISALLOWED_PREFIXES:
            if value.startswith(prefix):
                self.violations.append(
                    Violation(filename=self.filename, lineno=lineno or 0, prefix=prefix, literal=value)
                )
                break


def iter_python_files(paths: Sequence[Path] | None = None) -> Iterable[Path]:
    if paths:
        for path in paths:
            if path.is_file() and path.suffix == ".py":
                yield path
            elif path.is_dir():
                yield from (p for p in path.rglob("*.py") if p.is_file())
        return

    git_ls = subprocess.run(
        ["git", "ls-files", "*.py"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    tracked_files = [Path(line) for line in git_ls.stdout.splitlines() if line]

    for root in DEFAULT_SEARCH_ROOTS:
        for path in tracked_files:
            try:
                if path.is_relative_to(root):
                    yield path
            except AttributeError:  # pragma: no cover - Python <3.9 fallback
                if str(path).startswith(str(root)):
                    yield path


def scan_file(path: Path) -> List[Violation]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    visitor = ColumnPrefixVisitor(path)
    visitor.visit(tree)
    return visitor.violations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check for disallowed DataFrame column prefixes")
    parser.add_argument("paths", nargs="*", type=Path, help="Paths to scan. Defaults to git tracked Python files.")
    args = parser.parse_args(argv)

    violations: List[Violation] = []
    for file_path in iter_python_files(args.paths):
        violations.extend(scan_file(file_path))

    if violations:
        for violation in sorted(violations, key=lambda v: (v.filename, v.lineno)):
            print(violation.format(), file=sys.stderr)
        print(
            "\nFound disallowed column prefixes. Use the double-underscore namespaces (feat__, label__, target__, meta__).",
            file=sys.stderr,
        )
        return 1

    print("✅ Column prefix check passed: no legacy prefixes detected.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
