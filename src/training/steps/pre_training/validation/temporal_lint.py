"""Temporal leakage linting utilities.

This module provides lightweight linting helpers for detecting a handful of
common temporal leakage patterns in feature and label construction code. The
checks are intentionally conservative and target high risk pandas operations
that are easy to misuse when engineering time series inputs:

* ``rolling(..., center=True)`` – rolls centred on the current observation and
  therefore introduces look-ahead bias unless specifically guarded.
* ``shift(-n)`` – negative shifts pull future information backwards. They are
  only permitted inside labelling/target creation contexts where future values
  are expected.
* ``rolling(...)`` without an explicit ``closed=`` argument – the default can
  vary across pandas versions, so we require an explicit window closure to make
  the intent obvious.

The linter can be executed as a standalone script (``python -m ...``) or called
from other tooling such as pre-commit hooks or validation orchestrators.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

__all__ = [
    "TemporalLintError",
    "TemporalLintViolation",
    "lint_for_temporal_leakage",
    "run_temporal_linting",
    "main",
]

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalLintViolation:
    """A single temporal lint finding."""

    line: int
    message: str

    def format(self) -> str:
        return f"Line {self.line}: {self.message}"


class TemporalLintError(RuntimeError):
    """Raised when temporal leakage violations are detected."""


# ---------------------------------------------------------------------------
# Core linting helpers
# ---------------------------------------------------------------------------


_KEYWORDS = ("feature", "label", "target")
_DEFAULT_PATTERNS: Sequence[str] = (
    "src/**/*feature*.py",
    "src/**/*label*.py",
    "src/**/*target*.py",
)
_ALLOW_CENTER_COMMENT = "temporal-lint: allow-center"
_ALLOW_SHIFT_COMMENT = "temporal-lint: allow-shift"
_ALLOW_CLOSED_COMMENT = "temporal-lint: allow-closed"


def lint_for_temporal_leakage(file_path: Path | str) -> List[str]:
    """Inspect *file_path* and return temporal leakage violations.

    Args:
        file_path: Path to a Python source file.

    Returns:
        List of formatted violation messages (``"Line X: ..."``). The function
        never raises on its own – callers decide whether to treat the findings
        as fatal.
    """

    path = Path(file_path)
    if not path.exists():  # pragma: no cover - defensive guard for callers
        raise FileNotFoundError(f"Cannot lint missing file: {path}")

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - treat invalid syntax as fatal
        raise TemporalLintError(f"Failed to parse {path}: {exc}") from exc

    visitor = _TemporalLintVisitor(path=path, lines=lines)
    visitor.visit(tree)
    return [violation.format() for violation in visitor.violations]


def run_temporal_linting(
    paths: Optional[Sequence[str | Path]] = None,
    *,
    raise_on_violation: bool = True,
) -> Dict[str, List[str]]:
    """Run temporal linting across ``paths``.

    Args:
        paths: Optional iterable of files/directories to inspect. When omitted,
            the default feature/label glob patterns under ``src/`` are used.
        raise_on_violation: When ``True`` (default) a :class:`TemporalLintError`
            is raised if any violation is detected.

    Returns:
        Mapping of file path to violation messages. Empty when clean.
    """

    candidate_files = _collect_candidate_files(paths)
    violations: Dict[str, List[str]] = {}

    for file_path in candidate_files:
        try:
            file_violations = lint_for_temporal_leakage(file_path)
        except TemporalLintError as exc:
            file_violations = [str(exc)]

        if file_violations:
            violations[str(file_path)] = file_violations

    if violations and raise_on_violation:
        message_lines = ["Temporal leakage violations detected:"]
        for path, file_violations in sorted(violations.items()):
            message_lines.append(path)
            message_lines.extend(f"  - {msg}" for msg in file_violations)
        raise TemporalLintError("\n".join(message_lines))

    return violations


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point used by pre-commit/CI."""

    parser = argparse.ArgumentParser(description="Lint for temporal leakage patterns")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional files or directories to lint. When omitted the default feature/label patterns under src/ are scanned.",
    )
    parser.add_argument(
        "--no-raise",
        action="store_true",
        help="Do not raise on violations (useful for diagnostics).",
    )
    args = parser.parse_args(argv)

    try:
        results = run_temporal_linting(args.paths or None, raise_on_violation=not args.no_raise)
    except TemporalLintError as exc:
        print(str(exc))
        return 1

    if results:
        for path, file_violations in sorted(results.items()):
            print(path)
            for violation in file_violations:
                print(f"  - {violation}")
        return 1 if not args.no_raise else 0

    return 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_candidate_files(paths: Optional[Sequence[str | Path]]) -> List[Path]:
    if paths:
        collected: List[Path] = []
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                for candidate in sorted(path.rglob("*.py")):
                    if _looks_relevant(candidate):
                        collected.append(candidate)
            elif path.suffix == ".py" and path.exists():
                if _looks_relevant(path):
                    collected.append(path)
        return _deduplicate_preserve_order(collected)

    base = _repo_root()
    collected = []
    for pattern in _DEFAULT_PATTERNS:
        collected.extend(sorted(base.glob(pattern)))
    return _deduplicate_preserve_order(candidate for candidate in collected if _looks_relevant(candidate))


def _looks_relevant(path: Path) -> bool:
    lower_parts = [part.lower() for part in path.parts]
    return any(keyword in part for keyword in _KEYWORDS for part in lower_parts)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for _ in range(6):  # ascend to repository root (…/src -> …/Ares)
        current = current.parent
    return current


def _deduplicate_preserve_order(paths: Iterable[Path]) -> List[Path]:
    seen: set[Path] = set()
    ordered: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


class _TemporalLintVisitor(ast.NodeVisitor):
    def __init__(self, *, path: Path, lines: Sequence[str]):
        self.path = path
        self.lines = lines
        self.context_stack: List[bool] = []
        self.violations: List[TemporalLintViolation] = []

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _push_context(self, flag: bool) -> None:
        self.context_stack.append(flag)

    def _pop_context(self) -> None:
        if self.context_stack:
            self.context_stack.pop()

    def _in_label_context(self, lineno: int) -> bool:
        if any(self.context_stack):
            return True
        if 1 <= lineno <= len(self.lines):
            line = self.lines[lineno - 1].lower()
            if _ALLOW_SHIFT_COMMENT in line:
                return True
            return "label" in line or "target" in line
        return False

    def _line_has_allow_center(self, lineno: int) -> bool:
        if 1 <= lineno <= len(self.lines):
            return _ALLOW_CENTER_COMMENT in self.lines[lineno - 1].lower()
        return False

    def _line_has_allow_closed(self, lineno: int) -> bool:
        if 1 <= lineno <= len(self.lines):
            return _ALLOW_CLOSED_COMMENT in self.lines[lineno - 1].lower()
        return False

    def _add_violation(self, lineno: int, message: str) -> None:
        self.violations.append(TemporalLintViolation(line=lineno, message=message))

    # ------------------------------------------------------------------
    # Node visitors
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        flag = _looks_like_label_token(node.name)
        self._push_context(flag)
        self.generic_visit(node)
        self._pop_context()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # pragma: no cover - extremely rare
        flag = _looks_like_label_token(node.name)
        self._push_context(flag)
        self.generic_visit(node)
        self._pop_context()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        flag = _looks_like_label_token(node.name)
        self._push_context(flag)
        self.generic_visit(node)
        self._pop_context()

    def visit_Assign(self, node: ast.Assign) -> None:
        flag = any(_target_looks_like_label(target) for target in node.targets)
        self._push_context(flag)
        self.generic_visit(node)
        self._pop_context()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        flag = _target_looks_like_label(node.target)
        self._push_context(flag)
        self.generic_visit(node)
        self._pop_context()

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        flag = _target_looks_like_label(node.target)
        self._push_context(flag)
        self.generic_visit(node)
        self._pop_context()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            lineno = getattr(node, "lineno", 0)

            if attr == "rolling":
                if not self._line_has_allow_closed(lineno) and not _has_keyword(node, "closed"):
                    self._add_violation(lineno, "rolling() without explicit closed= parameter")

                if not self._line_has_allow_center(lineno) and _center_is_true(node):
                    self._add_violation(lineno, "rolling(..., center=True) uses future data")

            elif attr == "shift":
                shift_value = _extract_shift_amount(node)
                if shift_value is not None and shift_value < 0:
                    if not self._in_label_context(lineno):
                        self._add_violation(lineno, ".shift(-n) found outside label calculation")
        self.generic_visit(node)


def _looks_like_label_token(name: str) -> bool:
    lowered = name.lower()
    return "label" in lowered or "target" in lowered


def _target_looks_like_label(node: ast.expr) -> bool:
    for token in _iter_target_tokens(node):
        if _looks_like_label_token(token):
            return True
    return False


def _iter_target_tokens(node: ast.expr) -> Iterable[str]:
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.Attribute):
        yield node.attr
        yield from _iter_target_tokens(node.value)
    elif isinstance(node, ast.Subscript):
        yield from _iter_target_tokens(node.value)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            yield from _iter_target_tokens(element)


def _has_keyword(node: ast.Call, keyword: str) -> bool:
    for kw in node.keywords:
        if kw.arg == keyword:
            return True
    return False


def _center_is_true(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg == "center":
            value = _literal_bool(kw.value)
            if value is True:
                return True
    return False


def _literal_bool(node: ast.AST) -> Optional[bool]:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _extract_shift_amount(node: ast.Call) -> Optional[int]:
    if node.args:
        value = _literal_int(node.args[0])
        if value is not None:
            return value

    for kw in node.keywords:
        if kw.arg == "periods":
            value = _literal_int(kw.value)
            if value is not None:
                return value
    return None


def _literal_int(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        if isinstance(operand, ast.Constant) and isinstance(operand.value, int):
            return -operand.value
    return None


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
