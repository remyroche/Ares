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

# Import core utilities
try:
    from ...utils.tprint import tprint, tprint_debug, tprint_error, tprint_info, tprint_warning
    from ...utils.common_operations import (
        safe_file_exists, ensure_directory, safe_json_dump, safe_json_load,
        validate_positive, timed_operation, format_bytes
    )
    from ...utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    # Import matrix operations for batch processing
    from ...utils.matrix_operations import (
        batch_matrix_multiply, optimize_dataframe, get_batch_matrix_processor,
        matrix_correlation_analysis, get_vectorized_processing_core
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as e:
    # Fallback imports if utils are not available
    MATRIX_OPERATIONS_AVAILABLE = False
    def tprint(*args, **kwargs): pass
    def tprint_debug(*args, **kwargs): pass
    def tprint_error(*args, **kwargs): pass
    def tprint_info(*args, **kwargs): pass
    def tprint_warning(*args, **kwargs): pass
    def safe_file_exists(path): return Path(path).exists() if isinstance(path, (str, Path)) else False
    def ensure_directory(path): Path(path).mkdir(parents=True, exist_ok=True) if isinstance(path, (str, Path)) else None
    def safe_json_dump(data, file_path, **kwargs): None
    def safe_json_load(file_path, default=None): return default
    def validate_positive(value, name="value"): return value if value > 0 else 0.0
    def timed_operation(func): return func
    def format_bytes(bytes_value): return f"{bytes_value}B"
    def get_m1_memory_optimizer(): return None
    # Matrix operations fallbacks
    def batch_matrix_multiply(*args, **kwargs): return None
    def optimize_dataframe(df): return df
    def get_batch_matrix_processor(): return None
    def matrix_correlation_analysis(*args, **kwargs): return {}
    def get_vectorized_processing_core(): return None

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


@timed_operation
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
    if not safe_file_exists(path):
        tprint_error(f"Cannot lint missing file: {path}")
        raise FileNotFoundError(f"Cannot lint missing file: {path}")

    try:
        source = path.read_text(encoding="utf-8")
        file_size = len(source.encode('utf-8'))
        tprint_debug(f"Linting file: {path} ({format_bytes(file_size)})")
    except Exception as e:
        tprint_error(f"Failed to read file {path}: {e}")
        raise TemporalLintError(f"Failed to read {path}: {e}") from e

    lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - treat invalid syntax as fatal
        tprint_error(f"Failed to parse {path}: {exc}")
        raise TemporalLintError(f"Failed to parse {path}: {exc}") from exc

    visitor = _TemporalLintVisitor(path=path, lines=lines)
    visitor.visit(tree)

    violations = [violation.format() for violation in visitor.violations]
    if violations:
        tprint_warning(f"Found {len(violations)} temporal leakage violations in {path}")
        for violation in violations:
            tprint_warning(f"  - {violation}")
    else:
        tprint_debug(f"No temporal leakage violations found in {path}")

    return violations


@timed_operation
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

    tprint_info("Starting temporal linting across codebase")
    memory_optimizer = get_m1_memory_optimizer()

    candidate_files = _collect_candidate_files(paths)
    tprint_info(f"Found {len(candidate_files)} candidate files for linting")

    # Initialize matrix operations for batch processing if available
    batch_processor = get_batch_matrix_processor() if MATRIX_OPERATIONS_AVAILABLE else None
    vectorized_core = get_vectorized_processing_core() if MATRIX_OPERATIONS_AVAILABLE else None

    if MATRIX_OPERATIONS_AVAILABLE:
        tprint_debug("Matrix operations available for batch processing")
    else:
        tprint_debug("Matrix operations not available, using standard processing")

    violations: Dict[str, List[str]] = {}

    # Process files in batches for better performance
    batch_size = 10  # Process 10 files at a time
    file_batches = [candidate_files[i:i + batch_size] for i in range(0, len(candidate_files), batch_size)]

    tprint_debug(f"Processing {len(candidate_files)} files in {len(file_batches)} batches")

    for batch_idx, file_batch in enumerate(file_batches):
        tprint_debug(f"Processing batch {batch_idx + 1}/{len(file_batches)} ({len(file_batch)} files)")

        for file_path in file_batch:
            try:
                file_violations = lint_for_temporal_leakage(file_path)
                # Optimize memory after processing large files
                if memory_optimizer and hasattr(memory_optimizer, "cleanup_memory"):
                    memory_optimizer.cleanup_memory()
            except TemporalLintError as exc:
                file_violations = [str(exc)]
                tprint_error(f"Temporal linting error in {file_path}: {exc}")
            violations[str(file_path)] = file_violations

        total_violations = sum(len(v) for v in violations.values())
        if violations:
            if raise_on_violation:
                tprint_error(f"Temporal leakage violations detected in {len(violations)} files ({total_violations} total)")
                message_lines = ["Temporal leakage violations detected:"]
                for path, file_violations in sorted(violations.items()):
                    message_lines.append(path)
                    message_lines.extend(f"  - {msg}" for msg in file_violations)
            raise TemporalLintError("\n".join(message_lines))
        else:
            tprint_warning(f"Found {total_violations} temporal leakage violations in {len(violations)} files")
    else:
        tprint_info("No temporal leakage violations detected")

    tprint_info(f"Temporal linting completed: {total_violations} violations in {len(violations)} files")

    # Use vectorized core for final optimization if available
    if vectorized_core and hasattr(vectorized_core, 'optimize_processing'):
        try:
            vectorized_core.optimize_processing()
            tprint_debug("Applied final processing optimization")
        except Exception as e:
            tprint_debug(f"Final optimization failed: {e}")

    return violations


@timed_operation
def analyze_temporal_patterns_batch(file_paths: List[Path | str]) -> Dict[str, Any]:
    """Analyze temporal leakage patterns across multiple files using batch processing.

    Args:
        file_paths: List of file paths to analyze

    Returns:
        Dictionary containing aggregated temporal pattern analysis
    """
    tprint_info(f"Starting batch temporal pattern analysis for {len(file_paths)} files")

    if not MATRIX_OPERATIONS_AVAILABLE:
        tprint_warning("Matrix operations not available for batch analysis")
        return {}

    batch_processor = get_batch_matrix_processor()
    if not batch_processor:
        tprint_warning("Batch matrix processor not available")
        return {}

    # Collect violation data for correlation analysis
    violation_matrices = []
    file_sizes = []
    violation_counts = []

    for file_path in file_paths:
        try:
            path = Path(file_path)
            if not path.exists():
                continue

            # Get file size and basic metrics
            file_size = path.stat().st_size
            file_sizes.append(file_size)

            # Lint file and collect violations
            violations = lint_for_temporal_leakage(path)
            violation_counts.append(len(violations))

            # Create violation matrix (binary indicators for different violation types)
            violation_types = set()
            for violation in violations:
                # Extract violation type from message (simplified)
                if 'center=True' in violation:
                    violation_types.add('center_rolling')
                elif 'shift(-' in violation:
                    violation_types.add('negative_shift')
                elif 'closed=' in violation:
                    violation_types.add('missing_closed')
                else:
                    violation_types.add('other')

            # Create binary vector for this file
            violation_vector = [1 if vt in violation_types else 0 for vt in ['center_rolling', 'negative_shift', 'missing_closed', 'other']]
            violation_matrices.append(violation_vector)

        except Exception as e:
            tprint_debug(f"Batch analysis failed for {file_path}: {e}")
            continue

    if not violation_matrices:
        tprint_warning("No valid files for batch analysis")
        return {}

    try:
        # Use matrix operations for correlation analysis
        import numpy as np
        violation_array = np.array(violation_matrices)

        # Compute correlation matrix between violation types
        if violation_array.shape[0] > 1 and violation_array.shape[1] > 1:
            corr_matrix = safe_correlation_matrix(pd.DataFrame(violation_array))
            if corr_matrix is not None:
                correlation_analysis = matrix_correlation_analysis(
                    violation_array, method='correlation'
                )
            else:
                correlation_analysis = {}
        else:
            correlation_analysis = {}

        results = {
            'total_files_analyzed': len(violation_matrices),
            'violation_type_distribution': {
                'center_rolling': sum(v[0] for v in violation_matrices),
                'negative_shift': sum(v[1] for v in violation_matrices),
                'missing_closed': sum(v[2] for v in violation_matrices),
                'other': sum(v[3] for v in violation_matrices)
            },
            'average_violations_per_file': np.mean(violation_counts) if violation_counts else 0,
            'total_file_size_bytes': sum(file_sizes),
            'correlation_analysis': correlation_analysis,
            'violation_matrix_shape': violation_array.shape if 'violation_array' in locals() else None
        }

        tprint_info(f"Batch analysis completed: {results['total_files_analyzed']} files, "
                   f"{sum(results['violation_type_distribution'].values())} total violations")
        return results

    except Exception as e:
        tprint_error(f"Batch temporal analysis aggregation failed: {e}")
        return {}


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
