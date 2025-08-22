#!/usr/bin/env python3
"""
Universal Repository Checker for Ares

Runs a suite of checks:
- Syntax/indentation scan (reusing SyntaxErrorScanner)
- Ruff lint (configured via pyproject.toml)
- Ruff format check (no changes applied, just report)
- Mypy type checks (if available)
- Data file validation (JSON/CSV) in selected directories
- Decorator presence check for functions in selected paths
- Basic logging presence check per module

Outputs human-readable and JSON reports in reports/.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import logging
import os
import re
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
REPORTS_DIR_DEFAULT = REPO_ROOT / "reports"


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    summary: str
    details: Dict[str, object]


def _ensure_reports_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_syntax_scanner_class():
    """Load SyntaxErrorScanner from scripts/syntax_error_scanner.py without a package."""
    import importlib.util

    target = SCRIPTS_DIR / "syntax_error_scanner.py"
    spec = importlib.util.spec_from_file_location("syntax_error_scanner", target)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to load syntax_error_scanner module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SyntaxErrorScanner


# --------------------------------------------------------------------------------------
# Syntax/indentation check
# --------------------------------------------------------------------------------------
def run_syntax_check(code_roots: Sequence[Path], jobs: int, reports_dir: Path) -> CheckResult:
    SyntaxErrorScanner = _load_syntax_scanner_class()
    scanner = SyntaxErrorScanner()
    aggregate = {"files_processed": 0, "files_with_errors": 0, "total_errors": 0, "error_types": {}}

    for root in code_roots:
        res = scanner.scan_directory(root, jobs=jobs)
        for k in ("files_processed", "files_with_errors", "total_errors"):
            aggregate[k] = aggregate.get(k, 0) + int(res.get(k, 0))
        # Merge error types
        types = res.get("error_types", {}) or {}
        for t, c in types.items():
            aggregate.setdefault("error_types", {})
            aggregate["error_types"][t] = aggregate["error_types"].get(t, 0) + int(c)

    # Write a dedicated syntax report
    syntax_txt = reports_dir / "syntax_error_report.txt"
    syntax_json = reports_dir / "syntax_error_report.json"
    scanner.generate_report(syntax_txt)
    # Build JSON payload
    syntax_payload = {
        "summary": {
            "files_processed": scanner.files_processed,
            "files_with_errors": len(scanner.error_files),
            "total_errors": scanner.total_errors,
            "error_types": dict(scanner.error_types),
        },
        "files": {
            path: {
                "error_count": len(errs),
                "errors": errs,
            }
            for path, errs in sorted(scanner.error_files.items(), key=lambda kv: len(kv[1]), reverse=True)
        },
    }
    syntax_json.write_text(json.dumps(syntax_payload, indent=2), encoding="utf-8")

    passed = aggregate["files_with_errors"] == 0
    summary = f"processed={scanner.files_processed}, files_with_errors={len(scanner.error_files)}, total_errors={scanner.total_errors}"
    return CheckResult("syntax", passed, summary, syntax_payload)


# --------------------------------------------------------------------------------------
# Ruff checks (lint + format check)
# --------------------------------------------------------------------------------------
def _run_cmd(cmd: List[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def run_ruff_checks(code_roots: Sequence[Path], reports_dir: Path) -> CheckResult:
    # Ruff reads config from pyproject.toml at repo root
    targets = [str(p) for p in code_roots]

    # Lint
    lint_cmd = [sys.executable, "-m", "ruff", "check", "--output-format", "json", *targets]
    lint_rc, lint_out, lint_err = _run_cmd(lint_cmd, cwd=REPO_ROOT)
    try:
        lint_json = json.loads(lint_out or "[]")
    except json.JSONDecodeError:
        lint_json = []

    (reports_dir / "ruff_lint.json").write_text(json.dumps(lint_json, indent=2), encoding="utf-8")

    # Format check (no write)
    fmt_cmd = [sys.executable, "-m", "ruff", "format", "--check", "--diff", *targets]
    fmt_rc, fmt_out, fmt_err = _run_cmd(fmt_cmd, cwd=REPO_ROOT)
    (reports_dir / "ruff_format.txt").write_text(fmt_out or fmt_err or "", encoding="utf-8")

    passed = lint_rc == 0 and fmt_rc == 0
    summary = f"lint_issues={len(lint_json)}, needs_format={(fmt_rc != 0)}"
    return CheckResult("ruff", passed, summary, {"lint": lint_json, "format": fmt_out})


# --------------------------------------------------------------------------------------
# Mypy type checking
# --------------------------------------------------------------------------------------
def run_mypy(code_roots: Sequence[Path], reports_dir: Path) -> CheckResult:
    targets = [str(p) for p in code_roots]
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--hide-error-context",
        "--no-color-output",
        "--no-error-summary",
        "--error-format=json",
        *targets,
    ]
    rc, out, err = _run_cmd(cmd, cwd=REPO_ROOT)

    try:
        payload = json.loads(out or "[]")
    except json.JSONDecodeError:
        payload = []

    (reports_dir / "mypy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Determine pass/fail: rc==0 means no type errors
    passed = rc == 0
    summary = f"type_errors={sum(1 for item in payload if item.get('type') == 'error')}"
    return CheckResult("mypy", passed, summary, {"results": payload, "return_code": rc})


# --------------------------------------------------------------------------------------
# Data file validation (JSON/CSV)
# --------------------------------------------------------------------------------------
@dataclass
class DataIssue:
    path: str
    issue: str


def _iter_data_files(dirs: Sequence[Path], max_size_mb: float = 20.0) -> Iterable[Path]:
    max_bytes = int(max_size_mb * 1024 * 1024)
    exts = {".json", ".csv"}
    for base in dirs:
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "node_modules"}]
            for name in filenames:
                p = Path(dirpath) / name
                if p.suffix.lower() in exts:
                    try:
                        if p.stat().st_size <= max_bytes:
                            yield p
                    except FileNotFoundError:  # pragma: no cover
                        continue


def _validate_json_file(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"read_error: {exc}"
    try:
        json.loads(text)
        return None
    except json.JSONDecodeError as exc:
        return f"json_decode_error: {exc}"


def _validate_csv_file(path: Path, sample_rows: int = 1000) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            expected_cols: Optional[int] = None
            count = 0
            for row in reader:
                if expected_cols is None:
                    expected_cols = len(row)
                elif len(row) != expected_cols:
                    return f"inconsistent_column_count: expected={expected_cols}, got={len(row)} at row={count}"
                count += 1
                if count >= sample_rows:
                    break
        return None
    except Exception as exc:
        return f"csv_read_error: {exc}"


def run_data_validation(data_dirs: Sequence[Path], reports_dir: Path, jobs: int) -> CheckResult:
    files = list(_iter_data_files(data_dirs))
    issues: List[DataIssue] = []

    def validate(path: Path) -> Optional[DataIssue]:
        if path.suffix.lower() == ".json":
            res = _validate_json_file(path)
        else:
            res = _validate_csv_file(path)
        if res:
            return DataIssue(str(path), res)
        return None

    if jobs <= 1:
        for p in files:
            maybe = validate(p)
            if maybe:
                issues.append(maybe)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {ex.submit(validate, p): p for p in files}
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    issues.append(r)

    payload = {"files_checked": len(files), "issues": [asdict(x) for x in issues]}
    (reports_dir / "data_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    passed = len(issues) == 0
    summary = f"files_checked={len(files)}, issues={len(issues)}"
    return CheckResult("data_validation", passed, summary, payload)


# --------------------------------------------------------------------------------------
# Decorator and logging presence checks
# --------------------------------------------------------------------------------------
DEFAULT_DECORATOR_ALLOWLIST = {
    "validate_file_operation",
    "validate_dataframe_operation",
    "validate_step_operation",
    "validate_step1_operation",
    "validate_step1_5_operation",
    "validate_step2_operation",
    "validate_step4_operation",
    "log_call",
    "retry",
    "handle_errors",
}


@dataclass
class FunctionDecoratorIssue:
    file: str
    function: str
    decorators_found: List[str]


@dataclass
class LoggingIssue:
    file: str
    issue: str


def _iter_py_files(paths: Sequence[Path]) -> Iterable[Path]:
    for base in paths:
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "node_modules", "mlruns"}]
            for name in filenames:
                if name.endswith(".py"):
                    yield Path(dirpath) / name


def _get_decorator_names(node: ast.AST) -> List[str]:
    names: List[str] = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for d in node.decorator_list:
            if isinstance(d, ast.Name):
                names.append(d.id)
            elif isinstance(d, ast.Attribute):
                names.append(d.attr)
            elif isinstance(d, ast.Call):
                # Decorator used with args, e.g., @validate_file_operation("step")
                if isinstance(d.func, ast.Name):
                    names.append(d.func.id)
                elif isinstance(d.func, ast.Attribute):
                    names.append(d.func.attr)
    return names


def run_decorator_and_logging_checks(
    code_roots: Sequence[Path],
    enforce_globs: Sequence[str],
    decorator_allowlist: Sequence[str],
    reports_dir: Path,
) -> CheckResult:
    import fnmatch

    py_files = list(_iter_py_files(code_roots))
    decorator_issues: List[FunctionDecoratorIssue] = []
    logging_issues: List[LoggingIssue] = []

    for file_path in py_files:
        rel = str(file_path.relative_to(REPO_ROOT))
        enforce = any(fnmatch.fnmatch(rel, g) for g in enforce_globs)

        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            logging_issues.append(LoggingIssue(rel, f"read_error: {exc}"))
            continue

        # Logging check
        has_logging_import = bool(re.search(r"(^|\n)\s*(from\s+logging\s+import|import\s+logging)\b", text))
        has_logger_usage = "logger." in text or "logging.getLogger(" in text
        if not (has_logging_import or has_logger_usage):
            # Only flag if within enforce_globs to avoid noise
            if enforce:
                logging_issues.append(LoggingIssue(rel, "no_logging_detected"))

        # Decorator check via AST
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # Syntax issues are handled by syntax scanner
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if enforce:
                    decos = _get_decorator_names(node)
                    if not any(d in decorator_allowlist for d in decos):
                        decorator_issues.append(
                            FunctionDecoratorIssue(
                                file=rel,
                                function=node.name,
                                decorators_found=decos,
                            )
                        )

    payload = {
        "decorator_issues": [asdict(x) for x in decorator_issues],
        "logging_issues": [asdict(x) for x in logging_issues],
    }
    (reports_dir / "decorators_logging.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    passed = len(payload["decorator_issues"]) == 0 and len(payload["logging_issues"]) == 0
    summary = f"decorator_issues={len(payload['decorator_issues'])}, logging_issues={len(payload['logging_issues'])}"
    return CheckResult("decorators_logging", passed, summary, payload)


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run universal repository checks and produce reports.")
    p.add_argument("--code-dirs", nargs="*", default=["src", "scripts"], help="Directories to scan for code checks")
    p.add_argument("--data-dirs", nargs="*", default=["data", "results", "reports"], help="Directories to scan for data validation")
    p.add_argument("--reports-dir", default=str(REPORTS_DIR_DEFAULT), help="Directory to write reports")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="Parallelism for checks")
    p.add_argument("--skip-ruff", action="store_true", help="Skip ruff checks")
    p.add_argument("--skip-mypy", action="store_true", help="Skip mypy type checks")
    p.add_argument("--skip-data", action="store_true", help="Skip data file validation")
    p.add_argument(
        "--decorator-enforce-glob",
        action="append",
        default=["src/training/steps/*.py"],
        help="Glob for files where a decorator is required (repeatable)",
    )
    p.add_argument(
        "--decorator-allow",
        action="append",
        default=sorted(DEFAULT_DECORATOR_ALLOWLIST),
        help="Decorator names that satisfy the requirement (repeatable)",
    )
    p.add_argument("--fail-on-error", action="store_true", help="Exit with non-zero status if any check fails")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    reports_dir = Path(args.reports_dir).resolve()
    _ensure_reports_dir(reports_dir)

    code_roots = [REPO_ROOT / d for d in args.code_dirs]
    data_roots = [REPO_ROOT / d for d in args.data_dirs]

    results: List[CheckResult] = []

    # Run checks
    logger.info("Running syntax check...")
    results.append(run_syntax_check(code_roots, jobs=int(args.jobs), reports_dir=reports_dir))

    if not args.skip_ruff:
        logger.info("Running ruff checks...")
        results.append(run_ruff_checks(code_roots, reports_dir))

    if not args.skip_mypy:
        logger.info("Running mypy type checks...")
        try:
            results.append(run_mypy(code_roots, reports_dir))
        except FileNotFoundError:
            logger.warning("mypy not available; skipping type checks")

    if not args.skip_data:
        logger.info("Running data validation...")
        results.append(run_data_validation(data_roots, reports_dir, jobs=int(args.jobs)))

    logger.info("Running decorator/logging presence checks...")
    results.append(
        run_decorator_and_logging_checks(
            code_roots,
            enforce_globs=args.decorator_enforce_glob,
            decorator_allowlist=args.decorator_allow,
            reports_dir=reports_dir,
        )
    )

    # Aggregate
    overall_passed = all(r.passed for r in results)
    overall = {
        "overall_passed": overall_passed,
        "checks": [asdict(r) for r in results],
    }

    # Write combined reports
    (reports_dir / "universal_check_report.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    text_lines: List[str] = []
    text_lines.append("=" * 80)
    text_lines.append("UNIVERSAL CHECK REPORT")
    text_lines.append("=" * 80)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        text_lines.append(f"- {r.name}: {status} ({r.summary})")
    (reports_dir / "universal_check_report.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    logger.info("Universal check completed. Reports at %s", reports_dir)
    if args.fail_on_error and not overall_passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())