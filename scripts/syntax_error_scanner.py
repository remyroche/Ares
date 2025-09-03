#!/usr/bin/env python3
"""
Syntax Error Scanner for Ares Repository

This script scans the repository for Python syntax and indentation errors using
`py_compile`. It reports per-file error counts, error-type breakdowns, and a
comprehensive human-readable report, with optional JSON output.

Key features:
- Fast concurrent scanning with a thread pool
- Robust exclusion of common build/cache/test directories
- CLI options for root path, exclusions, jobs, timeouts, output paths
- JSON summary output and non-zero exit on errors (optional)
- Clear, readable text report with summaries and per-file details
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# --------------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger=logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Scanner implementation
# --------------------------------------------------------------------------------------
class SyntaxErrorScanner:
    """Comprehensive syntax error scanner using `python -m py_compile`.

    The scanner walks a root directory, discovers Python files, and compiles each
    in a separate Python subprocess to detect syntax and indentation errors.
    """

    DEFAULT_EXCLUDED_DIRS: tuple[str, ...] = (
        ".git",
        "__pycache__",
        "node_modules",
        "venv",
        "env",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        "build",
        "dist",
        ".eggs",
        "backup_",
        "mlruns",
        "results",
        "test_results",
        "data",
        "data_cache",
        "log",
        "logs",
    )

    def __init__(
        self,
        *,
        timeout_seconds: float=10.0,
        excluded_dirs: Sequence[str] | None = None,
        excluded_globs: Sequence[str] | None = None,
        file_extensions: Sequence[str] = (".py",),
    ) -> None:
        self.timeout_seconds=timeout_seconds
        self.excluded_dirs = set(excluded_dirs or ()) | set(self.DEFAULT_EXCLUDED_DIRS)
        self.excluded_globs=list(excluded_globs or [])
        self.file_extensions=tuple(file_extensions)

        # Aggregates
        self.error_files: dict[str, list[str]] = defaultdict(list)
        self.error_types: Counter[str] = Counter()
        self.total_errors: int=0
        self.files_processed: int = 0

    # ----------------------------
    # File discovery and scanning
    # ----------------------------
    def _iter_python_files(self, root: Path) -> Iterable[Path]:
        for dirpath, dirnames, filenames in os.walk(root):
            # Remove excluded directories in-place for pruning
            dirnames[:] = [
                d
                for d in dirnames
                if d not in self.excluded_dirs and not any(d.startswith(p) for p in self._prefix_like_excludes())
            ]

            for filename in filenames:
                if not filename.endswith(self.file_extensions):
                    continue
                file_path=Path(dirpath) / filename
                if self._is_glob_excluded(file_path):
                    continue
                yield file_path

    def _prefix_like_excludes(self) -> Iterable[str]:
        # Support prefix-like excludes such as "backup_" to exclude "backup_*" folders.
        return [name for name in self.excluded_dirs if name.endswith("_")]

    def _is_glob_excluded(self, file_path: Path) -> bool:
        if not self.excluded_globs:
            return False
        as_posix=file_path.as_posix()
        for pattern in self.excluded_globs:
            try:
                if Path(as_posix).match(pattern) or file_path.match(pattern):
                    return True
            except Exception:
                # Ignore bad patterns; treat as not excluded
                continue
        return False

    def scan_file(self, file_path: Path) -> list[str]:
        """Compile a single file, returning a list of error lines from stderr.

        Each returned string is a raw line of stderr that includes an error
        classification (e.g., SyntaxError, IndentationError, TabError).
        """
        try:
            result=subprocess.run(
                [sys.executable, "-m", "py_compile", str(file_path)],
                check=False, capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return ["TimeoutError: File took too long to compile"]
        except Exception as exc:  # pragma: no cover - defensive
            return [f"ScanError: {exc}"]

        if result.returncode== 0:
            return []

        stderr = result.stderr or ""
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        # Keep only meaningful error-class lines when possible; if none, keep all
        filtered: list[str] = [
            ln
            for ln in lines
            if (
                "SyntaxError" in ln
                or "IndentationError" in ln
                or "TabError" in ln
                or "ImportError" in ln
                or ln.startswith('File "')
            )
        ]
        return filtered if filtered else lines

    def parse_error_line(self, error_line: str) -> tuple[str, str, str, int | None]:
        """Parse an error line into (error_type, message, file, line_no).

        Attempts to extract file and line number when the line looks like a
        traceback 'File "...", line N' entry. Otherwise, uses best-effort
        parsing for error type and message.
        """
        error_type="UnknownError"
        if "SyntaxError:" in error_line:
            error_type = "SyntaxError"
        elif "IndentationError:" in error_line:
            error_type = "IndentationError"
        elif "TabError:" in error_line:
            error_type = "TabError"
        elif "ImportError:" in error_line:
            error_type = "ImportError"

        message: str = error_line
        if ":" in error_line:
            parts = error_line.split(":", 1)
            if len(parts) == 2:
                message=parts[1].strip() or error_line

        file_info="Unknown file"
        line_no: int | None = None
        m = re.search(r'File "([^"]+)", line (\d+)', error_line)
        if m:
            file_info=m.group(1)
            try:
                line_no=int(m.group(2))
            except Exception:
                line_no=None
        else:
            fm = re.search(r'File "([^"]+)"', error_line)
            if fm:
                file_info=fm.group(1)

        return error_type, message, file_info, line_no

    def scan_directory(self, directory: Path, *, jobs: int=1) -> dict[str, object]:
        logger.info(f"🔍 Scanning directory: {directory}")
        python_files=list(self._iter_python_files(directory))
        logger.info(f"📁 Found {len(python_files)} Python files")

        if not python_files:
            return {
                "files_processed": 0,
                "files_with_errors": 0,
                "total_errors": 0,
                "error_types": {},
            }

        # Concurrency is helpful because each scan invokes a subprocess
        if jobs <= 1:
            for file_path in python_files:
                self.files_processed += 1
                errors=self.scan_file(file_path)
                if errors:
                    self.error_files[str(file_path)] = errors
                    self.total_errors += len(errors)
                    for err in errors:
                        etype, _, _, _=self.parse_error_line(err)
                        self.error_types[etype] += 1
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=jobs) as executor:
                future_to_file={executor.submit(self.scan_file, fp): fp for fp in python_files}
                for future in as_completed(future_to_file):
                    file_path=future_to_file[future]
                    self.files_processed += 1
                    try:
                        errors = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        errors=[f"ScanError: {exc}"]

                    if errors:
                        self.error_files[str(file_path)] = errors
                        self.total_errors += len(errors)
                        for err in errors:
                            etype, _, _, _=self.parse_error_line(err)
                            self.error_types[etype] += 1

        return {
            "files_processed": self.files_processed,
            "files_with_errors": len(self.error_files),
            "total_errors": self.total_errors,
            "error_types": dict(self.error_types),
        }

    # ----------------------------
    # Reporting helpers
    # ----------------------------
    def generate_report(self, output_file: Path | None = None) -> str:
        report_lines: list[str] = []
        report_lines.append("=" * 80)
        report_lines.append("SYNTAX ERROR SCAN REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary
        report_lines.append("📊 SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Files processed: {self.files_processed}")
        report_lines.append(f"Files with errors: {len(self.error_files)}")
        report_lines.append(f"Total errors: {self.total_errors}")
        report_lines.append("")

        # Error types
        report_lines.append("🔍 ERROR TYPES BREAKDOWN")
        report_lines.append("-" * 40)
        for etype, count in self.error_types.most_common():
            pct=(count / self.total_errors * 100.0) if self.total_errors else 0.0
            report_lines.append(f"{etype}: {count} ({pct:.1f}%)")
        if not self.error_types:
            report_lines.append("No errors found.")
        report_lines.append("")

        # Files with errors
        report_lines.append("📁 FILES WITH ERRORS")
        report_lines.append("-" * 40)
        sorted_files=sorted(self.error_files.items(), key=lambda x: len(x[1]), reverse=True)
        for file_path, errors in sorted_files:
            abs_path=os.path.abspath(file_path)
            rel_path=os.path.relpath(file_path, ".")
            report_lines.append(f"\n{rel_path} ({len(errors)} errors):")
            report_lines.append(f"   Location: {abs_path}")

            # Per-file error type breakdown
            file_error_types: Counter[str] = Counter()
            for e in errors:
                etype, _, _, _=self.parse_error_line(e)
                file_error_types[etype] += 1
            for etype, count in file_error_types.most_common():
                report_lines.append(f"  - {etype}: {count}")

            # Show first few errors
            for i, err in enumerate(errors[:3]):
                etype, msg, _, line_no=self.parse_error_line(err)
                loc=f" at line {line_no}" if line_no is not None else ""
                report_lines.append(f"    {i + 1}. {etype}{loc}: {msg[:100]}...")
            if len(errors) > 3:
                report_lines.append(f"    ... and {len(errors) - 3} more errors")

        # Detailed breakdown
        report_lines.append("\n" + "=" * 80)
        report_lines.append("DETAILED ERROR BREAKDOWN")
        report_lines.append("=" * 80)
        for file_path, errors in sorted_files:
            rel=os.path.relpath(file_path, ".")
            abs_path=os.path.abspath(file_path)
            report_lines.append(f"\n{rel}:")
            report_lines.append(f"Location: {abs_path}")
            report_lines.append("-" * len(rel))
            for i, err in enumerate(errors, 1):
                report_lines.append(f"{i:3d}. {err}")

        report="\n".join(report_lines)
        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report, encoding="utf-8")
            logger.info(f"📄 Report written to: {output_file}")
        return report

    def get_files_by_error_count(self, min_errors: int=1) -> list[tuple[str, str, int]]:
        files: list[tuple[str, str, int]] = []
        for file_path, errors in self.error_files.items():
            if len(errors) >= min_errors:
                rel=os.path.relpath(file_path, ".")
                abs_path=os.path.abspath(file_path)
                files.append((rel, abs_path, len(errors)))
        return sorted(files, key=lambda x: x[2], reverse=True)

    def get_files_by_error_type(self, error_type: str) -> list[tuple[str, int]]:
        files: list[tuple[str, int]] = []
        for file_path, errors in self.error_files.items():
            type_count=sum(1 for e in errors if self.parse_error_line(e)[0] == error_type)
            if type_count > 0:
                rel=os.path.relpath(file_path, ".")
                files.append((rel, type_count))
        return sorted(files, key=lambda x: x[1], reverse=True)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Scan Python files for syntax errors and generate reports.")
    parser.add_argument("--root", type=str, default=".", help="Root directory to scan (default: .)")
    parser.add_argument("--output", type=str, default="reports/syntax_error_report.txt", help="Path to write text report")
    parser.add_argument("--json-output", type=str, default="reports/syntax_error_report.json", help="Optional JSON output path for summary + details")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="Number of concurrent jobs")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-file compile timeout in seconds")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory name to exclude (can be specified multiple times)",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Glob pattern to exclude files (can be specified multiple times)",
    )
    parser.add_argument("--top", type=int, default=10, help="Show top N files with most errors")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with non-zero code if any errors found")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logger.info("🚀 Starting syntax error scanner")
    args=_parse_args(argv)

    root=Path(args.root).resolve()
    output_path=Path(args.output).resolve() if args.output else None
    json_output_path=Path(args.json_output).resolve() if args.json_output else None

    scanner=SyntaxErrorScanner(
        timeout_seconds=float(args.timeout),
        excluded_dirs=args.exclude_dir,
        excluded_globs=args.exclude_glob,
    )

    results=scanner.scan_directory(root, jobs=max(1, int(args.jobs)))

    logger.info("📊 Scan Summary:")
    logger.info(f"   Files processed: {results['files_processed']}")
    logger.info(f"   Files with errors: {results['files_with_errors']}")
    logger.info(f"   Total errors: {results['total_errors']}")

    scanner.generate_report(output_path)

    # Top files with most errors
    print("\n" + "=" * 60)
    print("TOP FILES WITH MOST ERRORS")
    print("=" * 60)
    top_files=scanner.get_files_by_error_count(min_errors=1)[: int(args.top)]
    if top_files:
        for i, (rel_path, _abs_path, count) in enumerate(top_files, 1):
            print(f"{i:2d}. {rel_path} ({count} errors)")
    else:
        print("No files with errors.")

    # Files by error type
    print("\n" + "=" * 60)
    print("FILES BY ERROR TYPE")
    print("=" * 60)
    for error_type in ["SyntaxError", "IndentationError", "TabError"]:
        files=scanner.get_files_by_error_type(error_type)
        if files:
            print(f"\n{error_type} files:")
            for rel_path, count in files[:5]:
                print(f"  - {rel_path} ({count} errors)")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")

    # Optional JSON output
    if json_output_path is not None:
        json_payload={
            "summary": results,
            "files": {
                rel: {
                    "absolute_path": os.path.abspath(rel),
                    "error_count": len(errs),
                    "errors": errs,
                }
                for rel, errs in sorted(scanner.error_files.items(), key=lambda kv: len(kv[1]), reverse=True)
            },
        }
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        logger.info(f"📄 JSON report written to: {json_output_path}")

    logger.info("✅ Syntax error scanning completed!")
    if output_path:
        logger.info(f"📄 Detailed report saved to: {output_path}")

    if args.fail_on_error and results.get("files_with_errors", 0) > 0:
        return 2
    return 0


if __name__== "__main__":
    raise SystemExit(main())
