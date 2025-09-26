from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import re

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover - import guard
    from src.utils.tprint import tprint, tprint_error, tprint_warning
except Exception as exc:  # pragma: no cover - script should fast fail if logging is unavailable
    raise RuntimeError(
        "Unable to import tprint utilities. Ensure repository dependencies are available."
    ) from exc


PASS_PATTERN = re.compile(r"^\s*pass\s*(#.*)?$")
EXCEPT_PATTERN = re.compile(r"^\s*except\b.*:\s*(#.*)?$")

UNIMPLEMENTED_PATTERNS: Sequence[tuple[str, re.Pattern[str]]] = (
    ("TODO", re.compile(r"TODO", re.IGNORECASE)),
    ("FIXME", re.compile(r"FIXME", re.IGNORECASE)),
    ("NotImplementedError", re.compile(r"NotImplementedError")),
    ("NotImplemented", re.compile(r"\bNotImplemented\b")),
    ("Stub", re.compile(r"\bstub\b", re.IGNORECASE)),
    ("Mock", re.compile(r"\bmock\b", re.IGNORECASE)),
    ("XX", re.compile(r"\bxx\b", re.IGNORECASE)),
)

SKIP_DIRECTORIES = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "build",
    "dist",
}


@dataclass(slots=True)
class AuditIssue:
    """Represents a single violation detected by the audit."""

    category: str
    path: Path
    line: int
    content: str
    note: str | None = None

    def log(self) -> None:
        message = f"[{self.category}] {self.path}:{self.line} -> {self.content.strip()}"
        if self.category == "Pass Statement":
            tprint_error(message)
            if self.note:
                tprint_error(f"   ↳ {self.note}")
        else:
            tprint_warning(message)
            if self.note:
                tprint(f"   ↳ {self.note}")


def iter_files() -> Iterator[Path]:
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        yield path


def read_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise RuntimeError(f"Failed to read file: {path}") from exc


def collect_pass_statements() -> List[AuditIssue]:
    issues: List[AuditIssue] = []
    for path in iter_files():
        if path.suffix != ".py":
            continue
        lines = read_lines(path)
        for idx, line in enumerate(lines, start=1):
            if PASS_PATTERN.match(line):
                issues.append(
                    AuditIssue(
                        category="Pass Statement",
                        path=path.relative_to(REPO_ROOT),
                        line=idx,
                        content=line,
                        note="Replace pass with production-ready logic or remove the block.",
                    )
                )
    issues.sort(key=lambda issue: (issue.path.as_posix(), issue.line))
    return issues


def collect_silent_failures() -> List[AuditIssue]:
    issues: List[AuditIssue] = []
    for path in iter_files():
        if path.suffix != ".py":
            continue
        lines = read_lines(path)
        for idx, line in enumerate(lines, start=1):
            if not EXCEPT_PATTERN.match(line):
                continue
            indent = len(line) - len(line.lstrip())
            for follow_idx in range(idx + 1, len(lines) + 1):
                follow_line = lines[follow_idx - 1]
                if not follow_line.strip():
                    continue
                follow_indent = len(follow_line) - len(follow_line.lstrip())
                if follow_indent <= indent:
                    break
                if PASS_PATTERN.match(follow_line):
                    issues.append(
                        AuditIssue(
                            category="Silent Failure",
                            path=path.relative_to(REPO_ROOT),
                            line=idx,
                            content=line,
                            note="Handle the exception explicitly or remove the handler.",
                        )
                    )
                break
    issues.sort(key=lambda issue: (issue.path.as_posix(), issue.line))
    return issues


def collect_unimplemented_markers() -> List[AuditIssue]:
    issues: List[AuditIssue] = []
    for path in iter_files():
        if path.suffix != ".py":
            continue
        lines = read_lines(path)
        for idx, line in enumerate(lines, start=1):
            for label, pattern in UNIMPLEMENTED_PATTERNS:
                if pattern.search(line):
                    issues.append(
                        AuditIssue(
                            category=f"Unimplemented ({label})",
                            path=path.relative_to(REPO_ROOT),
                            line=idx,
                            content=line,
                            note="Implement the missing functionality and remove the placeholder.",
                        )
                    )
                    break
    issues.sort(key=lambda issue: (issue.path.as_posix(), issue.line))
    return issues


def build_markdown_report(sections: Dict[str, Sequence[AuditIssue]]) -> str:
    lines: List[str] = [
        "# Pass Statements, Silent Failures, and Unimplemented Code Audit",
        "",
        "The audit fast-fails when mock placeholders, pass statements, or TODO/FIXME markers are detected.",
        "",
    ]
    for title, issues in sections.items():
        lines.append(f"## {title} ({len(issues)})")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.path}:{issue.line}` | {issue.content.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def summarize_counts(sections: Dict[str, Sequence[AuditIssue]]) -> str:
    summary_parts = [f"{title}={len(issues)}" for title, issues in sections.items()]
    return ", ".join(summary_parts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the repository for placeholder code.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write a markdown report instead of the legacy docs file.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=100,
        help="Maximum number of issues to log to the console (default: 100).",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Allow the script to exit successfully even if issues are detected.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    sections: Dict[str, List[AuditIssue]] = {
        "Pass Statements": collect_pass_statements(),
        "Silent Failures": collect_silent_failures(),
        "Unimplemented Markers": collect_unimplemented_markers(),
    }

    total_issues = sum(len(issues) for issues in sections.values())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(build_markdown_report(sections), encoding="utf-8")

    if total_issues == 0:
        tprint("✅ No placeholder code detected.")
        return

    tprint_warning("Placeholder code detected — review required.")

    max_issues = max(args.max_issues, 1)
    logged = 0
    for title, issues in sections.items():
        if not issues:
            continue
        tprint_warning(f"{title}: {len(issues)} issues found")
        for issue in issues:
            if logged >= max_issues:
                remaining = total_issues - logged
                if remaining > 0:
                    tprint_warning(f"… {remaining} additional issues suppressed (use --max-issues to show more)")
                break
            issue.log()
            logged += 1
        if logged >= max_issues:
            break

    summary = summarize_counts(sections)
    if args.no_fail:
        tprint_warning(f"Audit completed with {total_issues} issues (no-fail mode). Summary: {summary}")
    else:
        raise SystemExit(f"Audit failed: {total_issues} issues detected. Summary: {summary}")


if __name__ == "__main__":
    main()
