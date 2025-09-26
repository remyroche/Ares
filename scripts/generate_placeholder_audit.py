#!/usr/bin/env python3
"""Generate an audit of placeholder constructs across the repository."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

PASS_PATTERN = re.compile(r"^\s*pass\b")
INLINE_SILENT_FAILURE_PATTERN = re.compile(r"^\s*except[^:]*:\s*pass\b")
BLOCK_EXCEPT_PATTERN = re.compile(r"^\s*except[^:]*:\s*$")
TODO_PATTERN = re.compile(r"\bTODO\b", re.IGNORECASE)
FIXME_PATTERN = re.compile(r"\bFIXME\b", re.IGNORECASE)
PLACEHOLDER_PATTERN = re.compile(r"placeholder", re.IGNORECASE)
MOCK_PATTERN = re.compile(r"\bmock\b", re.IGNORECASE)
STUB_PATTERN = re.compile(r"\bstub(s)?\b", re.IGNORECASE)

SKIP_DIRECTORIES = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "build",
    "dist",
    "audits",
}

SOURCE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".pyx",
    ".pxd",
    ".pxi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".rs",
    ".go",
    ".java",
    ".scala",
    ".kt",
    ".kts",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".php",
    ".rb",
    ".swift",
    ".m",
    ".mm",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",
    ".cs",
}

ALWAYS_INCLUDE_FILENAMES = {"Makefile"}

CATEGORY_ORDER = [
    "pass",
    "silent_failure",
    "todo",
    "fixme",
    "placeholder",
    "mock",
    "stub",
]

MAX_EXAMPLES_PER_FILE = 5


@dataclass
class Occurrence:
    path: Path
    line: int
    content: str

    def as_dict(self) -> Dict[str, str | int]:
        return {"line": self.line, "content": self.content.strip()}


@dataclass
class FileAudit:
    counts: Counter
    examples: Dict[str, List[Occurrence]]


def should_scan(path: Path) -> bool:
    if path.name in ALWAYS_INCLUDE_FILENAMES:
        return True
    if path.suffix in SOURCE_EXTENSIONS:
        return True
    return False


def iter_files() -> Iterator[Path]:
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if not should_scan(path):
            continue
        yield path


def detect_block_silent_failures(lines: Sequence[str]) -> List[int]:
    indices: List[int] = []
    for idx, line in enumerate(lines):
        if not BLOCK_EXCEPT_PATTERN.match(line):
            continue
        indent = len(line) - len(line.lstrip())
        j = idx + 1
        while j < len(lines):
            candidate = lines[j]
            if not candidate.strip():
                j += 1
                continue
            follow_indent = len(candidate) - len(candidate.lstrip())
            if follow_indent <= indent:
                break
            if PASS_PATTERN.match(candidate):
                indices.append(idx)
            break
    return indices


def scan_file(path: Path) -> FileAudit | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    lines = text.splitlines()
    counts: Counter = Counter()
    examples: Dict[str, List[Occurrence]] = defaultdict(list)
    is_python_file = path.suffix == ".py"

    def add_example(category: str, line_no: int) -> None:
        if len(examples[category]) >= MAX_EXAMPLES_PER_FILE:
            return
        examples[category].append(
            Occurrence(path=path.relative_to(REPO_ROOT), line=line_no, content=lines[line_no - 1])
        )

    for idx, line in enumerate(lines, start=1):
        if is_python_file and PASS_PATTERN.match(line):
            counts["pass"] += 1
            add_example("pass", idx)
        if is_python_file and INLINE_SILENT_FAILURE_PATTERN.match(line):
            counts["silent_failure"] += 1
            add_example("silent_failure", idx)
        if TODO_PATTERN.search(line):
            counts["todo"] += 1
            add_example("todo", idx)
        if FIXME_PATTERN.search(line):
            counts["fixme"] += 1
            add_example("fixme", idx)
        if PLACEHOLDER_PATTERN.search(line):
            counts["placeholder"] += 1
            add_example("placeholder", idx)
        if MOCK_PATTERN.search(line):
            counts["mock"] += 1
            add_example("mock", idx)
        if STUB_PATTERN.search(line):
            counts["stub"] += 1
            add_example("stub", idx)

    if path.suffix == ".py":
        block_indices = detect_block_silent_failures(lines)
        if block_indices:
            counts["silent_failure"] += len(block_indices)
            for offset in block_indices:
                add_example("silent_failure", offset + 1)

    if not any(counts.values()):
        return None

    return FileAudit(counts=counts, examples=examples)


def build_summary(file_results: Dict[Path, FileAudit]) -> Dict[str, int]:
    totals: Counter = Counter()
    for audit in file_results.values():
        totals.update(audit.counts)
    return {category: int(totals.get(category, 0)) for category in CATEGORY_ORDER}


def build_markdown(file_results: Dict[Path, FileAudit]) -> str:
    summary = build_summary(file_results)
    lines: List[str] = ["# Placeholder and TODO Audit", ""]
    lines.append("This report summarizes pass statements, silent failures, and placeholder markers across the repository.")
    lines.append("")
    lines.append("## Summary Totals")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("| --- | ---: |")
    for category in CATEGORY_ORDER:
        lines.append(f"| {category.replace('_', ' ').title()} | {summary[category]} |")
    lines.append("")

    # Prepare per-category rankings
    for category in CATEGORY_ORDER:
        ranked = sorted(
            (
                (path, audit.counts.get(category, 0))
                for path, audit in file_results.items()
                if audit.counts.get(category, 0)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked:
            continue
        lines.append(f"## Top files by {category.replace('_', ' ')} occurrences")
        lines.append("")
        lines.append("| File | Count |")
        lines.append("| --- | ---: |")
        for path, count in ranked[:25]:
            lines.append(f"| `{path.as_posix()}` | {count} |")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Only the top 25 files per category are shown. See the accompanying JSON file for full details, including example lines._")
    lines.append("")
    return "\n".join(lines)


def build_json(file_results: Dict[Path, FileAudit]) -> Dict[str, object]:
    return {
        "summary": build_summary(file_results),
        "files": [
            {
                "path": path.as_posix(),
                "counts": {category: audit.counts.get(category, 0) for category in CATEGORY_ORDER},
                "examples": {
                    category: [occurrence.as_dict() for occurrence in audit.examples.get(category, [])]
                    for category in CATEGORY_ORDER
                    if audit.counts.get(category, 0)
                },
            }
            for path, audit in sorted(file_results.items(), key=lambda item: item[0].as_posix())
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("audits/placeholder_audit.md"),
        help="Path to write the Markdown summary report.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("audits/placeholder_audit.json"),
        help="Path to write the detailed JSON report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results: Dict[Path, FileAudit] = {}
    for path in iter_files():
        audit = scan_file(path)
        if audit is None:
            continue
        results[path.relative_to(REPO_ROOT)] = audit

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)

    args.markdown.write_text(build_markdown(results), encoding="utf-8")
    args.json.write_text(json.dumps(build_json(results), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
