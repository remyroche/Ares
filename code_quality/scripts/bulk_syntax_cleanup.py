#!/usr/bin/env python3
"""
Bulk Syntax Cleanup

Purpose:
- Fix malformed docstring quote lines where more than three quotes were used
  on a single line that serves as a docstring delimiter, reducing them to
  exactly three quotes.
- Move accidental top-level import statements that were inserted inside
  multi-line "from ... import ( ... )" blocks back to just before the block.

This script is intentionally conservative and only applies localized line edits
without reformatting or re-indenting unrelated code.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple


def fix_docstring_quote_lines(content: str) -> Tuple[str, bool]:
    """Compress 4+ consecutive quotes on a line down to 3.

    Only affects lines that contain ONLY quotes (optionally with indentation),
    which is the typical pattern for opening/closing docstrings.
    """
    changed = False

    # Handle double quotes
    def _compress_double_quotes(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        indent = match.group(1) or ""
        return f"{indent}\"\"\""

    # Handle single quotes
    def _compress_single_quotes(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        indent = match.group(1) or ""
        return f"{indent}'''"

    new_content = re.sub(r"(?m)^(\s*)\"{4,}\s*$", _compress_double_quotes, content)
    new_content = re.sub(r"(?m)^(\s*)'{4,}\s*$", _compress_single_quotes, new_content)

    return new_content, changed


def _extract_and_rewrite_from_block(block_body: str) -> Tuple[str, List[str]]:
    """Given the inner body of a multi-line from-import block, remove any
    accidental top-level import statements found within and return the cleaned
    body plus a list of those import statements to hoist above the block.
    """
    lines = block_body.splitlines()
    hoisted: List[str] = []
    kept: List[str] = []

    for line in lines:
        stripped = line.strip()
        # Only hoist true top-level import statements; keep symbols in the import list
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Preserve original line exactly as written
            hoisted.append(stripped)
        else:
            kept.append(line)

    return "\n".join(kept), hoisted


def fix_imports_inside_from_blocks(content: str) -> Tuple[str, bool]:
    """Move `import ...` lines found inside a `from ... import ( ... )` block
    to just before the block.
    """
    changed = False

    pattern = re.compile(
        r"(?ms)"  # multiline + dotall
        r"(?P<head>^\s*from\s+[^\n]+?\s+import\s*\(\s*\n)"  # from ... import (\n
        r"(?P<body>.*?)(?P<tail>\n\))"  # body ... \n)
    )

    def _repl(match: re.Match[str]) -> str:
        nonlocal changed
        head = match.group("head")
        body = match.group("body")
        tail = match.group("tail")

        cleaned_body, hoisted = _extract_and_rewrite_from_block(body)

        if not hoisted:
            return match.group(0)

        changed = True

        # Compose hoisted imports. Ensure they are placed immediately before the block.
        hoisted_text = "\n".join(hoisted) + "\n"

        # Avoid producing multiple consecutive blank lines when body becomes empty
        if cleaned_body.strip():
            new_block = f"{head}{cleaned_body}{tail}"
        else:
            # Keep an empty line inside the block to avoid creating `from ... import ()`
            new_block = f"{head}{tail}"

        return hoisted_text + new_block

    new_content = pattern.sub(_repl, content)
    return new_content, changed


def process_file(path: Path, apply: bool) -> Tuple[bool, List[str]]:
    """Process a single file. Returns (changed, messages)."""
    messages: List[str] = []
    try:
        original = path.read_text(encoding="utf-8")
    except Exception as exc:
        return False, [f"Failed to read {path}: {exc}"]

    updated = original
    any_change = False

    # 1) Fix docstring quote lines
    updated, changed = fix_docstring_quote_lines(updated)
    if changed:
        any_change = True
        messages.append("fixed docstring quotes")

    # 2) Fix imports inside from-import blocks
    updated, changed = fix_imports_inside_from_blocks(updated)
    if changed:
        any_change = True
        messages.append("moved imports out of from-import blocks")

    if any_change and apply:
        try:
            path.write_text(updated, encoding="utf-8")
        except Exception as exc:
            return False, [f"Failed to write {path}: {exc}"]

    return any_change, messages


def find_python_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip common dirs
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".venv", "venv", "node_modules"}]
        for fname in filenames:
            if fname.endswith(".py"):
                files.append(Path(dirpath) / fname)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk syntax cleanup for docstrings and import blocks")
    parser.add_argument("--root", default="/workspace/src", help="Project root to process")
    parser.add_argument("--apply", action="store_true", help="Apply changes in place")
    args = parser.parse_args()

    root = Path(args.root)
    py_files = find_python_files(root)

    print(f"Scanning {len(py_files)} Python files under {root}...")
    changed_count = 0
    for p in py_files:
        changed, msgs = process_file(p, apply=args.apply)
        if changed:
            changed_count += 1
            print(f"Edited: {p}  ({'; '.join(msgs)})")

    print(f"\nSummary: {changed_count} files edited out of {len(py_files)} scanned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

