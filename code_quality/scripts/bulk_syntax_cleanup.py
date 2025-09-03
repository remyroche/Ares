#!/usr/bin/env python3
"""
Bulk syntax cleanup for malformed import fragments introduced by merges.

Fixes performed:
- Remove stray lines like "import," and "from,"
- In "from X import ..." lists, drop the reserved keyword "import"
  and deduplicate names; if no names remain, drop the line.
- Optionally normalize spacing.
"""

import re
from pathlib import Path
from typing import List


SRC_ROOT = Path("/workspace/src")


def clean_import_from_line(line: str) -> str | None:
    m = re.match(r"^(?P<indent>\s*)from\s+(?P<mod>[\w\.]+)\s+import\s+(?P<names>[^#\n]+)(?P<tail>.*)$", line)
    if not m:
        return line
    indent = m.group("indent")
    module = m.group("mod")
    names_part = m.group("names").strip()
    tail = m.group("tail")

    # Split names by comma outside parentheses
    raw = [n.strip() for n in names_part.split(",")]
    # Remove empty and the reserved keyword 'import'
    filtered: List[str] = [n for n in raw if n and n != "import"]
    # Deduplicate preserving order
    seen = set()
    final: List[str] = []
    for n in filtered:
        if n not in seen:
            seen.add(n)
            final.append(n)
    if not final:
        # Nothing valid remains; drop the line
        return None
    names_str = ", ".join(final)
    return f"{indent}from {module} import {names_str}{tail}"


def process_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    new_lines: List[str] = []
    changed = False
    for line in lines:
        # Remove stray syntax-only fragments
        if re.match(r"^\s*import,\s*$", line) or re.match(r"^\s*from,\s*$", line):
            changed = True
            continue
        cleaned = clean_import_from_line(line)
        if cleaned is None:
            changed = True
            continue
        if cleaned != line:
            changed = True
        new_lines.append(cleaned)
    if changed:
        path.write_text("\n".join(new_lines) + ("\n" if original.endswith("\n") else ""), encoding="utf-8")
    return changed


def main() -> None:
    py_files = list(SRC_ROOT.rglob("*.py"))
    modified = 0
    for f in py_files:
        try:
            if process_file(f):
                modified += 1
        except Exception as e:
            print(f"Warning: failed to process {f}: {e}")
    print(f"Cleanup complete. Modified {modified} files out of {len(py_files)}.")


if __name__ == "__main__":
    main()

