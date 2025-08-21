#!/usr/bin/env python3
"""
Post-cleaner to repair advanced parameter/type corruption in src/training/steps.

Repairs:
- dict[str = Any] -> dict[str, Any] (and inside tuple[], list[] too)
- chained param corruption in function defs: `a: T = b: U` -> `a: T, b: U`
- simple default literals in function defs: `a: T, 123` -> `a: T = 123`
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def fix_bracket_types(text: str) -> str:
    # Replace ' = ' with ', ' inside type brackets for dict/tuple/list annotations
    def repl(m: re.Match[str]) -> str:
        head = m.group(1)
        inside = m.group(2)
        fixed = inside.replace(" = ", ", ")
        return f"{head}[{fixed}]"

    pattern = re.compile(r"\b(dict|tuple|list)\[([^\]]+)\]")
    return pattern.sub(repl, text)


def fix_function_params(line: str) -> str:
    if not line.lstrip().startswith("def "):
        return line
    try:
        start = line.index("(")
        end = line.rindex(")")
    except ValueError:
        return line
    params = line[start + 1 : end]

    # 1) Repair chained type corruption: name1: T1 = name2: T2 -> name1: T1, name2: T2
    patt_chain = re.compile(r"([A-Za-z_]\w*):\s*([^,=()]+)\s*=\s*([A-Za-z_]\w*):\s*([^,=()]+)")
    for _ in range(10):
        new_params = patt_chain.sub(r"\1: \2, \3: \4", params)
        if new_params == params:
            break
        params = new_params

    # 2) Default literal fixer: a: T, 123 -> a: T = 123 (only for literals/None/True/False/quoted)
    patt_default = re.compile(
        r"([A-Za-z_]\w*:\s*[^,=()]+)\s*,\s*(\d+(?:\.\d+)?|True|False|None|'[^']*'|\"[^\"]*\")(?=\s*(?:,|\)|$))"
    )
    params = patt_default.sub(r"\1 = \2", params)

    return line[: start + 1] + params + line[end:]


def process_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    # Apply bracket type fixes globally
    text = fix_bracket_types(text)

    # Fix function def parameter lists line-by-line
    lines = text.splitlines()
    lines = [fix_function_params(ln) for ln in lines]
    text = "\n".join(lines)

    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"post-fixed {path}")
        return True
    return False


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: post_steps_cleaner.py <target_dir>")
        return 1
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Target not found: {target}")
        return 1
    count = 0
    for p in target.rglob("*.py"):
        try:
            if process_file(p):
                count += 1
        except Exception as e:
            print(f"error in {p}: {e}")
    print(f"post-updated {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

