#!/usr/bin/env python3
"""
Advanced syntax repair for src/training/steps.

Repairs:
- Multiline function parameter lists: insert missing commas between parameters
  and fix chained type corruption like `a: T = b: U` (with optional default).
- Decorator argument lists: insert missing commas between keyword args.
"""

import re
import sys
from pathlib import Path


def fix_multiline_function_params(text: str) -> str:
    # Match def ... ( ... ) with DOTALL to include newlines in params
    pattern = re.compile(r"(def\s+[A-Za-z_]\w*\s*\()([\s\S]*?)(\))", re.MULTILINE)
    return pattern.sub(repl, text)


def fix_decorator_kw_commas(text: str) -> str:
    # Matches @decorator( ... ) blocks spanning multiple lines
    pattern = re.compile(r"(@[A-Za-z_]\w*\(\n)([\s\S]*?)(\n\))", re.MULTILINE)
    return pattern.sub(repl, text)


def process_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    text = fix_multiline_function_params(text)
    text = fix_decorator_kw_commas(text)
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"advanced-fixed {path}")
        return True
    return False


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: advanced_syntax_repair.py <target_dir>")
        return 1
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Target not found: {target}")
        return 1
    n = 0
    for p in target.rglob("*.py"):
        try:
            if process_file(p):
                n += 1
        except Exception as e:
            print(f"error processing {p}: {e}")
    print(f"advanced-fixed {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

