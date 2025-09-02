#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


def apply_fixes(text: str) -> str:
    original=text
    # 1) Remove double parentheses in function defs
    text=re.sub(r"(def\s+[A-Za-z_]\w*\s*)\(\(", r"\1(", text)
    text=re.sub(r"(async\s+def\s+[A-Za-z_]\w*\s*)\(\(", r"\1(", text)
    text=re.sub(r"\)\)\s*->", r") ->", text)
    text=re.sub(r"\)\)\s*:", r"):", text)
    text=re.sub(r"\(\,\s*", r"(", text)  # def f(, x: int) -> def f(x: int)

    # 2) Fix dict/list/tuple type brackets having ' = '
    def fix_brackets(m: re.Match[str]) -> str:
        head=m.group(1)
        inside=m.group(2)
        fixed=inside.replace(" = ", ", ")
        return f"{head}[{fixed}]"

    text=re.sub(r"\b(dict|list|tuple)\[([^\]]+)\]", fix_brackets, text)

    # 3) Fix typed variable annotations using comma instead of equals
    # name: Type, value -> name: Type=value
    text = re.sub(
        r"^(\s*)([A-Za-z_][\w\.]*)\s*:\s*([^,\n]+?)\s*,\s*(.+)$",
        r"\1\2: \3=\4",
        text,
        flags=re.MULTILINE,
    )

    return text if text != original else text


def process(path: Path) -> bool:
    s=path.read_text(encoding="utf-8")
    ns=apply_fixes(s)
    if ns != s:
        path.write_text(ns, encoding="utf-8")
        print(f"surgical {path}")
        return True
    return False


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: surgical_pass2.py <target_dir>")
        return 1
    target=Path(sys.argv[1])
    n=0
    for p in target.rglob("*.py"):
        try:
            if process(p):
                n += 1
        except Exception as e:
            print(f"err {p}: {e}")
    print(f"surgical-modified {n} files")
    return 0


if __name__== "__main__":
    raise SystemExit(main())

