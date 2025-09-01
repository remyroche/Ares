#!/usr/bin/env python3
"""
Bulk fixer for corrupted syntax patterns in src/training/steps.

Fixes:
    pass  # TODO: Add implementation
- logger initialization using comma to equals
- typed variable annotations using comma to equals (var: Type, default -> var: Type = default)
- Optional annotations with None (Type | None, None -> Type | None = None)
- self.variable comma assignment (self.x, expr -> self.x = expr)
- function parameter defaults where comma is used (def f(a: T, default) -> def f(a: T = default))
"""


import re
import sys
from pathlib import Path


def fix_line_assignments(line: str) -> str:
    # logger = system_logger.getChild(...)
    line = re.sub(
        r"^(\s*)(self\.logger|logger)\s*,\s*(system_logger\.getChild\(.*)$",
        r"\1\2 = \3",
        line,
    )

    # typed var: name: Type, default -> name: Type = default (skip def lines)
    if not line.lstrip().startswith("def "):
        m = re.match(r"^(\s*)([A-Za-z_][\w\.]*)\s*:\s*(.+?)\s*,\s*(.+)$", line)
        if m and "=" not in line.split(":", 1)[1].split(",", 1)[0]:
            indent, var, vartype, value = m.groups()
            return f"{indent}{var}: {vartype.strip()} = {value}"

    # self.x, expr -> self.x = expr (skip control/def/class/import lines)
    stripped = line.lstrip()
    if not re.match(r"^(return|for |if |elif |else:|while |with |def |class |from |import |async |await|raise|yield|try:|except|finally|@|#)", stripped):
        m2 = re.match(r"^(\s*)(self\.[A-Za-z_][\w]*)\s*,\s*(.+)$", line)
        if m2:
            indent, var, value = m2.groups()
            return f"{indent}{var} = {value}"

        # Single name without '=': var, expr -> var = expr
        if "=" not in line and "," in line and "(" in line and line.rstrip().endswith(")"):
            last_comma = line.rfind(",")
            lhs = line[:last_comma].strip()
            rhs = line[last_comma + 1 :].strip()
            if re.match(r"^[A-Za-z_]\w*(\s*,\s*[A-Za-z_]\w*)*$", lhs):
                indent = re.match(r"^\s*", line).group(0)
                return f"{indent}{lhs} = {rhs}"

        # Generic: var, expr -> var = expr (no '=' and not a tuple unpack since no '=')
        if "=" not in line and "," in line and "(" not in line:
            m3 = re.match(r"^(\s*)([A-Za-z_]\w*)\s*,\s*(.+)$", line)
            if m3:
                indent, varname, value = m3.groups()
                return f"{indent}{varname} = {value}"

    # Remove trailing commas after assignments
    mtrail = re.match(r"^(\s*[^#\n=]+=[^#\n]+),\s*$", line)
    if mtrail:
        return mtrail.group(1)

    return line


def fix_function_def_params(line: str) -> str:
    """Within function definitions, replace ': T, default' to ': T = default'."""
    if not line.lstrip().startswith("def "):
        return line

    # Find parameter segment between first '(' and last ')'
    try:
        start = line.index('(')
        end = line.rindex(')')
    except ValueError:
        return line

    params = line[start + 1 : end]

    # Iteratively replace occurrences
    prev = None
    current = params
    pattern = re.compile(r"(:\s*[^=,()]+)\s*,\s*([^,()\s]+)")
    for _ in range(10):  # avoid infinite loops
        new = pattern.sub(lambda m: f"{m.group(1)} = {m.group(2)}", current)
        if new == current:
            break
        current = new

    if current == params:
        return line
    return line[: start + 1] + current + line[end:]


def process_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    lines = text.splitlines()
    fixed_lines = []
    for line in lines:
        line2 = fix_line_assignments(line)
        line3 = fix_function_def_params(line2)
        fixed_lines.append(line3)

    new_text = "\n".join(fixed_lines)
    if new_text != original:
        path.write_text(new_text, encoding="utf-8")
        print(f"fixed {path}")
        return True
    return False


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: bulk_steps_fixer.py <target_dir>")
        return 1
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Target not found: {target}")
        return 1
    total = 0
    for p in target.rglob("*.py"):
        try:
            if process_file(p):
                total += 1
        except Exception as e:
            print(f"error fixing {p}: {e}")
    print(f"updated {total} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

