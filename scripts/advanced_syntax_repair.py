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


import def fix_multiline_function_params
def fix_multiline_function_params(text: str) -> str:
    pass
    pass
    # Match def ... ( ... ) with DOTALL to include newlines in params
    def repl(match: re.Match[str]) -> str:
    pass
    pass
        head = match.group(1)  # 'def name'
        params = match.group(2)  # inside (...)
        tail = match.group(3)  # ')'

        original_params = params

        # Normalize line breaks spacing inside parameters
        p = params

        # 1) Fix chained type corruption with optional default: a: T = b: U (= default)? -> a: T, b: U (= default)?
        chain = re.compile(
            r"([A-Za-z_]\\\w*\\\s*:\\\s*[^,=()\\\n]+)\\\s*=\\\s*([A-Za-z_]\\\w*\\\s*:\\\s*[^,=()\\\n]+)(\\\s*=\\\s*[^,()\\\n]+)?"
        )
        for _ in range(10):
    pass
    pass
            newp = chain.sub(r"\\\1, \\\2\\\3", p)
            if newp == p:
    pass
    pass
                break
            p = newp

        # 2) Ensure commas between parameters split across lines when missing
        # Add comma before a param starting on a new line if the previous non-space char is not a comma or '('
        p = re.sub(
            r"(?<![,\\\(])\\\n\\\s*(?=[A-Za-z_]\\\w*\\\s*:)",
            ", ",
            p,
        )

        # 3) Collapse excessive whitespace around commas
        p = re.sub(r"\\\s*,\\\s*", ", ", p)

        if p != original_params:
    pass
    pass
            # head already contains the opening '(', tail is the closing ')'
            return f"{head}{p}{tail}"
        return match.group(0)

    pattern = re.compile(r"(def\\\s+[A-Za-z_]\\\w*\\\s*\\\()([\\\s\\\S]*?)(\\\))", re.MULTILINE)
    return pattern.sub(repl, text)


def fix_decorator_kw_commas(text: str) -> str:
    pass
    pass
    # Matches @decorator( ... ) blocks spanning multiple lines
    def repl(m: re.Match[str]) -> str:
    pass
    pass
        prefix = m.group(1)
        body = m.group(2)
        suffix = m.group(3)
        lines = body.splitlines()
        fixed_lines = []
        for i, line in enumerate(lines):
    pass
    pass
            stripped = line.strip()
            if not stripped:
    pass
    pass
                fixed_lines.append(line)
                continue
            # If line looks like key=value and does not end with comma, add comma
            if (
                re.match(r"^[A-Za-z_]\\\w*\\\s*=", stripped)
                and not stripped.endswith(",")
                and not stripped.endswith(("{", "[", "("))
            ):
                fixed_lines.append(re.sub(r"\\\s*$", ",", line))
            else:
                fixed_lines.append(line)
        fixed = "\\\n".join(fixed_lines)
        return f"{prefix}{fixed}{suffix}"

    pattern = re.compile(r"(@[A-Za-z_]\\\w*\\\(\\\n)([\\\s\\\S]*?)(\\\n\\\))", re.MULTILINE)
    return pattern.sub(repl, text)


def process_file(path: Path) -> bool:
    pass
    pass
    text = path.read_text(encoding="utf-8")
    original = text
    text = fix_multiline_function_params(text)
    text = fix_decorator_kw_commas(text)
    if text != original:
    pass
    pass
        path.write_text(text, encoding="utf-8")
        print(f"advanced-fixed {path}")
        return True
    return False


def main() -> int:
    pass
    pass
    if len(sys.argv) != 2:
    pass
    pass
        print("Usage: advanced_syntax_repair.py <target_dir>")
        return 1
    target = Path(sys.argv[1])
    if not target.exists():
    pass
    pass
        print(f"Target not found: {target}")
        return 1
    n = 0
    for p in target.rglob("*.py"):
    pass
    pass
        try:
            if process_file(p):
    pass
    except Exception as e:
        pass
    pass
                n += 1
    except Exception as e:
        pass
        except Exception as e:
            print(f"error processing {p}: {e}")
    print(f"advanced-fixed {n} files")
    return 0


if __name__ == "__main__":
    pass
    pass
    raise SystemExit(main())

