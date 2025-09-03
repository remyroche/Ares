#!/usr/bin/env python3
"""
Lightweight analyzer for placeholders, silent exception passes, and potential dead/unused code.
Uses only the Python standard library so it can run without external dependencies.

Outputs a JSON report with:
- placeholders: functions/methods that raise NotImplementedError or are empty (pass-only)
- todos: lines containing TODO/FIXME
- silent_except_pass: except Exception: pass occurrences
- potential_dead_defs: module-level functions/classes never referenced elsewhere in the tree
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


class FileAnalysis(ast.NodeVisitor):
    def __init__(self, source: str, filename: str):
        self.source = source
        self.filename = filename
        self.placeholders: list[dict[str, Any]] = []
        self.silent_except_pass: list[dict[str, Any]] = []
        self.module_level_defs: list[tuple[str, str, int]] = []  # (name, kind, line)
        self.name_loads: list[tuple[str, int]] = []  # (name, line)

        self._tree = ast.parse(source, filename=filename)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Module-level def detection
        if isinstance(getattr(node, "parent", None), ast.Module):
            self.module_level_defs.append((node.name, "function", node.lineno))

        # Placeholder: only pass statements in body
        if node.body and all(isinstance(stmt, ast.Pass) for stmt in node.body):
            self.placeholders.append(
                {
                    "file": self.filename,
                    "name": node.name,
                    "kind": "function",
                    "line": node.lineno,
                    "reason": "empty_function_pass",
                },
            )

        # Placeholder: raises NotImplementedError anywhere in the body
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Raise):
                exc = stmt.exc
                if isinstance(exc, ast.Name) and exc.id == "NotImplementedError" or isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                    self.placeholders.append(
                        {
                            "file": self.filename,
                            "name": node.name,
                            "kind": "function",
                            "line": getattr(stmt, "lineno", node.lineno),
                            "reason": "raises_NotImplementedError",
                        },
                    )

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        # Module-level class def detection
        if isinstance(getattr(node, "parent", None), ast.Module):
            self.module_level_defs.append((node.name, "class", node.lineno))

        # Placeholder: class body only pass
        if node.body and all(isinstance(stmt, ast.Pass) for stmt in node.body):
            self.placeholders.append(
                {
                    "file": self.filename,
                    "name": node.name,
                    "kind": "class",
                    "line": node.lineno,
                    "reason": "empty_class_pass",
                },
            )

        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        # Silent except: except Exception: pass
        is_exception = isinstance(node.type, ast.Name) and node.type.id == "Exception"
        only_pass = len(node.body) > 0 and all(isinstance(s, ast.Pass) for s in node.body)
        if is_exception and only_pass:
            self.silent_except_pass.append(
                {
                    "file": self.filename,
                    "line": node.lineno,
                    "context": "except Exception: pass",
                },
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.name_loads.append((node.id, node.lineno))

    def analyze(self) -> dict[str, Any]:
        # Set parent links for module-level detection
        for parent in ast.walk(self._tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent

        self.visit(self._tree)

        return {
            "placeholders": self.placeholders,
            "silent_except_pass": self.silent_except_pass,
            "module_level_defs": self.module_level_defs,
            "name_loads": self.name_loads,
        }


def scan_directory(root: Path) -> dict[str, Any]:
    files = sorted(root.rglob("*.py"))
    results: dict[str, Any] = {
        "files": {},
        "todos": [],
        "placeholders": [],
        "silent_except_pass": [],
        "potential_dead_defs": [],
    }

    all_load_names: set[str] = set()
    module_defs: list[tuple[str, str, int, str]] = []  # (name, kind, line, file)

    for path in files:
        try:
            text = read_text(path)
        except Exception:
            # Skip unreadable files
            continue

        # Collect TODO/FIXME
        for idx, line in enumerate(text.splitlines(), start=1):
            if "TODO" in line or "FIXME" in line:
                results["todos"].append({
                    "file": str(path),
                    "line": idx,
                    "text": line.strip(),
                })

        try:
            fa = FileAnalysis(text, str(path))
            analyzed = fa.analyze()
        except Exception:
            # If AST fails, move on
            continue

        results["files"][str(path)] = analyzed
        results["placeholders"].extend(analyzed["placeholders"])
        results["silent_except_pass"].extend(analyzed["silent_except_pass"])

        for name, kind, line in analyzed["module_level_defs"]:
            module_defs.append((name, kind, line, str(path)))

        for name, _ in analyzed["name_loads"]:
            all_load_names.add(name)

    # Potential dead defs: names never loaded anywhere
    for name, kind, line, file in module_defs:
        if name not in all_load_names:
            results["potential_dead_defs"].append({
                "file": file,
                "name": name,
                "kind": kind,
                "line": line,
                "reason": "never_referenced_in_tree",
            })

    return results


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Scan for placeholders and potential dead code")
    parser.add_argument("target", help="Directory to scan")
    parser.add_argument("--out", dest="out", help="Output JSON path", default="analyst_scan.json")
    args = parser.parse_args(argv)

    root = Path(args.target)
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        return 1

    results = scan_directory(root)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved scan report to {out_path}")
    print(f"Placeholders: {len(results['placeholders'])}")
    print(f"Silent except-pass: {len(results['silent_except_pass'])}")
    print(f"Potential dead defs: {len(results['potential_dead_defs'])}")
    print(f"TODO/FIXME: {len(results['todos'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

