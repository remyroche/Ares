#!/usr/bin/env python3
"""
Simple import checker for Python files without external dependencies.
Uses only built-in Python modules.
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Dict, Set


def find_python_files(directory: str, exclude_patterns: List[str] = None) -> List[str]:
    """Find all Python files in a directory."""
    exclude_patterns = exclude_patterns or [
        "__pycache__", ".git", "venv", "env", "node_modules",
        ".pytest_cache", ".mypy_cache", ".tox", "build", "dist",
    ]

    python_files = []

    for root, dirs, files in os.walk(directory):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                python_files.append(file_path)

    return python_files


def check_imports(file_path: str) -> Dict[str, Any]:
    """Check imports in a single Python file."""
    result = {
        "file": file_path,
        "imports": [],
        "import_errors": [],
        "undefined_names": [],
        "unused_imports": [],
    }

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Parse the file
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Skip files with syntax errors - they're handled by syntax checker
            result["import_errors"].append("File has syntax errors - skipping import analysis")
            return result

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    result["imports"].append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                    })

        # Basic undefined name detection (simplified)
        defined_names = set()
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.name)

        # Find potentially undefined names (very basic check)
        for name in used_names:
            if name not in defined_names and not name.startswith('_'):
                # Check if it's likely an imported name
                imported_names = set()
                for imp in result["imports"]:
                    if imp["type"] == "import":
                        imported_names.add(imp["alias"] or imp["module"].split(".")[0])
                    elif imp["type"] == "from_import":
                        imported_names.add(imp["alias"] or imp["name"])
                
                if name not in imported_names and name not in ['print', 'len', 'str', 'int', 'float', 'list', 'dict', 'tuple', 'set', 'bool', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr', 'delattr', 'dir', 'vars', 'locals', 'globals', 'open', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed', 'sum', 'min', 'max', 'abs', 'round', 'pow', 'divmod', 'bin', 'hex', 'oct', 'chr', 'ord', 'repr', 'ascii', 'format', 'input', 'eval', 'exec', 'compile', 'hash', 'id', 'iter', 'next', 'all', 'any', 'callable', 'issubclass', 'super', 'property', 'staticmethod', 'classmethod', 'object', 'Exception', 'BaseException', 'ValueError', 'TypeError', 'AttributeError', 'KeyError', 'IndexError', 'ImportError', 'ModuleNotFoundError', 'NameError', 'SyntaxError', 'IndentationError', 'TabError', 'UnicodeError', 'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeTranslateError', 'OSError', 'IOError', 'FileNotFoundError', 'PermissionError', 'ProcessLookupError', 'TimeoutError', 'ConnectionError', 'BrokenPipeError', 'ConnectionAbortedError', 'ConnectionRefusedError', 'ConnectionResetError', 'BlockingIOError', 'ChildProcessError', 'FileExistsError', 'FileNotFoundError', 'IsADirectoryError', 'NotADirectoryError', 'PermissionError', 'ProcessLookupError', 'TimeoutError', 'InterruptedError', 'NotImplementedError', 'ArithmeticError', 'FloatingPointError', 'OverflowError', 'ZeroDivisionError', 'AssertionError', 'AttributeError', 'BufferError', 'EOFError', 'ImportError', 'LookupError', 'MemoryError', 'NameError', 'OSError', 'ReferenceError', 'RuntimeError', 'StopIteration', 'StopAsyncIteration', 'SyntaxError', 'SystemError', 'SystemExit', 'TabError', 'TypeError', 'UnboundLocalError', 'UnicodeError', 'ValueError', 'Warning', 'UserWarning', 'DeprecationWarning', 'PendingDeprecationWarning', 'SyntaxWarning', 'RuntimeWarning', 'FutureWarning', 'ImportWarning', 'UnicodeWarning', 'BytesWarning', 'ResourceWarning']:
                    # Find the node that uses this name
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
                            result["undefined_names"].append({
                                "name": name,
                                "line": node.lineno,
                            })
                            break

    except Exception as e:
        result["import_errors"].append(f"Error analyzing imports: {str(e)}")

    return result


def main():
    """Main function to check imports across the repository."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Python import checker")
    parser.add_argument("path", nargs="?", default=".", help="Path to check (file or directory)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    # Determine if path is file or directory
    path = Path(args.path)

    if path.is_file():
        files_to_check = [str(path)]
    elif path.is_dir():
        print(f"Scanning directory: {path}")
        files_to_check = find_python_files(str(path))
        print(f"Found {len(files_to_check)} Python files")
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)

    # Check all files
    results = []
    files_with_import_issues = 0
    total_import_errors = 0
    total_undefined_names = 0

    for file_path in files_to_check:
        result = check_imports(file_path)
        results.append(result)

        if result["import_errors"] or result["undefined_names"]:
            files_with_import_issues += 1
            total_import_errors += len(result["import_errors"])
            total_undefined_names += len(result["undefined_names"])
            
            if not args.json:
                print(f"\n❌ {file_path}")
                for error in result["import_errors"]:
                    print(f"   Import Error: {error}")
                for undefined in result["undefined_names"]:
                    print(f"   Undefined: {undefined['name']} (line {undefined['line']})")

    # Summary
    if not args.json:
        print(f"\n{'='*60}")
        print(f"Total files checked: {len(files_to_check)}")
        print(f"Files with import issues: {files_with_import_issues}")
        print(f"Total import errors: {total_import_errors}")
        print(f"Total undefined names: {total_undefined_names}")

        if files_with_import_issues == 0:
            print("\n✅ No import issues found!")
        else:
            print(f"\n❌ Found import issues in {files_with_import_issues} files")

    # Output results
    output_data = {
        "summary": {
            "total_files": len(files_to_check),
            "files_with_import_issues": files_with_import_issues,
            "total_import_errors": total_import_errors,
            "total_undefined_names": total_undefined_names,
        },
        "results": results,
    }

    if args.json:
        print(json.dumps(output_data, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0 if files_with_import_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())