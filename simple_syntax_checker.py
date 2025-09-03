#!/usr/bin/env python3
"""
Simple syntax checker for Python files without external dependencies.
Uses only built-in Python modules.
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any


def check_syntax(file_path: str) -> dict[str, Any]:
    """Check syntax of a single Python file."""
    result = {
        "file": file_path,
        "valid": True,
        "errors": [],
    }

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Try to parse with AST
        try:
            ast.parse(content)
        except SyntaxError as e:
            result["valid"] = False
            result["errors"].append({
                "type": "SyntaxError",
                "line": e.lineno,
                "offset": e.offset,
                "message": str(e.msg),
                "text": e.text.strip() if e.text else "",
            })

        # Try to compile
        try:
            compile(content, file_path, "exec")
        except Exception as e:
            if not isinstance(e, SyntaxError):  # Avoid duplicates
                result["valid"] = False
                result["errors"].append({
                    "type": type(e).__name__,
                    "message": str(e),
                })

    except Exception as e:
        result["valid"] = False
        result["errors"].append({
            "type": "FileError",
            "message": f"Could not read file: {str(e)}",
        })

    return result


def find_python_files(directory: str, exclude_patterns: list[str] = None) -> list[str]:
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


def main():
    """Main function to check syntax across the repository."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Python syntax checker")
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
    syntax_errors = 0

    for file_path in files_to_check:
        result = check_syntax(file_path)
        results.append(result)

        if not result["valid"]:
            syntax_errors += 1
            if not args.json:
                print(f"\n❌ {file_path}")
                for error in result["errors"]:
                    print(f"   Line {error.get('line', '?')}: {error['message']}")
                    if error.get("text"):
                        print(f"   >>> {error['text']}")

    # Summary
    if not args.json:
        print(f"\n{'='*60}")
        print(f"Total files checked: {len(files_to_check)}")
        print(f"Files with syntax errors: {syntax_errors}")
        print(f"Valid files: {len(files_to_check) - syntax_errors}")

        if syntax_errors == 0:
            print("\n✅ All files have valid syntax!")
        else:
            print(f"\n❌ Found syntax errors in {syntax_errors} files")

    # Output results
    output_data = {
        "summary": {
            "total_files": len(files_to_check),
            "valid_files": len(files_to_check) - syntax_errors,
            "invalid_files": syntax_errors,
        },
        "results": results,
    }

    if args.json:
        print(json.dumps(output_data, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0 if syntax_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
