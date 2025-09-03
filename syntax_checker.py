#!/usr/bin/env python3
"""
Simple Python syntax checker that identifies syntax issues in Python files.
"""

import ast
import os
import sys
import tokenize
from pathlib import Path
from typing import Any


class SyntaxChecker:
    """Check Python files for syntax errors."""

    def __init__(self):
        self.syntax_errors = []
        self.valid_files = []
        self.invalid_files = []

    def check_file(self, file_path: str) -> tuple[bool, list[str]]:
        """
        Check a single Python file for syntax errors.

        Args:
            file_path: Path to the Python file

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            # Try to parse with AST
            with open(file_path, encoding="utf-8") as f:
                source = f.read()

            # Check if file is empty
            if not source.strip():
                return True, ["File is empty"]

            # Parse with AST
            ast.parse(source)

            # Try to tokenize
            with open(file_path, "rb") as f:
                list(tokenize.tokenize(f.readline))

            return True, []

        except SyntaxError as e:
            error_msg = f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset or 'unknown'}"
            errors.append(error_msg)
            return False, errors

        except UnicodeDecodeError as e:
            error_msg = f"UnicodeDecodeError: {e}"
            errors.append(error_msg)
            return False, errors

        except Exception as e:
            error_msg = f"Error: {e}"
            errors.append(error_msg)
            return False, errors

    def check_directory(self, directory: str) -> dict[str, Any]:
        """
        Check all Python files in a directory.

        Args:
            directory: Directory to check

        Returns:
            Dictionary with results
        """
        directory_path = Path(directory)
        python_files = list(directory_path.rglob("*.py"))

        print(f"Checking {len(python_files)} Python files for syntax issues...")

        for file_path in python_files:
            try:
                is_valid, errors = self.check_file(str(file_path))

                if is_valid:
                    self.valid_files.append(str(file_path))
                else:
                    self.invalid_files.append(str(file_path))
                    for error in errors:
                        self.syntax_errors.append({
                            "file": str(file_path),
                            "error": error,
                        })

            except Exception as e:
                print(f"Error checking {file_path}: {e}")

        return self._generate_summary()

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary of results."""
        return {
            "total_files": len(self.valid_files) + len(self.invalid_files),
            "valid_files": len(self.valid_files),
            "invalid_files": len(self.invalid_files),
            "total_errors": len(self.syntax_errors),
            "syntax_errors": self.syntax_errors,
            "valid_file_list": self.valid_files,
            "invalid_file_list": self.invalid_files,
        }


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python3 syntax_checker.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    checker = SyntaxChecker()
    results = checker.check_directory(directory)

    print("\n" + "="*60)
    print("SYNTAX CHECK RESULTS")
    print("="*60)
    print(f"Total Python files: {results['total_files']}")
    print(f"Valid files: {results['valid_files']}")
    print(f"Files with syntax issues: {results['invalid_files']}")
    print(f"Total syntax errors: {results['total_errors']}")

    if results["syntax_errors"]:
        print("\n" + "="*60)
        print("FILES WITH SYNTAX ISSUES:")
        print("="*60)

        # Group errors by file
        errors_by_file = {}
        for error in results["syntax_errors"]:
            file_path = error["file"]
            if file_path not in errors_by_file:
                errors_by_file[file_path] = []
            errors_by_file[file_path].append(error["error"])

        for file_path, errors in errors_by_file.items():
            print(f"\n{file_path}:")
            for error in errors:
                print(f"  - {error}")
    else:
        print("\n✅ All Python files have valid syntax!")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
