#!/usr/bin/env python3
"""
Comprehensive Code Quality Tools Runner
Finds compilable Python files and runs syntax fixing, unused import removal, and dead code removal.
"""

import os
import ast
import subprocess
import sys
from pathlib import Path


def check_file_compiles(filepath):
    """Check if a Python file compiles without syntax errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True
    except (SyntaxError, IndentationError, UnicodeDecodeError):
        return False
    except Exception:
        return False


def find_compilable_files(directory):
    """Find all Python files that compile successfully."""
    compilable_files = []
    directory_path = Path(directory)

    for py_file in directory_path.rglob("*.py"):
        if check_file_compiles(py_file):
            compilable_files.append(str(py_file))

    return compilable_files


def run_syntax_fixer(directory):
    """Run the syntax fixer on the directory."""
    print("\n=== Running Syntax Fixer ===")
    try:
        result = subprocess.run([
            sys.executable, "code_quality/tools/syntax_fixer.py",
            directory, "--no-dry-run"
        ], capture_output=True, text=True)

        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"Exit code: {result.returncode}")

    except Exception as e:
        print(f"Error running syntax fixer: {e}")


def run_unused_import_cleaner(files):
    """Run the unused import cleaner on specific files."""
    print("\n=== Running Unused Import Cleaner ===")

    if not files:
        print("No compilable files found for import cleaning.")
        return

    print(f"Processing {len(files)} compilable files...")

    processed_count = 0
    cleaned_count = 0

    for filepath in files:
        try:
            # First run in dry-run mode to see what would be cleaned
            result = subprocess.run([
                sys.executable, "code_quality/tools/batch_import_cleaner.py",
                filepath, "--dry-run"
            ], capture_output=True, text=True)

            if "Would remove line" in result.stdout:
                print(f"\nFound unused imports in: {filepath}")
                print(result.stdout)

                # Now actually clean the imports
                result_clean = subprocess.run([
                    sys.executable, "code_quality/tools/batch_import_cleaner.py",
                    filepath
                ], capture_output=True, text=True)

                if result_clean.returncode == 0:
                    cleaned_count += 1
                    print(f"Cleaned imports in: {filepath}")

            processed_count += 1

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print(f"\nImport cleaning summary:")
    print(f"Files processed: {processed_count}")
    print(f"Files cleaned: {cleaned_count}")


def run_dead_code_remover(files):
    """Run the dead code remover on specific files."""
    print("\n=== Running Dead Code Remover ===")

    if not files:
        print("No compilable files found for dead code removal.")
        return

    print(f"Processing {len(files)} compilable files...")

    processed_count = 0
    cleaned_count = 0
    total_removals = 0

    for filepath in files:
        try:
            # First run in dry-run mode (default) to see what would be removed
            result = subprocess.run([
                sys.executable, "code_quality/tools/dead_code_remover.py",
                filepath
            ], capture_output=True, text=True)

            if "Would remove line" in result.stdout:
                print(f"\nFound dead code in: {filepath}")
                print(result.stdout)

                # Now actually remove the dead code
                result_clean = subprocess.run([
                    sys.executable, "code_quality/tools/dead_code_remover.py",
                    filepath, "--no-dry-run"
                ], capture_output=True, text=True)

                if result_clean.returncode == 0:
                    cleaned_count += 1
                    print(f"Removed dead code from: {filepath}")

            processed_count += 1

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print(f"\nDead code removal summary:")
    print(f"Files processed: {processed_count}")
    print(f"Files cleaned: {cleaned_count}")


def main():
    """Main function to run all code quality tools."""
    src_directory = "src"

    if not os.path.exists(src_directory):
        print(f"Directory {src_directory} not found!")
        return

    print("=== Code Quality Tools Runner ===")
    print(f"Analyzing directory: {src_directory}")

    # Step 1: Run syntax fixer first
    run_syntax_fixer(src_directory)

    # Step 2: Find all compilable files after syntax fixing
    print(f"\n=== Finding Compilable Files ===")
    compilable_files = find_compilable_files(src_directory)
    print(f"Found {len(compilable_files)} compilable Python files")

    if len(compilable_files) > 0:
        print("Sample compilable files:")
        for i, filepath in enumerate(compilable_files[:10]):
            print(f"  {i+1}. {filepath}")
        if len(compilable_files) > 10:
            print(f"  ... and {len(compilable_files) - 10} more")

    # Step 3: Run unused import cleaner on compilable files
    run_unused_import_cleaner(compilable_files)

    # Step 4: Run dead code remover on compilable files
    run_dead_code_remover(compilable_files)

    print("\n=== Code Quality Analysis Complete ===")
    print(f"Total compilable files analyzed: {len(compilable_files)}")


if __name__ == "__main__":
    main()