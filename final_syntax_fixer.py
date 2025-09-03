#!/usr/bin/env python3
"""Final comprehensive syntax fixer for all remaining errors."""

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple


def run_syntax_fixer_tool():
    """Run the advanced syntax fixer tool from code_quality."""
    print("Running advanced syntax fixer tool...")
    
    cmd = [
        "python3",
        "/workspace/code_quality/scripts/advanced_syntax_fixer.py",
        "--project-root", "/workspace/src",
        "--fix"  # Actually fix the files
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("Tool output:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


def count_syntax_errors():
    """Count files with syntax errors."""
    src_dir = Path('/workspace/src')
    error_count = 0
    
    for py_file in src_dir.rglob('*.py'):
        result = subprocess.run(
            ['python3', '-m', 'py_compile', str(py_file)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            error_count += 1
    
    return error_count


def main():
    """Main function."""
    # Get initial count
    initial_count = count_syntax_errors()
    print(f"\nInitial files with syntax errors: {initial_count}")
    
    # Run the advanced syntax fixer tool
    if run_syntax_fixer_tool():
        print("\n✓ Advanced syntax fixer completed successfully")
    else:
        print("\n✗ Advanced syntax fixer encountered errors")
    
    # Get final count
    final_count = count_syntax_errors()
    print(f"\nFinal files with syntax errors: {final_count}")
    print(f"Fixed: {initial_count - final_count} files")
    
    # Generate detailed report
    if final_count > 0:
        print(f"\n{final_count} files still have syntax errors")
        print("Run 'python3 get_syntax_errors.py' for details")


if __name__ == "__main__":
    main()