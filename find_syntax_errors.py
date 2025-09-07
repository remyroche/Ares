#!/usr/bin/env python3
"""Find all Python files with syntax errors."""

import subprocess
import sys
from pathlib import Path

def find_syntax_errors():
    """Find all Python files with syntax errors."""
    project_root = Path(__file__).parent
    error_files = []
    
    # Find all Python files
    python_files = list(project_root.rglob("*.py"))
    
    print(f"Checking {len(python_files)} Python files for syntax errors...")
    
    for file_path in python_files:
        try:
            pass
            result = subprocess.run(
                ['python3', '-m', 'py_compile', str(file_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                error_files.append(str(file_path))
                print(f"❌ {file_path}")
                # Print first line of error
                if result.stderr:
                    error_lines = result.stderr.strip().split('\n')
                    if error_lines:
                        print(f"   {error_lines[0]}")
        except Exception as e:
            print(f"Error checking {file_path}: {e}")
    
    print(f"\nFound {len(error_files)} files with syntax errors:")
    for file_path in error_files:
        print(f"  - {file_path}")
    
    return error_files

if __name__ == "__main__":
    find_syntax_errors()
