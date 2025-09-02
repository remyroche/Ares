#!/usr/bin/env python3
"""
Simple script to check Python syntax errors across all Python files.
"""

import os
import sys
import py_compile
from pathlib import Path

def check_file_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function to check all Python files."""
    root_dir=Path(".")
    python_files=list(root_dir.rglob("*.py"))
    
    print(f"Checking syntax for {len(python_files)} Python files...")
    print("=" * 60)
    
    errors=[]
    valid_files = 0
    
    for file_path in python_files:
        # Skip the code_quality directory and virtual environment
        if "code_quality" in str(file_path) or "code_quality_env" in str(file_path):
            continue
            
        is_valid, error_msg=check_file_syntax(file_path)
        
        if is_valid:
            valid_files += 1
        else:
            errors.append((file_path, error_msg))
            print(f"❌ {file_path}: {error_msg}")
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total files: {len(python_files)}")
    print(f"  Valid files: {valid_files}")
    print(f"  Files with errors: {len(errors)}")
    
    if errors:
        print(f"\nFiles with syntax errors:")
        for file_path, error_msg in errors:
            print(f"  {file_path}: {error_msg}")
        return 1
    else:
        print(f"\n✅ All Python files have valid syntax!")
        return 0

if __name__== "__main__":
    sys.exit(main())