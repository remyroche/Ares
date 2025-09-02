#!/usr/bin/env python3
"""
Simple syntax checker for Python files.
"""

import ast
import os
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to compile the code
        compile(content, file_path, 'exec')
        
        # Try to parse with AST
        ast.parse(content)
        
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Main function to check all Python files."""
    workspace_path = Path(".")
    python_files = list(workspace_path.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files")
    print("=" * 50)
    
    files_with_issues = []
    total_files = 0
    
    for file_path in python_files:
        # Skip the code_quality directory for now
        if "code_quality" in str(file_path):
            continue
            
        total_files += 1
        is_valid, error = check_syntax(file_path)
        
        if not is_valid:
            files_with_issues.append((file_path, error))
            print(f"❌ {file_path}: {error}")
        else:
            print(f"✅ {file_path}")
    
    print("=" * 50)
    print(f"Total files checked: {total_files}")
    print(f"Files with syntax issues: {len(files_with_issues)}")
    
    if files_with_issues:
        print("\nFiles with syntax issues:")
        for file_path, error in files_with_issues:
            print(f"  - {file_path}: {error}")
        return 1
    else:
        print("\nAll Python files have valid syntax! 🎉")
        return 0

if __name__ == "__main__":
    sys.exit(main())