#!/usr/bin/env python3
"""
Focused syntax checker for core directories (src/, analysis/).
"""

import ast
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
    """Main function to check core Python files."""
    core_dirs = ["src", "analysis"]
    python_files = []
    
    for core_dir in core_dirs:
        if Path(core_dir).exists():
            python_files.extend(Path(core_dir).rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files in core directories")
    print("=" * 60)
    
    files_with_issues = []
    total_files = 0
    
    for file_path in python_files:
        total_files += 1
        is_valid, error = check_syntax(file_path)
        
        if not is_valid:
            files_with_issues.append((file_path, error))
            print(f"❌ {file_path}: {error}")
        else:
            print(f"✅ {file_path}")
    
    print("=" * 60)
    print(f"Total files checked: {total_files}")
    print(f"Files with syntax issues: {len(files_with_issues)}")
    
    if files_with_issues:
        print("\nFiles with syntax issues in core directories:")
        for file_path, error in files_with_issues:
            print(f"  - {file_path}: {error}")
        return 1
    else:
        print("\nAll core Python files have valid syntax! 🎉")
        return 0

if __name__ == "__main__":
    sys.exit(main())